"""Measurements that replace the planner's estimates with real numbers.

The placement planner works from nameplate bandwidths and a memory-bound cost
model. That is good enough to choose a layout, but the only way to know what
this machine actually does is to measure it -- particularly the host-to-device
link, which is the single number that decides whether streaming a layer from
RAM is viable, and which depends on the slot the card ended up in.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Optional

__all__ = ["run_benchmarks", "bench_link_bandwidth", "bench_kernels",
           "bench_decode", "bench_host_memory", "bench_cpu_kernels"]


def _sync(device: Any) -> None:
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _timeit(fn, iters: int, warmup: int, device: Any) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def bench_link_bandwidth(size_mb: int = 512, iters: int = 5) -> list[dict]:
    """Pinned host <-> device copy rate, per GPU, both directions.

    Pinned memory is used deliberately: that is what the streaming path uses,
    and a pageable-memory number would understate the link by roughly half.
    """
    import torch
    out = []
    if not torch.cuda.is_available():
        return out
    n = size_mb * 1024 * 1024 // 4
    host = torch.empty(n, dtype=torch.float32).pin_memory()
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        try:
            dst = torch.empty(n, dtype=torch.float32, device=dev)
        except torch.cuda.OutOfMemoryError:
            continue
        h2d = _timeit(lambda: dst.copy_(host, non_blocking=True), iters, 2, dev)
        d2h = _timeit(lambda: host.copy_(dst, non_blocking=True), iters, 2, dev)
        gb = size_mb / 1024
        props = torch.cuda.get_device_properties(i)
        out.append({
            "device": f"cuda:{i}", "name": props.name,
            "h2d_gb_s": round(gb / h2d, 1),
            "d2h_gb_s": round(gb / d2h, 1),
        })
        del dst
        torch.cuda.empty_cache()
    return out


def bench_vram_bandwidth(size_mb: int = 1024, iters: int = 20) -> list[dict]:
    """Device-local read bandwidth, as a large contiguous copy."""
    import torch
    out = []
    if not torch.cuda.is_available():
        return out
    n = size_mb * 1024 * 1024 // 2
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        try:
            src = torch.empty(n, dtype=torch.float16, device=dev)
            dst = torch.empty_like(src)
        except torch.cuda.OutOfMemoryError:
            continue
        t = _timeit(lambda: dst.copy_(src), iters, 5, dev)
        # a copy reads once and writes once
        out.append({"device": f"cuda:{i}",
                    "copy_gb_s": round(2 * size_mb / 1024 / t, 1)})
        del src, dst
        torch.cuda.empty_cache()
    return out


def bench_host_memory(size_mb: int = 512, iters: int = 5) -> dict:
    """Streaming read bandwidth of host RAM.

    This is the number that decides whether a host-resident layer should be
    computed on the CPU or copied to the GPU: compare it against the h2d
    figure from `bench_link_bandwidth`, and feed the winner to
    `acvram plan --host-gb-s`.
    """
    import torch
    n = size_mb * 1024 * 1024 // 4
    src = torch.empty(n, dtype=torch.float32)
    dst = torch.empty_like(src)
    t = _timeit(lambda: dst.copy_(src), iters, 2, "cpu")
    return {"copy_gb_s": round(2 * size_mb / 1024 / t, 1),
            "read_gb_s": round(size_mb / 1024 / t, 1),
            "threads": torch.get_num_threads()}


def bench_cpu_kernels(shapes: Optional[list] = None, iters: int = 5) -> list[dict]:
    """Fused CPU GEMV against the dequantize-then-matmul path it replaces."""
    import torch

    from .kernels.cpu import (cpu_build_info, int4_matmul_cpu, nvfp4_matmul_cpu)
    from .quant.int4 import dequantize_int4, quantize_int4
    from .quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4

    info = cpu_build_info()
    if not info["available"]:
        return []
    shapes = shapes or [(4096, 4096), (14336, 4096)]
    out = []
    for m, k in shapes:
        w = torch.randn(m, k) * 0.02
        x = torch.randn(1, k)
        t4 = quantize_int4(w)
        qn = quantize_nvfp4(w)
        rows = {"shape": f"{m}x{k}", "avx2": info["avx2"]}
        rows["int4_fused_ms"] = round(1e3 * _timeit(
            lambda: int4_matmul_cpu(x, t4), iters, 1, "cpu"), 3)
        rows["int4_dequant_ms"] = round(1e3 * _timeit(
            lambda: x @ dequantize_int4(t4, torch.float32).t(), iters, 1, "cpu"), 3)
        rows["nvfp4_fused_ms"] = round(1e3 * _timeit(
            lambda: nvfp4_matmul_cpu(x, qn), iters, 1, "cpu"), 3)
        rows["nvfp4_dequant_ms"] = round(1e3 * _timeit(
            lambda: x @ dequantize_nvfp4(qn, torch.float32).t(), iters, 1, "cpu"), 3)
        rows["int4_gb_s"] = round(t4.nbytes / 1e9 / (rows["int4_fused_ms"] / 1e3), 1)
        out.append(rows)
    return out


def bench_kernels(shapes: Optional[list[tuple[int, int]]] = None,
                  iters: int = 50) -> list[dict]:
    """Fused GEMV against the dequantize-then-cuBLAS path, per format."""
    import torch

    from . import kernels
    from .quant.int4 import quantize_int4
    from .quant.nvfp4 import quantize_nvfp4

    shapes = shapes or [(4096, 4096), (14336, 4096), (5120, 5120)]
    out: list[dict] = []
    if not torch.cuda.is_available():
        return out

    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        cap = torch.cuda.get_device_capability(i)
        sm = cap[0] * 10 + cap[1]
        for (m, k) in shapes:
            w = (torch.randn(m, k) * 0.02)
            x = torch.randn(1, k, device=dev, dtype=torch.float32)
            row = {"device": f"cuda:{i}", "sm": sm, "shape": f"{m}x{k}"}

            if sm >= 100:
                q = quantize_nvfp4(w).to(dev)
                row["format"] = "nvfp4"
                row["gemv_ms"] = round(1e3 * _timeit(
                    lambda: kernels.nvfp4_matmul(x, q), iters, 5, dev), 4)
                row["dequant_ms"] = round(1e3 * _timeit(
                    lambda: kernels.nvfp4_dequant(q, torch.bfloat16),
                    iters, 5, dev), 4)
                nbytes = q.nbytes
            else:
                q = quantize_int4(w).to(dev)
                row["format"] = "int4_awq"
                row["gemv_ms"] = round(1e3 * _timeit(
                    lambda: kernels.int4_matmul(x, q), iters, 5, dev), 4)
                row["dequant_ms"] = round(1e3 * _timeit(
                    lambda: kernels.int4_dequant(q, torch.float16),
                    iters, 5, dev), 4)
                nbytes = q.nbytes

            # A weight-only GEMV is memory bound: this is the fraction of the
            # card's read bandwidth the kernel manages to use.
            row["effective_gb_s"] = round(nbytes / 1e9 / (row["gemv_ms"] / 1e3), 1)
            wbf = w.to(dev, dtype=torch.bfloat16)
            row["bf16_ref_ms"] = round(1e3 * _timeit(
                lambda: torch.nn.functional.linear(x.to(torch.bfloat16), wbf),
                iters, 5, dev), 4)
            out.append(row)
            del q, wbf
            torch.cuda.empty_cache()
    return out


def bench_decode(model_dir: str, n_tokens: int = 64,
                 prompt_len: int = 128) -> dict:
    """End-to-end decode rate on the real model, which is the number that counts."""
    import torch

    from .engine.loader import load_model
    from .engine.runner import Engine
    from .engine.sampler import SamplingParams

    t0 = time.time()
    loaded = load_model(model_dir, dtype=torch.bfloat16)
    load_s = time.time() - t0
    engine = Engine(loaded, None, max_batch_size=1,
                    max_model_len=prompt_len + n_tokens + 16)
    prompt = [1] * prompt_len
    params = SamplingParams(temperature=0.0, max_tokens=n_tokens)

    t0 = time.time()
    produced = 0
    for out in engine.generate(prompt, params):
        produced += 1
    elapsed = time.time() - t0
    return {
        "model": model_dir,
        "load_seconds": round(load_s, 1),
        "weights_bytes": loaded.model.nbytes,
        "prompt_len": prompt_len,
        "generated": produced,
        "wall_seconds": round(elapsed, 2),
        "decode_tok_s": round(produced / elapsed, 2) if elapsed else 0.0,
        "engine": engine.stats.to_dict(),
        "planned_decode_tok_s": loaded.plan.est_decode_tok_s,
    }


def run_benchmarks(args: argparse.Namespace) -> int:
    results: dict[str, Any] = {}
    what = getattr(args, "what", "all")

    if what in ("all", "bandwidth"):
        results["host_link"] = bench_link_bandwidth()
        results["vram"] = bench_vram_bandwidth()
        results["host_memory"] = bench_host_memory()
    if what in ("all", "kernels"):
        results["kernels"] = bench_kernels()
        results["cpu_kernels"] = bench_cpu_kernels()
    if what in ("all", "decode") and getattr(args, "model", None):
        results["decode"] = bench_decode(args.model)

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
        return 0

    for row in results.get("host_link", []):
        print(f"  {row['device']:<8} {row['name']:<28} "
              f"h2d {row['h2d_gb_s']:>7.1f} GB/s   d2h {row['d2h_gb_s']:>7.1f} GB/s")
    for row in results.get("vram", []):
        print(f"  {row['device']:<8} {'local copy':<28} "
              f"{row['copy_gb_s']:>7.1f} GB/s")
    hm = results.get("host_memory")
    if hm:
        print(f"  {'host':<8} {'DDR streaming read':<28} "
              f"{hm['read_gb_s']:>7.1f} GB/s   ({hm['threads']} threads)")
        links = [r["h2d_gb_s"] for r in results.get("host_link", [])]
        if links:
            best = max(links)
            verdict = ("compute host-tier layers on the CPU"
                       if hm["read_gb_s"] > best
                       else "stream host-tier weights to the GPU")
            print(f"           -> {verdict}  "
                  f"(--host-exec {'cpu' if hm['read_gb_s'] > best else 'stream'}"
                  f" --host-gb-s {hm['read_gb_s']})")
    if results.get("cpu_kernels"):
        print()
        simd = "AVX2" if results["cpu_kernels"][0]["avx2"] else "scalar"
        print(f"  CPU kernels ({simd} path):")
        for r in results["cpu_kernels"]:
            print(f"    {r['shape']:<12} int4 {r['int4_fused_ms']:7.2f}ms "
                  f"(vs {r['int4_dequant_ms']:7.2f}ms dequant)   "
                  f"nvfp4 {r['nvfp4_fused_ms']:7.2f}ms "
                  f"(vs {r['nvfp4_dequant_ms']:7.2f}ms)")
    if results.get("kernels"):
        print()
        print(f"  {'device':<8} {'shape':<12} {'format':<10} {'gemv':>9} "
              f"{'bf16 ref':>9} {'eff bw':>11}")
        for row in results["kernels"]:
            print(f"  {row['device']:<8} {row['shape']:<12} {row['format']:<10} "
                  f"{row['gemv_ms']:>7.3f}ms {row['bf16_ref_ms']:>7.3f}ms "
                  f"{row['effective_gb_s']:>8.1f} GB/s")
    if results.get("decode"):
        d = results["decode"]
        print()
        print(f"  decode        {d['decode_tok_s']} tok/s measured, "
              f"{d['planned_decode_tok_s']} tok/s planned")
        print(f"  load          {d['load_seconds']} s")
    if not results:
        print("nothing to benchmark (no CUDA device, and no model given)")
    return 0
