"""Des mesures, pour remplacer les estimations du planificateur par de vrais
nombres.

Le planificateur de placement travaille à partir de bandes passantes de plaque
signalétique et d'un modèle de coût limité par la mémoire. C'est suffisant pour
choisir une disposition, mais la seule façon de savoir ce que fait réellement
cette machine est de la mesurer — en particulier le lien vers l'appareil, seul
nombre qui décide si transférer une couche depuis la mémoire vive est viable, et
qui dépend du port dans lequel la carte a atterri.
"""

from __future__ import annotations

import argparse
import json
import os
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
    """Débit de copie hôte-appareil en mémoire épinglée, par GPU, dans les deux sens.

    La mémoire épinglée est utilisée à dessein : c'est ce qu'emploie le chemin de
    transfert, et un chiffre mesuré en mémoire paginable sous-estimerait le lien
    de près de moitié.
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


def bench_peer_links(size_mb: int = 256, iters: int = 5) -> list[dict]:
    """Debit reel entre GPU, et disponibilite du P2P, dans les deux sens.

    Le P2P n'est pas garanti : sans NVLink et derriere un pont de chipset
    (``PHB`` chez ``nvidia-smi topo -m``), une copie entre cartes repasse par
    l'hote et peut se reveler plus lente qu'un aller-retour par la memoire
    vive. C'est une mesure, pas une hypothese : le placement en depend.
    """
    import torch
    out: list[dict] = []
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return out
    n = size_mb * 1024 * 1024 // 4
    for i in range(torch.cuda.device_count()):
        for j in range(torch.cuda.device_count()):
            if i == j:
                continue
            di, dj = torch.device(f"cuda:{i}"), torch.device(f"cuda:{j}")
            try:
                src = torch.empty(n, dtype=torch.float32, device=di)
                dst = torch.empty(n, dtype=torch.float32, device=dj)
            except torch.OutOfMemoryError:
                continue
            t = _timeit(lambda: dst.copy_(src), iters, 2, dj)
            out.append({
                "from": f"cuda:{i}", "to": f"cuda:{j}",
                "p2p": bool(torch.cuda.can_device_access_peer(i, j)),
                "gb_s": round(size_mb / 1024 / t, 1),
            })
            del src, dst
            torch.cuda.empty_cache()
    return out


def pcie_link_state() -> list[dict]:
    """Largeur et generation PCIe effectives, telles que le pilote les rapporte.

    Une carte cablee en x16 mais negociee en x8 divise par deux tout transfert :
    le planificateur doit connaitre la largeur courante, pas celle du connecteur.

    Attention a la lecture : au repos le pilote retrograde le lien (une carte
    peut se declarer en ``PCIe 1.0`` alors qu'elle remonte en 4.0 des la
    premiere copie). Seule la *largeur* est fiable a froid ; pour la
    generation, c'est le debit mesure par ``bench_link_bandwidth`` qui tranche.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,pcie.link.gen.current,"
             "pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        f = [c.strip() for c in line.split(",")]
        if len(f) < 6:
            continue
        rows.append({"device": f"cuda:{f[0]}", "name": f[1],
                     "gen": int(f[2]), "width": int(f[3]),
                     "gen_max": int(f[4]), "width_max": int(f[5]),
                     "degraded": int(f[3]) < int(f[5]) or int(f[2]) < int(f[4])})
    return rows


def topology(size_mb: int = 256) -> dict:
    """Carte complete des couts de transfert de cette machine.

    Ecrite une fois dans ``acvram-topology.json``, elle remplace les constantes
    du planificateur par des chiffres mesures ici.
    """
    import torch
    gpus = []
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        pr = torch.cuda.get_device_properties(i)
        cc = (pr.major, pr.minor)
        gpus.append({
            "device": f"cuda:{i}", "name": pr.name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "memory_bytes": pr.total_memory,
            "multiprocessors": pr.multi_processor_count,
        })
    host_ram = 0
    try:
        host_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass
    return {
        "acvram_topology": 1,
        "gpus": gpus,
        "host": {"ram_bytes": host_ram,
                 "memory": bench_host_memory(size_mb)},
        "pcie": pcie_link_state(),
        "links": bench_link_bandwidth(size_mb),
        "peers": bench_peer_links(size_mb),
        "vram": bench_vram_bandwidth(size_mb * 2),
    }


def bench_vram_bandwidth(size_mb: int = 1024, iters: int = 20) -> list[dict]:
    """Bande passante de lecture locale à l'appareil, mesurée par une grande copie contiguë."""
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
        # une copie lit une fois et écrit une fois
        out.append({"device": f"cuda:{i}",
                    "copy_gb_s": round(2 * size_mb / 1024 / t, 1)})
        del src, dst
        torch.cuda.empty_cache()
    return out


def bench_host_memory(size_mb: int = 512, iters: int = 5) -> dict:
    """Bande passante de lecture séquentielle de la mémoire vive.

    C'est le nombre qui décide si une couche résidant en RAM doit être calculée
    sur le processeur ou copiée vers le GPU : à comparer au chiffre hôte-appareil
    de `bench_link_bandwidth`, puis à donner au vainqueur via
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
    """GEMV processeur fusionné, face au chemin déquantification-puis-produit qu'il remplace."""
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
    """GEMV fusionné face au chemin déquantification-puis-cuBLAS, par format."""
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

            # Un GEMV sur les poids seuls est limité par la mémoire : voici la
            # fraction de la bande passante de lecture que le noyau exploite.
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
    """Débit de décodage de bout en bout sur le vrai modèle, le nombre qui compte."""
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


DEFAULT_TOPOLOGY_PATH = os.path.expanduser("~/.config/acvram/acvram-topology.json")


def load_topology(path: Optional[str] = None) -> Optional[dict]:
    """Relit la topologie mesuree, si ``acvram bench --what topology`` est passe."""
    p = path or DEFAULT_TOPOLOGY_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


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
    if what in ("all", "topology"):
        results["topology"] = topology()
        out = getattr(args, "topology_out", None) or DEFAULT_TOPOLOGY_PATH
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(results["topology"], fh, indent=2)
            results["topology_path"] = out
        except OSError as exc:
            results["topology_error"] = str(exc)
    if what in ("all", "decode") and getattr(args, "model", None):
        results["decode"] = bench_decode(args.model)

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
        return 0

    topo = results.get("topology")
    if topo:
        print()
        print("  topologie")
        for g in topo["gpus"]:
            print(f"    {g['device']:<8} {g['name']:<26} "
                  f"cc {g['compute_capability']:<5} "
                  f"{g['memory_bytes'] / 1024 ** 3:5.1f} Gio")
        for r in topo.get("pcie", []):
            etat = f"PCIe {r['gen']}.0 x{r['width']} au repos"
            if r["width"] < r["width_max"]:
                etat += (f"  (largeur bridee : la carte sait faire "
                         f"x{r['width_max']})")
            print(f"    {r['device']:<8} {etat}")
        for r in topo.get("peers", []):
            p2p = "P2P" if r["p2p"] else "sans P2P, par l'hote"
            print(f"    {r['from']} -> {r['to']:<8} {r['gb_s']:>6.1f} Go/s  ({p2p})")
        hm = topo["host"]["memory"]["read_gb_s"]
        pires = [r["gb_s"] for r in topo.get("peers", [])]
        if pires and hm > max(pires):
            print(f"    -> la memoire vive ({hm:.1f} Go/s) est plus rapide que "
                  f"le meilleur lien entre cartes ({max(pires):.1f} Go/s) : "
                  f"n'y faire transiter que ce qui y reside.")
        if results.get("topology_path"):
            print(f"    ecrit dans {results['topology_path']}")

    for row in results.get("host_link", []):
        print(f"  {row['device']:<8} {row['name']:<28} "
              f"h2d {row['h2d_gb_s']:>7.1f} GB/s   d2h {row['d2h_gb_s']:>7.1f} GB/s")
    for row in results.get("vram", []):
        print(f"  {row['device']:<8} {'copie locale':<28} "
              f"{row['copy_gb_s']:>7.1f} Go/s")
    hm = results.get("host_memory")
    if hm:
        print(f"  {'hote':<8} {'lecture sequentielle DDR':<28} "
              f"{hm['read_gb_s']:>7.1f} GB/s   ({hm['threads']} threads)")
        links = [r["h2d_gb_s"] for r in results.get("host_link", [])]
        if links:
            best = max(links)
            verdict = ("calculer l'etage hote sur le processeur"
                       if hm["read_gb_s"] > best
                       else "transferer les poids de l'etage hote vers le GPU")
            print(f"           -> {verdict}  "
                  f"(--host-exec {'cpu' if hm['read_gb_s'] > best else 'stream'}"
                  f" --host-gb-s {hm['read_gb_s']})")
    if results.get("cpu_kernels"):
        print()
        simd = "AVX2" if results["cpu_kernels"][0]["avx2"] else "scalar"
        print(f"  Noyaux processeur (chemin {simd}) :")
        for r in results["cpu_kernels"]:
            print(f"    {r['shape']:<12} int4 {r['int4_fused_ms']:7.2f}ms "
                  f"(vs {r['int4_dequant_ms']:7.2f}ms dequant)   "
                  f"nvfp4 {r['nvfp4_fused_ms']:7.2f}ms "
                  f"(vs {r['nvfp4_dequant_ms']:7.2f}ms)")
    if results.get("kernels"):
        print()
        print(f"  {'appareil':<8} {'forme':<12} {'format':<10} {'gemv':>9} "
              f"{'ref bf16':>9} {'bp eff':>11}")
        for row in results["kernels"]:
            print(f"  {row['device']:<8} {row['shape']:<12} {row['format']:<10} "
                  f"{row['gemv_ms']:>7.3f}ms {row['bf16_ref_ms']:>7.3f}ms "
                  f"{row['effective_gb_s']:>8.1f} GB/s")
    if results.get("decode"):
        d = results["decode"]
        print()
        print(f"  decodage      {d['decode_tok_s']} jetons/s mesures, "
              f"{d['planned_decode_tok_s']} prevus")
        print(f"  chargement    {d['load_seconds']} s")
    if not results:
        print("rien a mesurer (aucun peripherique CUDA, et aucun modele fourni)")
    return 0
