"""acvram command line.

    acvram doctor                     is this machine ready, and for what
    acvram detect                     what hardware is here
    acvram plan MODEL                 where every layer would go
    acvram convert MODEL -o DIR       quantize, one format per destination GPU
    acvram serve DIR                  OpenAI-compatible server
    acvram eval DIR [DIR ...]         perplexity, to rank formats on evidence
    acvram bench DIR                  measure what the plan only estimated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

__version__ = "0.2.0"


def _tty() -> bool:
    """Progress bars belong on a terminal, not in a pipe or a log file."""
    return sys.stderr.isatty() and not os.environ.get("NO_COLOR")


def _progress(text: str) -> None:
    if _tty():
        sys.stderr.write(f"\r{text[:100]:<100}")
        sys.stderr.flush()


def _progress_done() -> None:
    if _tty():
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()


def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def _h(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> int:
    from .hardware.detect import detect_rig
    rig = detect_rig(args.profile)
    if args.json:
        print(rig.to_json())
        return 0

    print(bold("hardware"))
    print(f"  source        {rig.source}")
    print(f"  distro        {rig.distro}")
    print(f"  kernel        {rig.kernel}")
    print(f"  cpu           {rig.cpu.model}")
    if rig.cpu.efficiency_cores:
        print(f"                {rig.cpu.performance_cores} P-cores + "
              f"{rig.cpu.efficiency_cores} E-cores, "
              f"pin workers to cpuset {rig.cpu.p_core_cpuset}")
    print(f"  host memory   {_h(rig.host.total)} total, "
          f"{_h(rig.host.available)} available")
    print(f"  driver / cuda {rig.driver_version or '?'} / {rig.cuda_version or '?'}")
    print()
    if not rig.gpus:
        print(yellow("  no NVIDIA GPU detected"))
        return 0
    print(bold("gpus"))
    for g in rig.gpus:
        caps = g.caps
        print(f"  [{g.index}] {g.name}")
        print(f"       {_h(g.total_mem)} VRAM, ~{g.vram_bandwidth_gbps:.0f} GB/s")
        print(f"       sm_{caps.sm} ({caps.arch_name})   "
              f"fp4={_yn(caps.fp4_tensor_core)} fp8={_yn(caps.fp8_tensor_core)} "
              f"int8={_yn(caps.int8_tensor_core)} bf16={_yn(caps.bf16)}")
        print(f"       PCIe gen{g.pcie_gen_cur or g.pcie_gen_max} "
              f"x{g.pcie_width_cur or g.pcie_width_max} "
              f"-> {g.host_link_gbps:.1f} GB/s to host")
        print(f"       {green('weights ' + caps.weight_format)}, kv {caps.kv_format}")
    if len(rig.gpus) > 1 and not any(rig.p2p_matrix[0][1:]):
        print()
        print(dim("  no peer-to-peer between GPUs (expected on GeForce): "
                  "inter-GPU tensors stage through pinned host memory"))
    return 0


def _yn(b: bool) -> str:
    return green("yes") if b else dim("no")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .hardware.detect import detect_rig
    ok = True
    print(bold("acvram doctor"))

    try:
        import torch
        print(f"  {green('ok')}    torch {torch.__version__}, "
              f"cuda {torch.version.cuda or 'cpu-only'}")
    except ImportError:
        print(f"  {red('FAIL')}  torch is not installed")
        return 1

    rig = detect_rig()
    if not rig.gpus:
        print(f"  {yellow('warn')}  no CUDA device; acvram will run on CPU only")
    for g in rig.gpus:
        caps = g.caps
        line = f"  {green('ok')}    [{g.index}] {g.name} sm_{caps.sm} -> {caps.weight_format}"
        print(line)
        if caps.sm >= 120:
            cuda = torch.version.cuda or "0.0"
            major, _, minor = cuda.partition(".")
            if (int(major or 0), int(minor or 0)) < (12, 8):
                ok = False
                print(f"  {red('FAIL')}  {g.name} is Blackwell (sm_{caps.sm}) but "
                      f"torch is built against CUDA {cuda}. sm_120 needs 12.8+. "
                      f"Install: pip install torch --index-url "
                      f"https://download.pytorch.org/whl/cu128")
        if g.pcie_width_cur and g.pcie_width_cur < g.pcie_width_max:
            print(f"  {yellow('warn')}  [{g.index}] link is running at x"
                  f"{g.pcie_width_cur} of x{g.pcie_width_max}; streamed layers "
                  f"will be {g.pcie_width_max / g.pcie_width_cur:.0f}x slower")

    from . import kernels
    info = kernels.build_info()
    if info["available"]:
        print(f"  {green('ok')}    fused CUDA kernels compiled for "
              f"{', '.join(info['device_caps']) or 'n/a'}")
    else:
        print(f"  {yellow('warn')}  fused CUDA kernels unavailable "
              f"({info['error']}); using the reference path")

    cpu = info["cpu"]
    if cpu["available"]:
        simd = "AVX2+FMA" if cpu["avx2"] else "scalar"
        note = "" if cpu["avx2"] else "  (no AVX2 on this CPU -- much slower)"
        print(f"  {green('ok')}    CPU kernels built, {simd} path{note}")
    else:
        print(f"  {yellow('warn')}  CPU kernels unavailable ({cpu['error']}); "
              f"host-tier layers will be slow")

    fp4 = info["fp4_tensorcore"]
    if fp4["available"]:
        print(f"  {green('ok')}    FP4 tensor-core GEMM via {fp4['impl']}")
    else:
        print(f"  {yellow('warn')}  no FP4 tensor-core GEMM: {fp4['reason']}")
        print(f"        prefill falls back to dequantize + cuBLAS; decode is "
              f"unaffected")

    for mod in ("safetensors", "fastapi", "uvicorn", "tokenizers", "jinja2"):
        try:
            __import__(mod)
            print(f"  {green('ok')}    {mod}")
        except ImportError:
            ok = False
            print(f"  {red('FAIL')}  {mod} is missing (pip install {mod})")

    if rig.host.total and rig.host.total < 32 * 1024 ** 3:
        print(f"  {yellow('warn')}  {_h(rig.host.total)} of host RAM limits the "
              f"host tier")
    return 0 if ok else 1


def cmd_plan(args: argparse.Namespace) -> int:
    from .engine.config import load_model_spec
    from .hardware.detect import detect_rig
    from .memory.tiering import PlannerOptions, auto_plan
    rig = detect_rig(args.profile)
    spec = load_model_spec(args.model, args.name)
    opts = PlannerOptions(
        max_model_len=args.max_model_len,
        max_concurrent_seqs=args.max_seqs,
        kv_bits=args.kv_bits,
        host_fraction=args.host_fraction,
        group_size=args.group_size,
        allow_host_tier=not args.no_host,
        force_format=args.format,
        gpus=args.gpus,
        host_exec=args.host_exec,
        host_compute_gb_s=args.host_gb_s,
    )
    plan, trials = auto_plan(spec, rig, opts)
    if args.json:
        print(json.dumps({"spec": spec.to_dict(), "plan": plan.to_dict(),
                          "trials": trials}, indent=2))
        return 0
    print(bold(spec.summary()))
    print()
    print(plan.render())
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .engine.config import load_model_spec
    from .hardware.detect import detect_rig
    from .memory.tiering import PlannerOptions, auto_plan
    from .quant.convert import ConversionOptions, convert_checkpoint

    rig = detect_rig(args.profile)
    spec = load_model_spec(args.model, args.name)
    plan, _ = auto_plan(spec, rig, PlannerOptions(
        max_model_len=args.max_model_len, max_concurrent_seqs=args.max_seqs,
        group_size=args.group_size, force_format=args.format, gpus=args.gpus,
        host_exec=args.host_exec, host_compute_gb_s=args.host_gb_s))
    print(bold(spec.summary()))
    print()
    print(plan.render())
    print()
    if any("does not fit" in w for w in plan.warnings) and not args.force:
        print(red("refusing to convert: the model does not fit. "
                  "Re-run with --force to write the shards anyway."))
        return 2

    # AWQ without activation statistics is a no-op: the grid search has
    # nothing to weight the channels by and settles on a flat scale. So the
    # statistics are collected up front, and if that is impossible we say so
    # and drop to round-to-nearest rather than claiming a scaling that never
    # happened.
    stats = None
    use_awq = not args.no_awq
    if use_awq:
        from .quant.collect import collect_activation_stats, load_calib_ids
        from .server.chat import load_tokenizer
        try:
            tokenizer = load_tokenizer(args.model)
            calib = load_calib_ids(tokenizer, args.calib_file, args.calib_seqs,
                                   args.calib_len, spec.vocab_size)
            print(f"  calibrating on {len(calib)} sequences "
                  f"({sum(len(c) for c in calib)} tokens) ...")

            def cprog(done: int, total: int) -> None:
                _progress(f"  calibrating layer {done}/{total}")

            stats = collect_activation_stats(
                args.model, spec, calib, device=args.calib_device,
                progress=cprog)
            _progress_done()
            print(f"  collected statistics for {len(stats)} tensors")
        except Exception as exc:                      # noqa: BLE001
            print(yellow(f"  calibration unavailable ({exc}); "
                         f"falling back to round-to-nearest"))
            use_awq = False
            stats = None

    opts = ConversionOptions(
        out_dir=args.out, awq=use_awq, use_hadamard=args.hadamard,
        group_size=args.group_size, n_grid=args.grid,
        lm_head_format=args.lm_head_format, dry_run=args.dry_run,
        mixed_precision=args.mixed_precision, snr_floor=args.snr_floor)

    last = [0.0]

    def progress(name: str, n: int, _: int) -> None:
        now = time.time()
        if now - last[0] < 0.5:
            return
        last[0] = now
        _progress(f"  {n} tensors  {name}")

    report = convert_checkpoint(args.model, plan, opts, spec=spec, stats=stats,
                                progress=progress)
    _progress_done()
    print(report.render())
    if not args.dry_run:
        print()
        print(f"  serve it with: {bold(f'acvram serve {args.out}')}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import torch
    import uvicorn

    from .engine.loader import load_model
    from .engine.runner import Engine
    from .server.app import create_app
    from .server.chat import load_tokenizer

    print(f"loading {args.model} ...")
    t0 = time.time()
    loaded = load_model(args.model, dtype=torch.bfloat16 if not args.fp16
                        else torch.float16,
                        max_model_len=args.max_model_len,
                        device_override=args.device)
    tokenizer = load_tokenizer(args.model)
    speculator = None
    if args.speculative == "ngram":
        from .engine.speculative import NGramProposer
        speculator = NGramProposer()
    elif args.speculative == "draft":
        if not args.draft_model:
            print(red("--speculative draft needs --draft-model"))
            return 2
        from .engine.speculative import DraftModelProposer
        draft = load_model(args.draft_model, dtype=torch.bfloat16,
                           device_override=args.draft_device)
        speculator = DraftModelProposer(draft, max_model_len=args.max_model_len)
        print(f"  draft model: {args.draft_model} "
              f"({_h(draft.model.nbytes)})")

    engine = Engine(loaded, tokenizer, max_batch_size=args.max_batch,
                    max_model_len=args.max_model_len,
                    enable_prefix_cache=not args.no_prefix_cache,
                    speculator=speculator, spec_k=args.spec_k)
    print(f"  loaded in {time.time() - t0:.1f} s, "
          f"{_h(loaded.model.nbytes)} of weights")
    print(f"  kv blocks: {engine.allocator.num_blocks} "
          f"({engine.allocator.num_blocks * 16} tokens per layer)")
    print(f"  prefix cache:  {'off' if args.no_prefix_cache else 'on'}")
    print(f"  speculation:   {args.speculative}"
          f"{'' if args.speculative == 'none' else f', k={args.spec_k}'}")
    if tokenizer:
        print(f"  chat template: {tokenizer.template_source}")
    else:
        print(yellow("  no tokenizer.json found; /v1 endpoints that take text "
                     "will fail"))

    name = args.served_name or os.path.basename(os.path.abspath(args.model))
    app = create_app(engine, tokenizer, name,
                     {"model_path": args.model, "version": __version__})
    print()
    print(f"  {bold('OpenAI API')}  http://{args.host}:{args.port}/v1")
    print(f"  {dim('models')}      curl http://{args.host}:{args.port}/v1/models")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    from .evaluate import perplexity, render
    results = []
    for path in args.models:
        def prog(done: int, total: int, _p: str = path) -> None:
            _progress(f"  {os.path.basename(_p)}: window {done}/{total}")
        results.append(perplexity(
            path, args.corpus, window=args.window, stride=args.stride,
            max_tokens=args.max_tokens, device=args.device, progress=prog))
        _progress_done()
    results.sort(key=lambda r: r.perplexity)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    print(render(results))
    if len(results) > 1:
        print()
        print(f"  best: {bold(results[0].model)} at {results[0].perplexity:.3f}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmarks
    return run_benchmarks(args)


def cmd_profiles(args: argparse.Namespace) -> int:
    from .hardware.profiles import list_profiles
    for name, desc in list_profiles().items():
        print(f"  {bold(name)}\n      {desc}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acvram",
        description="anticitoyen VRAM/RAM - tiered, per-GPU-quantized inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--version", action="version", version=f"acvram {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="report the local hardware")
    d.add_argument("--profile", help="use a declared profile instead of probing")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_detect)

    doc = sub.add_parser("doctor", help="check that this machine can run acvram")
    doc.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("profiles", help="list declared hardware profiles")
    pr.set_defaults(func=cmd_profiles)

    def add_plan_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("model", help="HF model directory (with config.json)")
        sp.add_argument("--name", help="override the model name")
        sp.add_argument("--profile", help="plan for a declared profile")
        sp.add_argument("--max-model-len", type=int, default=8192)
        sp.add_argument("--max-seqs", type=int, default=8,
                        help="concurrent sequences the KV cache must hold")
        sp.add_argument("--group-size", type=int, default=128)
        sp.add_argument("--format", help="force one weight format everywhere")
        sp.add_argument("--gpus", default="auto",
                        help="auto (let throughput decide), all, or indices "
                             "like 0,1")
        sp.add_argument("--host-exec", choices=["auto", "stream", "cpu"],
                        default="auto",
                        help="how host-resident weights are computed: copied "
                             "to the GPU, or in place on the CPU")
        sp.add_argument("--host-gb-s", type=float, default=70.0,
                        help="measured DDR read bandwidth; acvram bench "
                             "reports it")

    pl = sub.add_parser("plan", help="show where every layer would be placed")
    add_plan_args(pl)
    pl.add_argument("--kv-bits", type=int, default=8)
    pl.add_argument("--host-fraction", type=float, default=0.85)
    pl.add_argument("--no-host", action="store_true",
                    help="refuse to use host RAM as a tier")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_plan)

    cv = sub.add_parser("convert", help="quantize a checkpoint into acvram shards")
    add_plan_args(cv)
    cv.add_argument("-o", "--out", required=True, help="output directory")
    cv.add_argument("--no-awq", action="store_true",
                    help="plain round-to-nearest, no activation-aware scaling")
    cv.add_argument("--hadamard", choices=["auto", "always", "never"],
                    default="auto")
    cv.add_argument("--grid", type=int, default=20,
                    help="AWQ search grid resolution")
    cv.add_argument("--lm-head-format", help="format for the output projection")
    cv.add_argument("--dry-run", action="store_true",
                    help="report sizes and error without writing shards")
    cv.add_argument("--force", action="store_true",
                    help="convert even if the model does not fit")
    cv.add_argument("--calib-file", help="text file to calibrate on "
                                         "(default: a small built-in corpus)")
    cv.add_argument("--calib-seqs", type=int, default=16)
    cv.add_argument("--calib-len", type=int, default=512)
    cv.add_argument("--calib-device", default="cuda:0",
                    help="device to run calibration forwards on")
    cv.add_argument("--mixed-precision", choices=["auto", "off"], default="auto",
                    help="promote tensors that quantize badly to a wider format")
    cv.add_argument("--snr-floor", type=float, default=25.0,
                    help="layer-output SNR (dB) below which a tensor is promoted")
    cv.set_defaults(func=cmd_convert)

    sv = sub.add_parser("serve", help="run the OpenAI-compatible server")
    sv.add_argument("model", help="converted model directory")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--max-model-len", type=int, default=8192)
    sv.add_argument("--max-batch", type=int, default=16)
    sv.add_argument("--served-name", help="name reported by /v1/models")
    sv.add_argument("--device", help="force every layer onto one device")
    sv.add_argument("--fp16", action="store_true",
                    help="compute in float16 instead of bfloat16")
    sv.add_argument("--log-level", default="info")
    sv.add_argument("--speculative", choices=["none", "ngram", "draft"],
                    default="ngram",
                    help="ngram costs nothing and pays off when the output "
                         "quotes the input; draft needs --draft-model")
    sv.add_argument("--draft-model", help="converted directory of a small "
                                          "model to propose tokens")
    sv.add_argument("--draft-device", help="device for the draft model "
                                           "(default: the idlest GPU)")
    sv.add_argument("--spec-k", type=int, default=4,
                    help="tokens proposed per step")
    sv.add_argument("--no-prefix-cache", action="store_true",
                    help="disable KV reuse across requests")
    sv.set_defaults(func=cmd_serve)

    ev = sub.add_parser("eval", help="perplexity of one or more converted models")
    ev.add_argument("models", nargs="+", help="converted model directories")
    ev.add_argument("--corpus", help="text file to evaluate on")
    ev.add_argument("--window", type=int, default=512)
    ev.add_argument("--stride", type=int, default=256)
    ev.add_argument("--max-tokens", type=int, default=8192)
    ev.add_argument("--device", help="force a device")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)

    bn = sub.add_parser("bench", help="measure kernels, bandwidth and throughput")
    bn.add_argument("model", nargs="?", help="converted model directory")
    bn.add_argument("--what", default="all",
                    choices=["all", "kernels", "bandwidth", "decode"])
    bn.add_argument("--json", action="store_true")
    bn.set_defaults(func=cmd_bench)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(red(f"not found: {exc}"), file=sys.stderr)
        return 2
    except Exception as exc:                          # noqa: BLE001
        print(red(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        if os.environ.get("ACVRAM_TRACEBACK"):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
