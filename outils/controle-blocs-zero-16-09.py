#!/usr/bin/env python3
"""Contrôle blocs-à-zéro, GLM-4.7-Flash — ordre poste7 (main 86b152b,
revue/poste7-glm-pile-correctif-16-09.md), 16/09/2026.

`nvfp4_quant_act_kernel` (acvram/kernels/acvram_kernels.cu:2021-2050) n'a
AUCUNE échelle globale (contrairement à `quantize_nvfp4`, poids,
nvfp4.py:237) : un bloc de 16 dont `amax/6 < 2⁻⁹` (plancher dénormal E4M3)
encode une échelle nulle (`sdec == 0.f`) et **tout le bloc est écrit à
zéro** (`if (sdec > 0.f)` garde la boucle d'empaquetage, acvram_kernels.cu
:2042). `outils/discriminateur-glm-mma0-16-09.py` réutilisait
`quantize_nvfp4`/`dequantize_nvfp4` (poids, échelle globale par appel) pour
fake-quantifier l'activation — ce plancher n'y est jamais atteint, qu'il le
soit ou non dans le noyau réel.

Contrôle, à sec, aucun GPU : réutilise le mini-converti bf16 déjà construit
par le discriminateur (mêmes 8 experts, même x réel capturé à l'entrée de
la couche MoE 1). Fraction de blocs de 16 dont `amax < 6×2⁻⁹` (zéro forcé)
sur `x` brut contre `x / s_up[e]` (échelle AWQ réelle de `up_proj`, converti
alpha-commun `GLM-4.7-Flash-srcbf16-nvfp4`).

Scellé (poste7) : ratio ≥ 3× ET fraction(x/s) ≥ 1 % → cause confirmée.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors import safe_open

SOURCE = "/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16"
NVFP4_CONVERTI = _RACINE + "/GLM-4.7-Flash-srcbf16-nvfp4"
SCRATCH = Path("/tmp/glm-discriminateur-mma0")
MINI = SCRATCH / "mini-hf"
CONVERTI_BF16 = SCRATCH / "mini-acvram-bf16"
REPO = Path(__file__).resolve().parent.parent
VENV_PROJET = os.environ.get("ACVRAM_PY", f"{REPO}/../../anticitoyen-vram/.venv/bin/python")

LAYER = 1
EXPERTS = list(range(8))
N_JETONS = 16
VOCAB_SUR = 150000
BLOCK = 16
SEUIL_AMAX = 6.0 * (2.0 ** -9)  # amax/6 < 2^-9 (plancher dénormal E4M3)


def invite() -> list[int]:
    return [(1000 + i * 13) % VOCAB_SUR + 10 for i in range(N_JETONS)]


def extraire_mini() -> None:
    from safetensors.torch import save_file

    MINI.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(SOURCE, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["num_hidden_layers"] = 2
    with open(MINI / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(SOURCE, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    voulus = {k for k in weight_map
              if k.startswith(("model.embed_tokens.", "model.norm.", "lm_head."))
              or k.startswith("model.layers.0.") or k.startswith("model.layers.1.")}
    par_fragment: dict[str, list[str]] = {}
    for k in voulus:
        par_fragment.setdefault(weight_map[k], []).append(k)
    tenseurs = {}
    for fragment, cles in par_fragment.items():
        with safe_open(os.path.join(SOURCE, fragment), framework="pt", device="cpu") as f:
            for k in cles:
                tenseurs[k] = f.get_tensor(k)
    save_file(tenseurs, str(MINI / "model.safetensors"))


def convertir_bf16() -> None:
    if CONVERTI_BF16.exists():
        shutil.rmtree(CONVERTI_BF16)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [VENV_PROJET, "-m", "acvram", "convert", str(MINI),
         "--format", "bf16", "--no-awq", "--quant-device", "cpu",
         "--host-exec", "cpu", "--max-model-len", "64", "-o", str(CONVERTI_BF16)],
        capture_output=True, text=True, env=env, cwd=str(REPO))
    if r.returncode != 0:
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"conversion mini bf16 échouée (code {r.returncode})")


class _Capture(Exception):
    pass


def capturer_activations() -> torch.Tensor:
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    loaded = load_model(str(CONVERTI_BF16), dtype=torch.bfloat16, max_model_len=64,
                        device_override="cpu")
    capture: dict = {}

    def hook(module, args):
        capture["x"] = args[0].detach().clone()
        raise _Capture()

    h = loaded.model.layers[LAYER].mlp.register_forward_pre_hook(hook)
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=64,
                    enable_cuda_graphs=False)
    engine.add_request(invite(), SamplingParams(temperature=0.0, max_tokens=1),
                       request_id="s0")
    try:
        for _ in range(3):
            if not engine.running and not engine.waiting:
                break
            engine.step()
    except _Capture:
        pass
    finally:
        h.remove()
    if "x" not in capture:
        raise RuntimeError("aucune activation capturée à l'entrée de la couche MoE")
    return capture["x"].to(torch.float32)


def lire_awq_reel(proj: str, expert_ids: list[int]) -> dict[int, torch.Tensor]:
    with open(os.path.join(NVFP4_CONVERTI, "acvram_manifest.json")) as fh:
        m = json.load(fh)
    wm = m["weight_map"]
    out = {}
    for i in expert_ids:
        k = f"model.layers.{LAYER}.mlp.experts.{i}.{proj}.weight.act_scale"
        with safe_open(os.path.join(NVFP4_CONVERTI, wm[k]), framework="pt", device="cpu") as f:
            out[i] = f.get_tensor(k).to(torch.float32)
    return out


def fraction_blocs_zero(t: torch.Tensor) -> tuple[float, int, int]:
    rows, k = t.shape
    assert k % BLOCK == 0
    blocs = t.reshape(rows, k // BLOCK, BLOCK)
    amax = blocs.abs().amax(dim=-1)
    nuls = (amax < SEUIL_AMAX)
    return float(nuls.float().mean()), int(nuls.sum()), int(nuls.numel())


def main() -> int:
    if not CONVERTI_BF16.exists():
        print("[1/3] extraction + conversion mini bf16 (pas de cache)...", flush=True)
        extraire_mini()
        convertir_bf16()
    else:
        print("[1/3] mini-converti bf16 déjà présent (cache), réutilisé", flush=True)
    print("[2/3] forward réel, capture activation couche MoE...", flush=True)
    x = capturer_activations()
    print(f"  x: {tuple(x.shape)}", flush=True)
    print("[3/3] table AWQ up_proj réelle (converti alpha-commun), fraction blocs-à-zéro...",
          flush=True)
    s_up = lire_awq_reel("up_proj", EXPERTS)

    frac_x, nuls_x, total_x = fraction_blocs_zero(x)
    par_expert = {}
    total_nuls_xs = 0
    total_blocs_xs = 0
    for i in EXPERTS:
        xs = x / s_up[i]
        frac_xs, nuls_xs, total_xs = fraction_blocs_zero(xs)
        par_expert[i] = {"fraction": frac_xs, "nuls": nuls_xs, "total": total_xs,
                         "s_up_min": float(s_up[i].min()), "s_up_max": float(s_up[i].max())}
        total_nuls_xs += nuls_xs
        total_blocs_xs += total_xs
    frac_xs_pool = total_nuls_xs / total_blocs_xs

    print(json.dumps({"x_brut": {"fraction": frac_x, "nuls": nuls_x, "total": total_x},
                      "x_sur_s_up": {"fraction_poolee": frac_xs_pool,
                                     "nuls": total_nuls_xs, "total": total_blocs_xs},
                      "par_expert": par_expert}, indent=2))

    ratio = frac_xs_pool / frac_x if frac_x > 0 else float("inf")
    print(f"\nfraction(x brut) = {frac_x:.6f} ({nuls_x}/{total_x})")
    print(f"fraction(x/s_up, poolée 8 experts) = {frac_xs_pool:.6f} "
          f"({total_nuls_xs}/{total_blocs_xs})")
    print(f"ratio = {ratio:.4f}")
    confirme = ratio >= 3.0 and frac_xs_pool >= 0.01
    print(f"VERDICT: {'cause CONFIRMÉE (plancher dénormal)' if confirme else 'cause écartée'}"
          f" (seuil ratio>=3x ET fraction>=1% : ratio={ratio:.4f}, fraction={frac_xs_pool:.4%})")

    out = SCRATCH / "resultat-blocs-zero.json"
    with open(out, "w") as fh:
        json.dump({"fraction_x_brut": frac_x, "fraction_x_sur_s_up_poolee": frac_xs_pool,
                  "ratio": ratio, "confirme": confirme, "par_expert": par_expert}, fh, indent=2)
    print(f"\nécrit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
