#!/usr/bin/env python3
"""Discriminateur précision vs noyau, GLM-4.7-Flash — ordre poste7
(revue/poste7-glm-mma0-verdict-16-09.md §3, main f48c0f7), 16/09/2026.

Quatre bras `fake_quantize_nvfp4_activation` (aucun GPU requis) sur la
couche MoE 1 (`first_k_dense_replace=1`), 8 experts, poids bf16 de
`GLM-4.7-Flash-bf16` (source), table AWQ réelle lue dans le converti
alpha-commun `GLM-4.7-Flash-srcbf16-nvfp4`, activations = un forward réel
(16 jetons synthétiques, mêmes jetons que l'équivalence GLM du 15/09) sur un
mini-répertoire 2 couches converti en bf16 pur (aucun AWQ, aucune
quantification) — la même méthode que outils/equivalence-glm-2couches.py.

Scellé (poste7) : err(iv)/err(iii) ≥ 1,5 → (1b) précision, noyau innocent ;
≤ 1,1 → (1a) application, correctif poste4. Entre les deux : témoin ulp
d'poste1 (`poste7-glm-w4a4` §1.2 / `revue/prediction-plancher-w4a4-16-09.md`)
tranche : delta(alpha-commun) ≤ 2×plancher sans-AWQ = (1b).
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


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # à sec, avant tout import torch/acvram

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
    """x réel [16, H] entrant dans MoEBlock (couche 1), forward CPU teacher-forcé."""
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


def lire_poids_bf16(expert_ids: list[int]) -> dict[int, torch.Tensor]:
    with open(os.path.join(SOURCE, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    out = {}
    for i in expert_ids:
        k = f"model.layers.{LAYER}.mlp.experts.{i}.gate_proj.weight"
        with safe_open(os.path.join(SOURCE, weight_map[k]), framework="pt", device="cpu") as f:
            out[i] = f.get_tensor(k).to(torch.float32)
    return out


def lire_awq_reel(expert_ids: list[int]) -> dict[int, torch.Tensor]:
    with open(os.path.join(NVFP4_CONVERTI, "acvram_manifest.json")) as fh:
        m = json.load(fh)
    wm = m["weight_map"]
    out = {}
    for i in expert_ids:
        k = f"model.layers.{LAYER}.mlp.experts.{i}.gate_proj.weight.act_scale"
        with safe_open(os.path.join(NVFP4_CONVERTI, wm[k]), framework="pt", device="cpu") as f:
            out[i] = f.get_tensor(k).to(torch.float32)
    return out


def discriminer(x: torch.Tensor, poids: dict[int, torch.Tensor],
                echelles: dict[int, torch.Tensor], expert_ids: list[int]) -> dict:
    from acvram.quant.nvfp4 import quantize_nvfp4, dequantize_nvfp4
    from acvram.quant.fakequant_activation import fake_quantize_nvfp4_activation

    def norme_relative(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-12))

    piles = {"i": [], "ii": [], "iii": [], "iv": []}
    par_expert = {}
    for i in expert_ids:
        w = poids[i]
        s = echelles[i]
        y_ref = x @ w.t()

        wq = dequantize_nvfp4(quantize_nvfp4(w.to(torch.bfloat16)), torch.float32)
        ws = w * s.unsqueeze(0)
        wqs = dequantize_nvfp4(quantize_nvfp4(ws.to(torch.bfloat16)), torch.float32)
        xs = x / s
        xq = fake_quantize_nvfp4_activation(x.to(torch.bfloat16)).to(torch.float32)
        xsq = fake_quantize_nvfp4_activation(xs.to(torch.bfloat16)).to(torch.float32)

        y_i = x @ wq.t()
        y_ii = xs @ wqs.t()
        y_iii = xq @ wq.t()
        y_iv = xsq @ wqs.t()

        piles["i"].append((y_i, y_ref))
        piles["ii"].append((y_ii, y_ref))
        piles["iii"].append((y_iii, y_ref))
        piles["iv"].append((y_iv, y_ref))
        par_expert[i] = {
            "i": norme_relative(y_i, y_ref), "ii": norme_relative(y_ii, y_ref),
            "iii": norme_relative(y_iii, y_ref), "iv": norme_relative(y_iv, y_ref),
        }

    erreurs = {}
    for bras, paires in piles.items():
        ys = torch.cat([p[0] for p in paires], dim=0)
        refs = torch.cat([p[1] for p in paires], dim=0)
        erreurs[bras] = norme_relative(ys, refs)
    return {"pooled": erreurs, "par_expert": par_expert}


def main() -> int:
    print("[1/4] extraction mini-répertoire (couches 0-1)...", flush=True)
    extraire_mini()
    print("[2/4] conversion bf16 pure (aucun AWQ, CPU)...", flush=True)
    convertir_bf16()
    print("[3/4] forward réel, capture activation couche MoE...", flush=True)
    x = capturer_activations()
    print(f"  x: {tuple(x.shape)}", flush=True)
    print("[4/4] lecture poids bf16 (source) + table AWQ réelle (converti alpha-commun)...",
          flush=True)
    poids = lire_poids_bf16(EXPERTS)
    echelles = lire_awq_reel(EXPERTS)
    resultat = discriminer(x, poids, echelles, EXPERTS)

    e = resultat["pooled"]
    ratio = e["iv"] / e["iii"]
    print(json.dumps(resultat, indent=2))
    print(f"\nerr(i)={e['i']:.6f}  err(ii)={e['ii']:.6f}  "
          f"err(iii)={e['iii']:.6f}  err(iv)={e['iv']:.6f}")
    print(f"ratio err(iv)/err(iii) = {ratio:.4f}")
    if ratio >= 1.5:
        verdict = "(1b) précision — noyau innocent"
    elif ratio <= 1.1:
        verdict = "(1a) application — correctif poste4"
    else:
        verdict = "zone grise — témoin ulp tranche (poste7-glm-w4a4 §1.2)"
    print(f"VERDICT: {verdict}")

    out = SCRATCH / "resultat.json"
    with open(out, "w") as fh:
        json.dump({**resultat, "ratio_iv_iii": ratio, "verdict": verdict}, fh, indent=2)
    print(f"\nécrit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
