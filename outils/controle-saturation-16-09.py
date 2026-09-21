#!/usr/bin/env python3
"""Contrôle saturation haute + entrée down_proj, GLM (16/09) — ordre poste7 §3
(main da2290f, revue/poste7-glm-pile-correctif-16-09.md), suite au contrôle
blocs-à-zéro (plancher bas écarté).

(2) « Le haut » : `nvfp4_quant_act_kernel` borne l'échelle décodée à 448
(E4M3 max, `acvram_kernels.cu:2039`, `s = min(amax/6, 448)`). Un bloc dont
`amax > 6×448 = 2688` reçoit une échelle TROP PETITE (448 au lieu de
`amax/6`) : les valeurs qui en résultent dépassent 6 et saturent
(`satfinite`) à ±6 dans la conversion E2M1 — `quantize_nvfp4` (poids,
échelle globale float32 non bornée à 448, `nvfp4.py:237`) ne sature
jamais de cette façon. Mesuré sur `x / s_g[e]` et `x / s_u[e]` (échelles
AWQ réelles, mêmes 8 experts / même x réel que les contrôles précédents) :
max et p99,9 de `amax` par bloc de 16, fraction de blocs `amax > 2688`.
Scellé : fraction ≥ 0,1 % → cause confirmée.

(3) Troisième point d'entrée nvfp4 : `down_proj`. `act = silu(g)·u / s_d`
(`model.py:1028`, via `moe_act`/repli torch) puis `nvfp4_quant_act(act)`.
Mêmes quatre bras (W4A16/W4A4 × sans/avec échelle, cette fois sur
`down_proj`) + les deux fractions (bas `amax < 2⁻⁹`, haut `amax > 2688`)
sur `act`/`act/s_d`. Scellé : ratio err(iv)/err(iii) ≥ 1,5 OU fraction
(bas ou haut) ≥ 1 % → cause confirmée.

À sec, aucun GPU. Réutilise le mini-converti bf16 déjà construit par
`outils/discriminateur-glm-mma0-16-09.py` (cache `/tmp/glm-discriminateur-mma0`).
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
import torch.nn.functional as F
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
SEUIL_AMAX_BAS = 6.0 * (2.0 ** -9)   # amax/6 < 2^-9 -> bloc a zero
SEUIL_AMAX_HAUT = 6.0 * 448.0        # amax > 2688 -> echelle saturee a 448, valeurs clampees a +-6


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


def lire_poids_bf16(proj: str, expert_ids: list[int]) -> dict[int, torch.Tensor]:
    with open(os.path.join(SOURCE, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    out = {}
    for i in expert_ids:
        k = f"model.layers.{LAYER}.mlp.experts.{i}.{proj}.weight"
        with safe_open(os.path.join(SOURCE, weight_map[k]), framework="pt", device="cpu") as f:
            out[i] = f.get_tensor(k).to(torch.float32)
    return out


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


def stats_blocs(t: torch.Tensor) -> dict:
    rows, k = t.shape
    assert k % BLOCK == 0
    amax = t.reshape(rows, k // BLOCK, BLOCK).abs().amax(dim=-1).reshape(-1)
    return {"amax": amax}


def pool_stats(liste_amax: list[torch.Tensor]) -> dict:
    amax = torch.cat(liste_amax)
    n = amax.numel()
    frac_bas = float((amax < SEUIL_AMAX_BAS).float().mean())
    frac_haut = float((amax > SEUIL_AMAX_HAUT).float().mean())
    return {"max": float(amax.max()), "p999": float(torch.quantile(amax, 0.999)),
           "fraction_bas": frac_bas, "fraction_haut": frac_haut, "n_blocs": n}


def partie2(x: torch.Tensor) -> dict:
    s_g = lire_awq_reel("gate_proj", EXPERTS)
    s_u = lire_awq_reel("up_proj", EXPERTS)
    amax_g = [stats_blocs(x / s_g[i])["amax"] for i in EXPERTS]
    amax_u = [stats_blocs(x / s_u[i])["amax"] for i in EXPERTS]
    return {"x_sur_s_gate": pool_stats(amax_g), "x_sur_s_up": pool_stats(amax_u)}


def partie3(x: torch.Tensor) -> dict:
    from acvram.quant.nvfp4 import quantize_nvfp4, dequantize_nvfp4
    from acvram.quant.fakequant_activation import fake_quantize_nvfp4_activation

    w_gate = lire_poids_bf16("gate_proj", EXPERTS)
    w_up = lire_poids_bf16("up_proj", EXPERTS)
    w_down = lire_poids_bf16("down_proj", EXPERTS)
    s_down = lire_awq_reel("down_proj", EXPERTS)

    def norme_relative(a, b):
        return float((a - b).norm() / b.norm().clamp(min=1e-12))

    piles = {"i": [], "ii": [], "iii": [], "iv": []}
    amax_act, amax_acts = [], []
    par_expert = {}
    for i in EXPERTS:
        g = x @ w_gate[i].t()
        u = x @ w_up[i].t()
        act = F.silu(g) * u                       # [16, 1536], entree reelle de down_proj

        wd = w_down[i]
        sd = s_down[i]
        wdq = dequantize_nvfp4(quantize_nvfp4(wd.to(torch.bfloat16)), torch.float32)
        wds = wd * sd.unsqueeze(0)
        wdqs = dequantize_nvfp4(quantize_nvfp4(wds.to(torch.bfloat16)), torch.float32)
        acts = act / sd
        actq = fake_quantize_nvfp4_activation(act.to(torch.bfloat16)).to(torch.float32)
        actsq = fake_quantize_nvfp4_activation(acts.to(torch.bfloat16)).to(torch.float32)

        y_ref = act @ wd.t()
        y_i = act @ wdq.t()
        y_ii = acts @ wdqs.t()
        y_iii = actq @ wdq.t()
        y_iv = actsq @ wdqs.t()

        piles["i"].append((y_i, y_ref)); piles["ii"].append((y_ii, y_ref))
        piles["iii"].append((y_iii, y_ref)); piles["iv"].append((y_iv, y_ref))
        par_expert[i] = {k: norme_relative(v, y_ref) for k, (v, _) in
                         zip(("i", "ii", "iii", "iv"), ((y_i, None), (y_ii, None),
                                                        (y_iii, None), (y_iv, None)))}
        amax_act.append(stats_blocs(act)["amax"])
        amax_acts.append(stats_blocs(acts)["amax"])

    erreurs = {}
    for bras, paires in piles.items():
        ys = torch.cat([p[0] for p in paires], dim=0)
        refs = torch.cat([p[1] for p in paires], dim=0)
        erreurs[bras] = norme_relative(ys, refs)

    return {"erreurs": erreurs, "par_expert": par_expert,
           "act_brut": pool_stats(amax_act), "act_sur_s_down": pool_stats(amax_acts)}


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

    print("[3/3] partie (2) saturation haute sur gate/up...", flush=True)
    r2 = partie2(x)
    print(json.dumps(r2, indent=2))
    frac_g, frac_u = r2["x_sur_s_gate"]["fraction_haut"], r2["x_sur_s_up"]["fraction_haut"]
    confirme2 = max(frac_g, frac_u) >= 0.001
    print(f"\n(2) fraction(amax>2688) gate={frac_g:.6%} up={frac_u:.6%}  "
         f"VERDICT: {'cause CONFIRMEE' if confirme2 else 'cause ecartee'} (seuil >=0,1%)")

    print("\npartie (3) down_proj, quatre bras + fractions bas/haut...", flush=True)
    r3 = partie3(x)
    print(json.dumps(r3, indent=2))
    e = r3["erreurs"]
    ratio3 = e["iv"] / e["iii"]
    fb, fh = r3["act_sur_s_down"]["fraction_bas"], r3["act_sur_s_down"]["fraction_haut"]
    confirme3 = ratio3 >= 1.5 or max(fb, fh) >= 0.01
    print(f"\n(3) err(iii)={e['iii']:.6f} err(iv)={e['iv']:.6f} ratio={ratio3:.4f}  "
         f"fraction_bas(act/s_d)={fb:.6%} fraction_haut(act/s_d)={fh:.6%}  "
         f"VERDICT: {'cause CONFIRMEE' if confirme3 else 'cause ecartee'} "
         f"(seuil ratio>=1,5 OU fraction>=1%)")

    out = SCRATCH / "resultat-saturation.json"
    with open(out, "w") as fh_:
        json.dump({"partie2": r2, "confirme2": confirme2,
                  "partie3": r3, "ratio3": ratio3, "confirme3": confirme3}, fh_, indent=2)
    print(f"\nécrit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
