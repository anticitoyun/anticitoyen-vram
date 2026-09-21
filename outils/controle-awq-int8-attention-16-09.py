#!/usr/bin/env python3
"""Contrôle AWQ int8 sur q/kv/o (attention), GLM — ordre poste7 §5
(revue/poste7-glm-pile-correctif-16-09.md, main f127a2e), 16/09/2026.

+2,06 ms/pas (3,7 %) viennent de 124 tables AWQ sur les projections
**int8** (attention + down dense). L'AWQ compense une quantification
4 bits ; à 8 bits l'erreur est déjà ~16× plus petite et la division coûte
autant. Avant de les retirer pour la reconversion (recette poste2), un
contrôle à sec, même montage que le discriminateur MoE : poids bf16 réels
de `GLM-4.7-Flash-bf16` (source), table AWQ réelle de
`GLM-4.7-Flash-srcbf16-nvfp4-avant-noawq-experts` (int8, group_size=128),
`x` réel (forward mini-répertoire bf16 pur, CPU) — erreur relative de
`x·Wᵀ` en int8 par groupe, avec et sans échelle.

`q` et `kv` : `q_a_proj`/`kv_a_proj_with_mqa`, entrée = x réel de
`self_attn` (couche 6, choisie car SEULE couche <10 où les DEUX ont une
table AWQ réelle non-identité — scan des 47 couches, cf. note). `o` :
`o_proj` n'a **aucune table AWQ réelle sur AUCUNE des 47 couches**
(scan manifeste, `has_act_scale=False` partout) — rapporté tel quel, pas
de calcul nécessaire (rien à retirer, contribution nulle aux +2,06 ms).

Scellé (poste7, note originale — ATTENTION, sens du ratio, cf. verdict) :
err(avec)/err(sans) ≥ 0,9 → retrait sans risque ; < 0,7 → l'AWQ aide,
garder ; entre les deux → garder, trancher plus tard par PPL dédiée.

À sec, aucun GPU. Mini-répertoire ÉTENDU à 7 couches (0-6, pas 2 comme le
discriminateur MoE — nécessaire pour atteindre la couche 6), scratch
séparé.
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
INT8_CONVERTI = (_RACINE + "/"
                 "GLM-4.7-Flash-srcbf16-nvfp4-avant-noawq-experts")
SCRATCH = Path("/tmp/glm-awq-int8-attn")
MINI = SCRATCH / "mini-hf"
CONVERTI_BF16 = SCRATCH / "mini-acvram-bf16"
REPO = Path(__file__).resolve().parent.parent
VENV_PROJET = os.environ.get("ACVRAM_PY", f"{REPO}/../../anticitoyen-vram/.venv/bin/python")

N_COUCHES = 7          # 0..6, pour atteindre la couche 6 (q+kv reels ensemble)
LAYER = 6
N_JETONS = 16
VOCAB_SUR = 150000
GROUP_SIZE = 128


def invite() -> list[int]:
    return [(1000 + i * 13) % VOCAB_SUR + 10 for i in range(N_JETONS)]


def extraire_mini() -> None:
    from safetensors.torch import save_file

    MINI.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(SOURCE, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["num_hidden_layers"] = N_COUCHES
    with open(MINI / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(SOURCE, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    prefixes = tuple(f"model.layers.{i}." for i in range(N_COUCHES))
    voulus = {k for k in weight_map
              if k.startswith(("model.embed_tokens.", "model.norm.", "lm_head."))
              or k.startswith(prefixes)}
    par_fragment: dict[str, list[str]] = {}
    for k in voulus:
        par_fragment.setdefault(weight_map[k], []).append(k)
    tenseurs = {}
    for fragment, cles in par_fragment.items():
        with safe_open(os.path.join(SOURCE, fragment), framework="pt", device="cpu") as f:
            for k in cles:
                tenseurs[k] = f.get_tensor(k)
    save_file(tenseurs, str(MINI / "model.safetensors"))
    octets = sum(t.numel() * t.element_size() for t in tenseurs.values())
    print(f"  {octets / 1e9:.2f} Gio ecrits dans {MINI}", flush=True)


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

    couche = loaded.model.layers[LAYER]
    # `DecoderLayerGDN` : nom historique, sert aussi d'enveloppe générique
    # MLA — le module d'attention vit sous `.linear_attn` dans les deux cas.
    module_attn = getattr(couche, "self_attn", None) or couche.linear_attn
    h = module_attn.register_forward_pre_hook(hook)
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
        raise RuntimeError("aucune activation capturée à l'entrée de self_attn")
    return capture["x"].to(torch.float32)


def lire_poids_bf16(nom_proj: str) -> torch.Tensor:
    with open(os.path.join(SOURCE, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    k = f"model.layers.{LAYER}.self_attn.{nom_proj}.weight"
    with safe_open(os.path.join(SOURCE, weight_map[k]), framework="pt", device="cpu") as f:
        return f.get_tensor(k).to(torch.float32)


def lire_awq_reel(nom_proj: str) -> torch.Tensor:
    with open(os.path.join(INT8_CONVERTI, "acvram_manifest.json")) as fh:
        m = json.load(fh)
    wm = m["weight_map"]
    k = f"model.layers.{LAYER}.self_attn.{nom_proj}.weight.act_scale"
    with safe_open(os.path.join(INT8_CONVERTI, wm[k]), framework="pt", device="cpu") as f:
        return f.get_tensor(k).to(torch.float32)


def discriminer_int8(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> dict:
    from acvram.quant import formats

    def qd(t: torch.Tensor) -> torch.Tensor:
        q = formats.quantize(t, "int8", group_size=GROUP_SIZE)
        return formats.dequantize(q, torch.float32)

    def norme_relative(a, b):
        return float((a - b).norm() / b.norm().clamp(min=1e-12))

    y_ref = x @ w.t()
    wq = qd(w)
    y_sans = x @ wq.t()
    ws = w * s.unsqueeze(0)
    wqs = qd(ws)
    xs = x / s
    y_avec = xs @ wqs.t()

    err_sans = norme_relative(y_sans, y_ref)
    err_avec = norme_relative(y_avec, y_ref)
    return {"err_sans": err_sans, "err_avec": err_avec,
           "ratio_avec_sur_sans": err_avec / err_sans if err_sans > 0 else float("inf")}


def main() -> int:
    if not CONVERTI_BF16.exists():
        print(f"[1/3] extraction mini-répertoire ({N_COUCHES} couches)...", flush=True)
        extraire_mini()
        print("[2/3] conversion bf16 pure (aucun AWQ, CPU)...", flush=True)
        convertir_bf16()
    else:
        print("[1-2/3] mini-converti déjà présent (cache), réutilisé", flush=True)
    print(f"[3/3] forward réel, capture activation self_attn couche {LAYER}...", flush=True)
    x = capturer_activations()
    print(f"  x: {tuple(x.shape)}", flush=True)

    resultats = {}
    for proj in ("q_a_proj", "kv_a_proj_with_mqa"):
        w = lire_poids_bf16(proj)
        s = lire_awq_reel(proj)
        resultats[proj] = discriminer_int8(x, w, s)
        r = resultats[proj]
        print(f"{proj}: err_sans={r['err_sans']:.6f} err_avec={r['err_avec']:.6f} "
             f"ratio={r['ratio_avec_sur_sans']:.4f}")

    print("\no_proj : aucune table AWQ réelle sur aucune des 47 couches "
         "(scan manifeste has_act_scale) — rien à calculer, rien à retirer.")

    ratios = [resultats[p]["ratio_avec_sur_sans"] for p in resultats]
    pire = min(ratios)
    if pire >= 0.9:
        verdict = "retrait SANS RISQUE (q et kv, o trivial)"
    elif pire < 0.7:
        verdict = "l'AWQ int8 aide reellement — garder"
    else:
        verdict = "zone grise — garder pour la reconversion, trancher par PPL dediee"
    print(f"\npire ratio (q,kv) = {pire:.4f}  VERDICT: {verdict}")

    out = SCRATCH / "resultat.json"
    with open(out, "w") as fh:
        json.dump({"par_projection": resultats, "o_proj": "identite partout (0/47 couches)",
                  "pire_ratio": pire, "verdict": verdict}, fh, indent=2)
    print(f"\nécrit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
