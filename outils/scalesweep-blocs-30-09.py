#!/usr/bin/env python3
"""ScaleSweep sur de vrais blocs de poids (poste6, 30/09, ordre chef ; arXiv 2606.07618) — mesure CPU, à sec.

Pour chaque tenseur nvfp4 retenu d'un modèle DÉJÀ converti, le poids bf16 d'ORIGINE est requantifié avec les quatre
échelles de bloc de `acvram.quant.nvfp4` (max6 = AbsMax, 4sur6, balayage = ScaleSweep-MSE, balayage-w = ScaleSweep-WMSE
quand le converti porte un `act_scale`, importance = act_scale², proxy nommé de diag(XᵀX)) ; on relève MSE, WMSE, échelles
E4M3 sous-normales, échelles « écrasées » par la conversion Marlin (`marlin_port.echelles_ecrasees`, critère du refus ya1,
facteur commun et par ligne) et le temps. Les piles d'experts (`--experts tous`) sont jugées comme le moteur : échelles des
128 experts empilées, un facteur par (pile, colonne).
Sortie : une ligne par tenseur, les totaux relatifs à max6, un JSON (`--sortie`). En-tête : commit, charge hôte, fils.
Prédiction et seuils : revue/poste6-scalesweep-verdict-30-09.md, écrits AVANT. Aucune carte : CUDA_VISIBLE_DEVICES vide."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from acvram.quant.nvfp4 import quantize_nvfp4, dequantize_nvfp4, sous_normales_e4m3  # noqa: E402
from acvram.kernels import marlin_port as MP  # noqa: E402
from racine_modeles import racine_modeles  # noqa: E402

ECHELLES = ("max6", "4sur6", "balayage", "balayage-w")


def _ouvrir_index(dossier: Path) -> dict:
    idx = dossier / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    seul = sorted(dossier.glob("*.safetensors"))
    assert len(seul) == 1, f"ni index ni safetensors unique sous {dossier}"
    with safe_open(seul[0], "pt") as f:
        return {k: seul[0].name for k in f.keys()}


def _lire(dossier: Path, carte: dict, nom: str) -> torch.Tensor:
    with safe_open(dossier / carte[nom], "pt") as f:
        return f.get_tensor(nom)


def _tenseurs_retenus(manifest: dict, couches: list[int], experts: str) -> list[str]:
    noms = []
    for nom, e in manifest["tensors"].items():
        if e.get("format") != "nvfp4":
            continue
        parts = nom.split(".")
        if "layers" not in parts:
            continue
        couche = int(parts[parts.index("layers") + 1])
        if couche not in couches:
            continue
        if ".experts." in nom:
            if experts == "aucun":
                continue
            if experts != "tous":
                lo, hi = (int(x) for x in experts.split("-"))
                if not lo <= int(parts[parts.index("experts") + 1]) <= hi:
                    continue
        noms.append(nom)
    return sorted(noms)


def _ecrasees(bs_liste: list[torch.Tensor]) -> tuple[int, int]:
    """(facteur commun, facteur par ligne) sur les échelles empilées [E, N, K/16] → comme le moteur (moe.py:489, 514)."""
    bs = torch.stack(bs_liste) if len(bs_liste) > 1 else bs_liste[0].unsqueeze(0)
    return MP.echelles_ecrasees(bs, par_ligne=False), MP.echelles_ecrasees(bs, par_ligne=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converti", required=True, help="nom sous la racine des convertis, ou chemin")
    ap.add_argument("--source", required=True, help="dossier HF bf16 d'origine")
    ap.add_argument("--couches", default="0", help="ex. 0,12,24,36")
    ap.add_argument("--experts", default="aucun", help="aucun | tous | a-b")
    ap.add_argument("--sortie", required=True, help="JSON de résultats")
    ap.add_argument("--commit", default=None, help="assert git rev-parse HEAD == ce commit (en-tête du verdict)")
    ap.add_argument("--fils", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.fils)
    head = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    if a.commit and not head.startswith(a.commit) and not a.commit.startswith(head):
        print(f"ÉCHEC : HEAD {head} ≠ commit attendu {a.commit}"); return 2
    conv = Path(a.converti) if os.sep in a.converti else Path(racine_modeles()) / a.converti
    if not conv.exists():
        conv = Path("/mnt/AI_GENERATOR/models_acvram") / a.converti
    src = Path(a.source)
    manifest = json.load(open(conv / "acvram_manifest.json"))
    carte_src = _ouvrir_index(src)
    couches = [int(x) for x in a.couches.split(",")]
    noms = _tenseurs_retenus(manifest, couches, a.experts)
    charge = os.getloadavg()
    print(f"# commit {head} · charge {charge[0]:.2f}/{charge[1]:.2f}/{charge[2]:.2f} · fils {a.fils} · CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"# converti {conv.name} · source {src.name} · couches {couches} · experts {a.experts} · {len(noms)} tenseurs")
    resultats, piles = [], {}
    tot = {e: {"mse": 0.0, "wmse": 0.0, "sn": 0, "t": 0.0, "blocs": 0, "balayes": 0, "n_wmse": 0} for e in ECHELLES}
    t_debut = time.time()
    for i, nom in enumerate(noms):
        ent = manifest["tensors"][nom]
        w = _lire(src, carte_src, nom)
        imp = None
        if ent.get("has_act_scale"):
            cle = nom + ".act_scale"
            imp = _lire(conv, manifest["weight_map"], cle).to(torch.float32) ** 2
        ligne = {"nom": nom, "forme": list(w.shape), "act_scale": imp is not None}
        w32 = w.to(torch.float32)
        for e in ECHELLES:
            if e == "balayage-w" and imp is None:
                continue
            t0 = time.time()
            q = quantize_nvfp4(w, echelle=e, importance=imp if e == "balayage-w" else None)
            dt = time.time() - t0
            err = (w32 - dequantize_nvfp4(q, torch.float32)) ** 2
            mse = float(err.mean())
            wmse = float((err * imp).mean()) if imp is not None else None
            sn = sous_normales_e4m3(q.block_scale)
            ec = MP.echelles_ecrasees(q.block_scale, par_ligne=False), MP.echelles_ecrasees(q.block_scale, par_ligne=True)
            ligne[e] = {"mse": mse, "wmse": wmse, "sous_normales": sn, "ecrasees": ec[0], "ecrasees_ligne": ec[1],
                        "balayes": q.echelle_stats.get("balayes", 0), "blocs": q.echelle_stats["blocs"], "t": dt}
            tot[e]["mse"] += mse * w.numel(); tot[e]["sn"] += sn; tot[e]["t"] += dt
            tot[e]["blocs"] += q.echelle_stats["blocs"]; tot[e]["balayes"] += q.echelle_stats.get("balayes", 0)
            if wmse is not None:
                tot[e]["wmse"] += wmse * w.numel(); tot[e]["n_wmse"] += w.numel()
            if ".experts." in nom:
                p = nom.split(".experts.")[0] + "." + nom.rsplit(".", 2)[-2]
                piles.setdefault((p, e), []).append(q.block_scale)
        r = ligne["max6"]["mse"]
        print(f"{i + 1:3d}/{len(noms)} {nom.replace('model.layers.', 'L')} {tuple(w.shape)} "
              + " ".join(f"{e}:{ligne[e]['mse'] / r:.3f}" for e in ECHELLES if e in ligne)
              + f" sn:{'/'.join(str(ligne[e]['sous_normales']) for e in ECHELLES if e in ligne)}"
              + f" t:{'/'.join(f'{ligne[e]['t']:.1f}' for e in ECHELLES if e in ligne)}", flush=True)
        resultats.append(ligne)
    n = sum(int(torch.tensor(l["forme"]).prod()) for l in resultats)
    print("\n# TOTAUX (relatifs à max6 ; MSE pondérée par le nombre de poids)")
    piles_res = {}
    for e in ECHELLES:
        if tot[e]["blocs"] == 0:
            continue
        mse_rel = tot[e]["mse"] / tot["max6"]["mse"]
        wmse_rel = (tot[e]["wmse"] / tot["max6"]["wmse"]) if tot[e]["n_wmse"] and tot["max6"]["wmse"] else None
        print(f"{e:11s} MSE {mse_rel:.4f}  WMSE {wmse_rel if wmse_rel is None else f'{wmse_rel:.4f}'}  "
              f"sous-normales {tot[e]['sn']} ({tot[e]['sn'] / max(tot['max6']['sn'], 1):.2f}× max6)  "
              f"balayés {tot[e]['balayes']}/{tot[e]['blocs']}  temps {tot[e]['t']:.0f} s ({tot[e]['t'] / max(tot['max6']['t'], 1e-9):.1f}× max6)")
    if piles:
        print("\n# PILES d'experts (échelles empilées, facteur commun / par ligne) — critère du refus Marlin")
        for (p, e), liste in sorted(piles.items()):
            ec = _ecrasees(liste)
            piles_res[f"{p}|{e}"] = {"experts": len(liste), "ecrasees": ec[0], "ecrasees_ligne": ec[1]}
            print(f"{p.replace('model.layers.', 'L'):40s} {e:11s} experts {len(liste):3d} écrasées {ec[0]:6d} par ligne {ec[1]:6d}")
    json.dump({"commit": head, "charge": charge, "fils": a.fils, "converti": conv.name, "source": src.name,
               "couches": couches, "experts": a.experts, "poids": n, "duree_s": time.time() - t_debut,
               "totaux": tot, "piles": piles_res, "tenseurs": resultats}, open(a.sortie, "w"), indent=1)
    print(f"\n# {len(noms)} tenseurs, {n / 1e6:.0f} M poids, {time.time() - t_debut:.0f} s → {a.sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
