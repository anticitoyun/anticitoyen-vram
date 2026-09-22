"""Pièce 26 : prédicteur hors ligne de « Four Over Six » (échelle de bloc
amax/4 quand elle fait mieux que amax/6), jugé contre les mesures déjà
faites (scellé E : KL 4sur6 **1,39** contre max6 recalibré **0,87** — 4sur6
PERD).

Sur les poids bf16 de la source, par tenseur et par bloc de 16, avec les
FONCTIONS DU DÉPÔT (`quantize_nvfp4`, `_mse_bloc`, `_coder`) — mêmes blocs,
mêmes arrondis E4M3 de l échelle et E2M1 des quartets qu à la conversion :
  G      = (Σ MSE6 − Σ MSE4) / Σ MSE6   par tenseur, par couche, global
           (pondéré par le nombre de blocs) — le critère du 21/09 ;
  part4  = part des blocs où amax/4 est retenu ;
  clamp4 = part des blocs où l échelle amax/4 sature (≥ 448 en E4M3) ;
  Linf   = erreur MAXIMALE par bloc (et la part des blocs où 4sur6
           l AUGMENTE) — la grandeur que la MSE ne voit pas et que la KL,
           elle, paie : réduire l erreur quadratique en rapprochant l échelle
           des petites valeurs éloigne les grandes.

**Classe des blocs gagnants — MESURÉE (test à sec, 22/09), contre l intuition
courante** : amax/4 gagne sur les blocs **PLATS** (seize valeurs du même ordre,
platitude moyenne(|w|)/amax élevée) et **jamais** sur les blocs piqués. Raison :
les niveaux E2M1 hauts sont espacés (…, 2, 3, 4, 6) — avec l échelle amax/6 une
valeur à 0,8·amax tombe entre 4 et 6 (erreur ≤ 0,167·amax), avec amax/4 elle
tombe entre 3 et 4 (≤ 0,125·amax) ; à l inverse, sur un bloc piqué, amax/6
donne le pas fin (0,5·s = amax/12 contre amax/8) là où sont les quinze petites
valeurs, et le pic reste exact (6·s = amax). Les modèles extérieurs qui
annoncent « amax/4 gagne sur les queues lourdes » ont donc le signe à l envers ;
`platitude_gagnants` / `platitude_perdants` publient la mesure par tenseur.

**Ce que le prédicteur ne peut pas faire, dit d avance** : `4sur6` choisit
par bloc le MEILLEUR des deux au sens de la MSE (`nvfp4.py:302`), donc
**G ≥ 0 par construction** — un G positif ne prédit rien, il constate la
règle de sélection. Le prédicteur n est donc utile que par `Linf` et
`clamp4`. Verdict : si G > 0 et que Linf augmente sur une part notable des
blocs, le prédicteur est **d accord** avec le scellé E (4sur6 perd en
qualité malgré une MSE plus basse) ; si Linf n augmente nulle part, le
prédicteur est **réfuté** — il prédirait 4sur6 gagnant, contre la mesure.

Usage : python outils/gpu/mesure/predire-4sur6.py SOURCE_BF16 [--limite 40] [--json S]
        (~10 min sur le 31B avec --limite 0 ; un shard à la fois)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
import torch  # noqa: E402

E2M1_MAX, E4M3_MAX = 6.0, 448.0
RE_COUCHE = re.compile(r"layers\.(\d+)\.")


def par_bloc(w: torch.Tensor, bloc: int = 16) -> dict:
    """MSE, erreur L∞ et saturation des deux échelles, bloc par bloc, avec les
    fonctions du dépôt (mêmes arrondis qu à la conversion)."""
    from acvram.quant.nvfp4 import _coder, _levels_tensor
    x = w.detach().to(torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    out_f, k = x.shape[0], x.shape[1]
    if k % bloc:
        x = torch.nn.functional.pad(x, (0, bloc - k % bloc))
        k = x.shape[1]
    wb = x.reshape(out_f, k // bloc, bloc)
    amax = wb.abs().amax()
    gs = (amax / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    block_amax = wb.abs().amax(dim=-1)

    def candidat(div: float):
        e4m3 = (block_amax / div / gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        eff = e4m3.to(torch.float32) * gs
        codes = _coder(wb, eff)
        vals = _levels_tensor(wb.device)[codes.long()] * eff.unsqueeze(-1)
        err = (wb.abs() - vals).abs()
        return {"e4m3": e4m3, "mse": (err ** 2).sum(dim=-1), "linf": err.amax(dim=-1)}
    c6, c4 = candidat(E2M1_MAX), candidat(4.0)
    mieux = c4["mse"] < c6["mse"]                       # la règle exacte de `quantize_nvfp4`
    n = int(block_amax.numel())
    # platitude d un bloc = moyenne(|w|) / amax du bloc : 1 = seize valeurs égales,
    # → 0 = une valeur écrase les quinze autres. C est la CLASSE des blocs gagnants.
    plat = (wb.abs().mean(dim=-1) / block_amax.clamp(min=1e-30))
    return {"blocs": n,
            "mse6": float(c6["mse"].sum()), "mse4": float(torch.where(mieux, c4["mse"], c6["mse"]).sum()),
            "part4": float(mieux.float().mean()),
            "clamp4": float(((c4["e4m3"].to(torch.float32) >= E4M3_MAX) & mieux).float().mean()),
            "linf6": float(c6["linf"].sum()), "linf4": float(torch.where(mieux, c4["linf"], c6["linf"]).sum()),
            "linf_pire": float(((c4["linf"] > c6["linf"]) & mieux).float().mean()),
            "linf_pire_ratio": float((torch.where(mieux, c4["linf"], c6["linf"]) / c6["linf"].clamp(min=1e-30)).amax()),
            "platitude_gagnants": float(plat[mieux].mean()) if bool(mieux.any()) else None,
            "platitude_perdants": float(plat[~mieux].mean()) if bool((~mieux).any()) else None,
            "plat_som_g": float(plat[mieux].sum()), "plat_n_g": int(mieux.sum()),
            "plat_som_p": float(plat[~mieux].sum()), "plat_n_p": int(n - int(mieux.sum()))}


def agreger(source: str, limite: int = 0) -> dict:
    from safetensors import safe_open
    fichiers = sorted(f for f in os.listdir(source) if f.endswith(".safetensors"))
    par_tenseur, par_couche = {}, defaultdict(lambda: {"blocs": 0, "mse6": 0.0, "mse4": 0.0, "linf6": 0.0, "linf4": 0.0})
    tot = {"blocs": 0, "mse6": 0.0, "mse4": 0.0, "linf6": 0.0, "linf4": 0.0, "part4_blocs": 0.0,
           "clamp4_blocs": 0.0, "linf_pire_blocs": 0.0, "plat_som_g": 0.0, "plat_n_g": 0,
           "plat_som_p": 0.0, "plat_n_p": 0}
    vus = 0
    for fn in fichiers:
        with safe_open(os.path.join(source, fn), framework="pt", device="cpu") as fh:
            for cle in fh.keys():
                if not (cle.endswith(".weight") and ("proj" in cle or "experts" in cle or "lm_head" in cle)):
                    continue
                w = fh.get_tensor(cle)
                if w.dim() < 2 or w.numel() < 4096:
                    continue
                r = par_bloc(w)
                par_tenseur[cle] = {"blocs": r["blocs"], "G": round((r["mse6"] - r["mse4"]) / max(r["mse6"], 1e-30), 6),
                                    "part4": round(r["part4"], 4), "clamp4": round(r["clamp4"], 5),
                                    "G_linf": round((r["linf6"] - r["linf4"]) / max(r["linf6"], 1e-30), 6),
                                    "linf_pire": round(r["linf_pire"], 5)}
                m = RE_COUCHE.search(cle)
                if m:
                    c = par_couche[int(m.group(1))]
                    for k_ in ("blocs", "mse6", "mse4", "linf6", "linf4"):
                        c[k_] += r[k_]
                for k_ in ("blocs", "mse6", "mse4", "linf6", "linf4"):
                    tot[k_] += r[k_]
                tot["part4_blocs"] += r["part4"] * r["blocs"]
                tot["clamp4_blocs"] += r["clamp4"] * r["blocs"]
                tot["linf_pire_blocs"] += r["linf_pire"] * r["blocs"]
                for k_ in ("plat_som_g", "plat_n_g", "plat_som_p", "plat_n_p"):
                    tot[k_] += r[k_]
                vus += 1
                if limite and vus >= limite:
                    break
        if limite and vus >= limite:
            break
    n = max(tot["blocs"], 1)
    out = {"source": source, "tenseurs": vus, "blocs": tot["blocs"],
           "G_global": round((tot["mse6"] - tot["mse4"]) / max(tot["mse6"], 1e-30), 6),
           "G_linf_global": round((tot["linf6"] - tot["linf4"]) / max(tot["linf6"], 1e-30), 6),
           "part4": round(tot["part4_blocs"] / n, 4), "clamp4": round(tot["clamp4_blocs"] / n, 5),
           "linf_pire": round(tot["linf_pire_blocs"] / n, 5),
           "platitude_gagnants": round(tot["plat_som_g"] / tot["plat_n_g"], 4) if tot["plat_n_g"] else None,
           "platitude_perdants": round(tot["plat_som_p"] / tot["plat_n_p"], 4) if tot["plat_n_p"] else None,
           "par_couche": {str(c): round((v["mse6"] - v["mse4"]) / max(v["mse6"], 1e-30), 6) for c, v in sorted(par_couche.items())},
           "par_tenseur": par_tenseur}
    out.update(verdict(out))
    return out


def verdict(r: dict) -> dict:
    """Jugé contre le scellé E (KL 4sur6 1,39 > max6 0,87 : 4sur6 PERD)."""
    g, linf_pire, clamp = r["G_global"], r["linf_pire"], r["clamp4"]
    if g < -1e-9:
        v = f"INVALIDE : G global {g} < 0 — impossible par construction (4sur6 prend le meilleur des deux par bloc), l instrument est faux"
    elif linf_pire >= 0.01 or clamp >= 0.01:
        v = (f"D ACCORD avec le scellé E : G = {g:.4f} > 0 par CONSTRUCTION (règle de sélection, ne prédit rien), "
             f"mais l erreur maximale augmente sur {linf_pire:.2%} des blocs et {clamp:.2%} saturent — "
             f"c est ce que la KL paie (4sur6 : 1,39 contre 0,87). La MSE par bloc n est PAS un prédicteur de qualité.")
    else:
        v = (f"RÉFUTÉ comme prédicteur : G = {g:.4f} > 0 et rien n augmente (L∞ pire sur {linf_pire:.2%} des blocs, "
             f"saturation {clamp:.2%}) — il prédirait 4sur6 gagnant, contre la mesure (KL 1,39 > 0,87)")
    return {"verdict": v, "scelle_E": {"kl_4sur6": 1.39, "kl_max6": 0.87, "4sur6_gagne": False}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="dossier des poids bf16 (la source de la conversion)")
    ap.add_argument("--limite", type=int, default=0, help="n tenseurs au plus (0 = tous)")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = agreger(a.source, a.limite)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[4sur6] {r['tenseurs']} tenseurs, {r['blocs']} blocs · G global {r['G_global']:.4f} · "
          f"G(L∞) {r['G_linf_global']:.4f} · amax/4 retenu {r['part4']:.1%} · saturés {r['clamp4']:.2%} · "
          f"L∞ pire {r['linf_pire']:.2%}")
    pires = sorted(r["par_tenseur"].items(), key=lambda kv: -kv[1]["linf_pire"])[:5]
    for nom, t in pires:
        print(f"    {nom[:58]:58s} G {t['G']:+.4f}  part4 {t['part4']:.0%}  L∞ pire {t['linf_pire']:.1%}")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
