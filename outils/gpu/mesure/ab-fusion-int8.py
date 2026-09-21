#!/usr/bin/env python3
"""Ce que la fusion rapporte sur un int8 CALIBRE, dans le regime servi.

Ordre pose par le circuit le 10/09 : mesurer le GAIN avant de calculer le PRIX
d'un `alpha` commun. Le prix est calculable exactement — la recherche AWQ
evalue deja les 21 valeurs de la grille — et c'est le piege : un prix precis
face a un gain suppose fait pencher la decision du cote du chiffre qu'on a.

Le seul chiffre existant, +2,60 %, est mesure en BF16 PUR, ou 100 % des groupes
fusionnent. Sur Llama-2-7b-int8, `_scaler_commun` n'en accepte que 5 sur 64
(7,8 %), parce que la recherche choisit un exposant par tenseur. Si le gain est
proportionnel, il vaut deux dixiemes de pourcent et la question se ferme sans
rien payer. S'il est plus grand — les groupes qui fusionnent etant peut-etre les
plus chauds — le prix merite d'etre calcule.

PROTOCOLE
  ABBA : A1 B1 B2 A2. Alterner ne corrige pas la derive thermique, ABBA si
  (+3 W en 12 passages mesures le 9/09).
  UNE VALEUR PAR PROCESSUS : un modele charge par-dessus un autre dans le meme
  processus a deja fabrique un faux resultat.
  Le cache page est rendu entre deux bras.
  Dispersion INTRA-BRAS publiee : sans elle, un ecart n'est pas interpretable.
  Le nombre de groupes REELLEMENT fusionnes est releve dans chaque bras — c'est
  le temoin : si le bras « sans fusion » en fusionne un seul, la manche est
  sans objet.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


RACINE = Path(__file__).resolve().parent.parent
BASE = Path(_RACINE)
PY = os.environ.get("ACVRAM_PY", f"{Path(__file__).resolve().parents[3]}/../../anticitoyen-vram/.venv/bin/python")

SONDE = r'''
import json, os, sys, torch
sys.path.insert(0, %(racine)r)
from acvram.engine.loader import load_model
from acvram.bench import bench_decode
charge = load_model(%(modele)r, dtype=torch.bfloat16)
# TEMOIN : combien de groupes ont REELLEMENT fusionne dans ce bras
from acvram.engine.model import MLP, Attention
fusionnes = 0
for m in charge.model.modules():
    if isinstance(m, MLP) and getattr(m, "gate_up", None) is not None:
        fusionnes += 1
    if isinstance(m, Attention) and getattr(m, "qkv_proj", None) is not None:
        fusionnes += 1
d = charge.model.nbytes_detail()
del charge
import gc; gc.collect(); torch.cuda.empty_cache()
r = bench_decode(%(modele)r, n_tokens=%(jetons)d)
# Le banc porte TROIS refus explicites (plan rejoue, prefill servi par le cache,
# EOS avant la fin) qui rendent un champ `refus` a la place d'un debit. Ne lire
# que `decode_tok_s` les transformait en None : un harnais qui jette la raison
# rend l'absence indistinguable d'une panne. On la remonte.
print("RESULTAT " + json.dumps({
    "decode_tok_s": r.get("decode_tok_s"),
    "refus": r.get("refus"),
    "groupes_fusionnes": fusionnes,
    "octets_stockage": d["octets_stockage_uniques"],
    "dupliques_par_vues": d["octets_dupliques_par_les_vues"]}))
'''


def rendre_le_cache(dossier: Path) -> None:
    for f in sorted(dossier.glob("*.safetensors")):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def un_bras(modele: Path, sans_fusion: bool, jetons: int, etiquette: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RACINE)
    env["ACVRAM_KERNEL_CACHE"] = env.get("ACVRAM_KERNEL_CACHE", "/tmp/oceane-noyaux")
    env["ACVRAM_CUDA_HOME"] = "/usr/local/cuda-13.2"
    # Le manifeste vise ['cuda:0'], la machine en expose deux : le plan est
    # rejoue et le banc REFUSE, a raison — le debit ne porterait pas sur la
    # configuration demandee. On epingle plutot que de forcer avec
    # ACVRAM_BANC_ACCEPTE_REPLAN : forcer mesurerait un autre placement.
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if sans_fusion:
        env["ACVRAM_SANS_FUSION"] = "1"
    else:
        env.pop("ACVRAM_SANS_FUSION", None)
        env.pop("ACVRAM_SANS_FUSION_BF16", None)
    code = SONDE % {"racine": str(RACINE), "modele": str(modele), "jetons": jetons}
    p = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True,
                       timeout=1800)
    ligne = next((l for l in p.stdout.splitlines() if l.startswith("RESULTAT ")),
                 None)
    if ligne is None:
        print(f"  {etiquette} : ECHEC\n{p.stdout[-600:]}\n{p.stderr[-600:]}",
              flush=True)
        return {"bras": etiquette, "echec": True}
    r = json.loads(ligne[len("RESULTAT "):])
    if r.get("refus"):
        print(f"  {etiquette} : LE BANC REFUSE — {r['refus']}", flush=True)
        r["echec"] = True
        return r
    r["bras"] = etiquette
    r["sans_fusion"] = sans_fusion
    print(f"  {etiquette:4s} {'sans' if sans_fusion else 'avec'} fusion : "
          f"{r['decode_tok_s']} jetons/s, {r['groupes_fusionnes']} groupes "
          f"fusionnes, {r['dupliques_par_vues'] / 2**20:.0f} Mio en vues",
          flush=True)
    rendre_le_cache(modele)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="Llama-2-7b-int8")
    ap.add_argument("--jetons", type=int, default=256)
    ap.add_argument("--sortie", default="")
    a = ap.parse_args()
    d = BASE / a.modele
    if not d.exists():
        print(f"ECHEC / CAUSE: {d} absent")
        return 2
    print(f"A/B FUSION sur {a.modele}, ABBA, {a.jetons} jetons par bras")
    res = [un_bras(d, False, a.jetons, "A1"),
           un_bras(d, True, a.jetons, "B1"),
           un_bras(d, True, a.jetons, "B2"),
           un_bras(d, False, a.jetons, "A2")]
    if any(r.get("echec") for r in res):
        print("ECHEC / CAUSE: un bras n'a pas rendu de chiffre / "
              "SUITE: lire la sortie ci-dessus")
        return 1
    A = [r["decode_tok_s"] for r in res if not r["sans_fusion"]]
    B = [r["decode_tok_s"] for r in res if r["sans_fusion"]]
    gA, gB = [r["groupes_fusionnes"] for r in res if not r["sans_fusion"]], \
        [r["groupes_fusionnes"] for r in res if r["sans_fusion"]]
    print(f"\n  avec fusion  {A}  moyenne {sum(A)/2:.2f}  "
          f"dispersion {abs(A[0]-A[1]):.2f}")
    print(f"  sans fusion  {B}  moyenne {sum(B)/2:.2f}  "
          f"dispersion {abs(B[0]-B[1]):.2f}")
    print(f"  groupes fusionnes : avec {gA}, sans {gB}")
    if any(g != 0 for g in gB):
        print("  MANCHE SANS OBJET : le bras « sans fusion » en fusionne "
              "encore. L'echappement n'a pas pris.")
        return 3
    if all(g == 0 for g in gA):
        print("  MANCHE SANS OBJET : le bras « avec fusion » n'en fusionne "
              "AUCUN. Il n'y a rien a mesurer sur ce modele.")
        return 3
    ecart = (sum(A) / 2) / (sum(B) / 2) - 1
    disp = max(abs(A[0] - A[1]), abs(B[0] - B[1])) / (sum(B) / 2)
    print(f"\n  GAIN DE LA FUSION : {100 * ecart:+.2f} %  "
          f"(dispersion intra-bras au plus {100 * disp:.2f} %)")
    if abs(ecart) <= disp:
        print("  NON CONCLUANT : l'ecart ne depasse pas la dispersion.")
    print(f"\n  Pour memoire, le meme gain en bf16 pur : +2,60 % avec 100 % "
          f"des groupes fusionnes ; ici {gA[0]} groupes.")
    if a.sortie:
        Path(a.sortie).write_text(json.dumps(
            {"bras": res, "gain": ecart, "dispersion_relative": disp,
             "modele": a.modele, "jetons": a.jetons}, indent=2,
            ensure_ascii=False))
    print(f"\nFAIT / TESTE: 4 bras ABBA, gain {100 * ecart:+.2f} % / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
