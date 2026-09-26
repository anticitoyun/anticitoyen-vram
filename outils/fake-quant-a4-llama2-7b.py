#!/usr/bin/env python3
"""Bead anticitoyen-vram-brd, étape 1 : coût en PPL du fake-quant des
ACTIVATIONS en E2M1 (W4A4) et E4M3 (W4A8, repli mxf8f6f4), sur le dossier
NVFP4 déjà converti de Llama-2-7B (`Llama-2-7b-nvfp4`).

Prédiction et seuil scellés AVANT mesure, à sec, dans
revue/prediction-a4-fakequant-llama2-7b.md : seuil de décision ≤ +1 % de
PPL relatif au témoin W4A16 (5,6102).

Trois régimes, TOUS sur le MÊME dossier converti (les poids ne changent
jamais — seul le fake-quant des activations d'entrée des 7 projections
change, via `outils.hooks_activations_a4`) :

  a16   témoin, aucun hook — doit reproduire 5,6102 (déjà mesuré par
        outils/campagne-quota.py ; une seule passe suffit ici pour
        vérifier la reproduction, pas la remesurer en entier)
  a4    fake_quantize_nvfp4_activation sur les 7 projections
  a8    fake_quantize_e4m3_activation  sur les 7 projections

Régime nommé : Llama-2-7B, wiki-gptq.txt, ctx 2048, protocole GPTQ — MEME
corpus que outils/campagne-quota.py (sha vérifié).

Rien ne part sur la carte sans --pour-de-vrai : sans lui, le plan
s'imprime et le script s'arrête (règle commune à toutes les campagnes de
ce dépôt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.hooks_activations_a4 import installer_hooks, retirer_hooks  # noqa: E402
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
DOSSIER = BASE / "Llama-2-7b-nvfp4"

# MEME CORPUS que outils/campagne-quota.py — deux campagnes qui ne lisent
# pas le meme fichier ne mesurent pas la meme chose.
CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
CORPUS_SHA = "e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"

# Etalon W4A16 deja mesure et publie (courbe-du-quota-deux-points.md,
# SYNTHESE-poste1-10-09.md, outils/campagne-quota.py) : plancher tout-nvfp4.
PPL_A16_ETALON = 5.6102
SEUIL_RELATIF = 0.01                          # +1 %, seuil scelle par chef
PPL_SEUIL = round(PPL_A16_ETALON * (1 + SEUIL_RELATIF), 4)

EVAL_KW = dict(window=2048, stride=2048, max_tokens=344064, min_context=0)

REGIMES = ("a16", "a4", "a8")


def gio(x: int) -> str:
    return f"{x / 1024**3:.4f} Gio"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true",
                    help="sans ce drapeau, imprime le plan et s'arrete")
    ap.add_argument("--regimes", default=",".join(REGIMES),
                    help="sous-ensemble, ex. a4,a8 pour sauter le temoin a16 "
                         "deja mesure (voir la note du docstring)")
    ap.add_argument("--sortie",
                    default=str(sorties() / "a4-fakequant"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    inconnus = [r for r in regimes if r not in REGIMES]
    if inconnus:
        print(f"ECHEC / CAUSE: regimes inconnus {inconnus}, attendu {REGIMES}")
        return 2

    print("BEAD anticitoyen-vram-brd, etape 1 — fake-quant activations A4/A8")
    print(f"  dossier      {DOSSIER}")
    print(f"  corpus       {CORPUS.name}")
    print(f"  etalon a16   PPL {PPL_A16_ETALON} (deja mesure, "
          f"courbe-du-quota-deux-points.md)")
    print(f"  seuil scelle PPL <= {PPL_SEUIL} (+{SEUIL_RELATIF*100:.0f} %)")
    for r in regimes:
        print(f"  regime {r}")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne. "
              "Relancer avec --pour-de-vrai quand la carte est libre.")
        return 0

    if not DOSSIER.exists():
        print(f"ECHEC / CAUSE: dossier absent {DOSSIER}")
        return 2
    if not CORPUS.exists():
        print(f"ECHEC / CAUSE: corpus absent {CORPUS}")
        return 2
    sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if sha != CORPUS_SHA:
        print(f"ECHEC / CAUSE: corpus de sha {sha[:16]} au lieu de "
              f"{CORPUS_SHA[:16]}")
        return 2

    from acvram.evaluate import perplexity

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    # FUSIONNE avec un resultats.json existant, jamais un remplacement pur :
    # le verrou de carte se rend ENTRE deux regimes lances separement (pour
    # laisser passer une autre session), et deux invocations successives de
    # ce script sur le meme --sortie doivent ACCUMULER, pas s'ecraser l'une
    # l'autre. Manque le 13/09 : un premier lancement a4 seul, suivi d'un
    # second a8 seul, a fait disparaitre le resultat a4 du fichier — corrige
    # ici, le journal (a4.log/a8.log) portait heureusement les deux chiffres.
    resultats = {}
    fichier_resultats = sortie / "resultats.json"
    if fichier_resultats.exists():
        try:
            resultats = json.loads(fichier_resultats.read_text()).get("resultats", {})
        except (json.JSONDecodeError, OSError):
            resultats = {}
    for regime in regimes:
        t0 = time.time()

        def _brancher(model, _regime=regime):
            handles, trouves = installer_hooks(model, _regime)
            if _regime != "a16" and not trouves:
                raise RuntimeError(
                    f"regime {_regime} : aucune projection trouvee, le "
                    f"fake-quant ne s'appliquerait a rien")
            print(f"  {_regime} : {len(trouves)} projections hookees "
                  f"({'aucune, temoin' if _regime == 'a16' else 'fake-quant actif'})",
                  flush=True)
            _brancher.handles = handles

        r = perplexity(str(DOSSIER), str(CORPUS), device=a.device,
                       apres_chargement=_brancher, **EVAL_KW)
        retirer_hooks(getattr(_brancher, "handles", []))

        ppl = r.perplexity
        ecart = 100 * (ppl / PPL_A16_ETALON - 1)
        ok = ppl <= PPL_SEUIL
        print(f"  {regime} : PPL {ppl}  ecart {ecart:+.3f} %  "
              f"seuil {'RESPECTE' if ok else 'DEPASSE'}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        resultats[regime] = {"ppl": ppl, "ecart_pct": ecart,
                             "seuil_respecte": ok,
                             "releve": r.to_dict()}

    if "a16" in resultats:
        dp16 = abs(resultats["a16"]["ppl"] - PPL_A16_ETALON)
        if dp16 > 0.01:
            print(f"\nATTENTION : le temoin a16 rend {resultats['a16']['ppl']} "
                  f"au lieu de {PPL_A16_ETALON} attendu (ecart {dp16:.4f} PPL) "
                  f"— les regimes a4/a8 de CETTE campagne ne sont pas "
                  f"attribuables tant que ce temoin ne reproduit pas "
                  f"l'etalon.", flush=True)

    (sortie / "resultats.json").write_text(json.dumps(
        {"seuil_ppl": PPL_SEUIL, "ppl_a16_etalon": PPL_A16_ETALON,
         "resultats": resultats}, indent=2, ensure_ascii=False))
    print(f"\nFAIT / TESTE: {sortie / 'resultats.json'} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
