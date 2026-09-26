#!/usr/bin/env python3
"""Bead anticitoyen-vram-brd — cible réelle : A4 sur les projections
d'EXPERTS uniquement (gate/up/down du chemin MoE groupé), Qwen3-Coder-
30B-A3B-nvfp4. Routeur, attention et éventuel expert partagé restent A16.

Recadrage de chef (13/09/2026) : le noyau MMA de poste4
(`nvfp4_gemm_grouped_mma`) ne sert QUE le chemin MoE groupé — pas
q/k/v/o. Le seuil réel se joue donc sur ce sous-ensemble, pas sur les 7
genres mesurés jusqu'ici sur Llama-2-7B (dense, sans MoE). C'est ce
régime précis, s'il passe, qui ferait basculer `ACVRAM_MOE_MMA` à 1 par
défaut.

Prédiction et seuil scellés dans revue/prediction-moe-experts-a4.md,
AVANT mesure.

Deux régimes, sur le MÊME dossier Qwen3-Coder-30B-A3B-nvfp4 :

  a16          témoin — mesure la PPL étalon de CE modèle, dans CE
               régime (pas de valeur archivée réutilisée : aucune PPL
               publiée pour ce dossier précis n'existe encore dans le
               dépôt — voir revue/prediction-moe-experts-a4.md).
  moe-a4       fake-quant E2M1 sur gate_proj/up_proj/down_proj des
               EXPERTS MoE uniquement (via `installer_hooks_moe_experts`,
               qui monkeypatch `MoEBlock._grouped` — les experts
               n'appellent jamais `QuantLinear.forward()`), routeur et
               attention intacts en A16.

Rien ne part sur la carte sans --pour-de-vrai.

LIMITE ASSUMÉE : mesure en un seul passage par régime (pas de jumelles),
comme les campagnes A4/A8/SmoothQuant précédentes de ce chantier — le
temps de carte est partagé entre plusieurs sessions. Si le résultat est
proche du seuil, une seconde passe est recommandée avant de conclure.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.hooks_activations_a4 import (installer_hooks_moe_experts,  # noqa: E402
                                         retirer_hooks)
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
DOSSIER = BASE / "Qwen3-Coder-30B-A3B-nvfp4"

CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
CORPUS_SHA = "e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"
SEUIL_RELATIF = 0.01

EVAL_KW = dict(window=2048, stride=2048, max_tokens=344064, min_context=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true")
    ap.add_argument("--regimes", default="a16,moe-a4")
    ap.add_argument("--sortie",
                    default=str(sorties() / "moe-experts-a4-qwen3-coder"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    connus = {"a16", "moe-a4"}
    inconnus = [r for r in regimes if r not in connus]
    if inconnus:
        print(f"ECHEC / CAUSE: regimes inconnus {inconnus}, attendu {sorted(connus)}")
        return 2

    print("BEAD anticitoyen-vram-brd — cible reelle : MoE experts A4, "
          "Qwen3-Coder-30B-A3B-nvfp4")
    print(f"  dossier      {DOSSIER}")
    print(f"  corpus       {CORPUS.name}")
    print(f"  seuil        +{SEUIL_RELATIF*100:.0f} % relatif au temoin a16 "
          f"MESURE ICI (aucun etalon archive pour ce dossier)")
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
        print(f"ECHEC / CAUSE: corpus de sha {sha[:16]} au lieu de {CORPUS_SHA[:16]}")
        return 2

    from acvram.evaluate import perplexity

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    fichier_resultats = sortie / "resultats.json"
    resultats = {}
    if fichier_resultats.exists():
        try:
            resultats = json.loads(fichier_resultats.read_text()).get("resultats", {})
        except (json.JSONDecodeError, OSError):
            resultats = {}

    for regime in regimes:
        t0 = time.time()

        def _brancher(model, _regime=regime):
            if _regime == "a16":
                _brancher.handles = []
                print(f"  {_regime} : temoin, aucun hook", flush=True)
                return
            handles, blocs = installer_hooks_moe_experts(model, "a4")
            if not blocs:
                raise RuntimeError(
                    "moe-a4 : aucun MoEBlock trouve — ce dossier n'est peut-etre "
                    "pas un modele a experts, ou le nommage de classe a change")
            print(f"  {_regime} : {len(blocs)} MoEBlock hookes (gate/up/down A4), "
                  f"routeur+attention en A16", flush=True)
            _brancher.handles = handles

        r = perplexity(str(DOSSIER), str(CORPUS), device=a.device,
                       apres_chargement=_brancher, **EVAL_KW)
        retirer_hooks(getattr(_brancher, "handles", []))
        # Deux régimes, un seul processus : sans ce nettoyage, le modèle du
        # régime précédent (~17 Gio pour Qwen3-Coder-30B en nvfp4) reste
        # résident tant que le ramasse-miettes cyclique n'est pas passé (les
        # hooks/parents forment des cycles), et le second chargement (deux
        # copies ~34 Gio > 32 Gio de la 5090) force un exil massif — observé
        # comme un plantage CUDA (ScatterGatherKernel, index hors bornes)
        # dans le chemin d'experts exilés, pas dans nos hooks. Llama-2-7B
        # (~7 Gio/copie) ne l'a jamais révélé.
        gc.collect()
        torch.cuda.empty_cache()

        ppl = r.perplexity
        resultats[regime] = {"ppl": ppl, "releve": r.to_dict()}
        print(f"  {regime} : PPL {ppl}  ({time.time() - t0:.0f} s)", flush=True)

        if regime == "moe-a4" and "a16" in resultats:
            ppl_a16 = resultats["a16"]["ppl"]
            ecart = 100 * (ppl / ppl_a16 - 1)
            seuil_ppl = round(ppl_a16 * (1 + SEUIL_RELATIF), 4)
            ok = ppl <= seuil_ppl
            print(f"  moe-a4 vs a16 mesure : ecart {ecart:+.3f} %  "
                  f"seuil {seuil_ppl} {'RESPECTE' if ok else 'DEPASSE'}", flush=True)
            resultats[regime]["ecart_pct_vs_a16_mesure"] = ecart
            resultats[regime]["seuil_ppl"] = seuil_ppl
            resultats[regime]["seuil_respecte"] = ok

        fichier_resultats.write_text(json.dumps(
            {"seuil_relatif": SEUIL_RELATIF, "resultats": resultats},
            indent=2, ensure_ascii=False))

    print(f"\nFAIT / TESTE: {fichier_resultats} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
