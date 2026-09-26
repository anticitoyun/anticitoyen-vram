#!/usr/bin/env python3
"""Bead anticitoyen-vram-brd, étape 1, suite : SmoothQuant + fake-quant A4,
balayage alpha, et isolation de down_proj seul.

Suite de outils/fake-quant-a4-llama2-7b.py (verdict :
revue/verdict-a4-fakequant-llama2-7b.md — A4 nu +2,58 %, seuil dépassé).
Prédiction et seuil scellés AVANT mesure, à sec, dans
revue/prediction-smoothquant-a4-sweep.md : seuil ≤ 5,6663 (+1 % vs A16),
inchangé.

Quatre régimes, tous sur le MÊME dossier `Llama-2-7b-nvfp4` (le poids
stocké sur DISQUE ne change jamais ; SmoothQuant replie une échelle dans
une COPIE en mémoire du poids, jamais persistée) :

  smooth-a0.50   SmoothQuant alpha=0,50 + fake-quant E2M1 sur les 7 genres
  smooth-a0.65   SmoothQuant alpha=0,65 + fake-quant E2M1 sur les 7 genres
  smooth-a0.80   SmoothQuant alpha=0,80 + fake-quant E2M1 sur les 7 genres
  downproj-seul  fake-quant E2M1 SUR down_proj UNIQUEMENT, les 6 autres
                 genres restent en A16 (aucun SmoothQuant sur ce régime :
                 question isolée — la dégradation A4 nue vient-elle
                 surtout de down_proj ?)
  moe-proj-seul  fake-quant E2M1 sur gate_proj+up_proj+down_proj (les
                 trois genres du MLP), q/k/v/o restent en A16. Recadrage
                 de chef (13/09) : le noyau MMA de poste4 ne sert QUE
                 ces trois projections (chemin MoE groupé) ; le seuil
                 réel se joue sur ce sous-ensemble, pas sur les 7 genres.

Statistiques d'activation (max|X| par canal d'entrée) relevées sur le
POINT DE CONTRÔLE HF D'ORIGINE (`Llama-2-7b-hf`, PAS le dossier converti :
la calibration se fait en pleine précision), avec
`acvram.quant.collect.collect_activation_stats`, 512 séquences — même
mécanisme que la calibration AWQ du convertisseur.

Rien ne part sur la carte sans --pour-de-vrai.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.hooks_activations_a4 import (installer_hooks_genres,  # noqa: E402
                                         installer_hooks_smoothquant,
                                         retirer_hooks)
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
DOSSIER = BASE / "Llama-2-7b-nvfp4"
SOURCE_HF = BASE / "Llama-2-7b-hf"

CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
CORPUS_SHA = "e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"
PPL_A16_ETALON = 5.6102
SEUIL_RELATIF = 0.01
PPL_SEUIL = round(PPL_A16_ETALON * (1 + SEUIL_RELATIF), 4)

EVAL_KW = dict(window=2048, stride=2048, max_tokens=344064, min_context=0)

ALPHAS = (0.5, 0.65, 0.8)
CALIB_SEQS = 512
CALIB_LEN = 512


def _calibrer(device: str):
    """Statistiques d'activation sur le point de controle HF d'origine —
    memes fonctions que la calibration AWQ du convertisseur."""
    from acvram.engine.config import load_model_spec
    from acvram.quant.collect import collect_activation_stats, load_calib_ids
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(str(SOURCE_HF), "llama2-7b")
    tok = load_tokenizer(str(SOURCE_HF))
    calib = load_calib_ids(tok, str(CORPUS), CALIB_SEQS, CALIB_LEN, spec.vocab_size)
    print(f"  calibration sur {len(calib)} sequences "
          f"({sum(len(c) for c in calib)} jetons)...", flush=True)
    return collect_activation_stats(str(SOURCE_HF), spec, calib, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true")
    ap.add_argument("--regimes", default="smooth-a0.50,smooth-a0.65,smooth-a0.80,downproj-seul,moe-proj-seul")
    ap.add_argument("--sortie",
                    default=str(sorties() / "smoothquant-a4-sweep"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    connus = {"smooth-a0.50", "smooth-a0.65", "smooth-a0.80", "downproj-seul",
              "moe-proj-seul"}
    inconnus = [r for r in regimes if r not in connus]
    if inconnus:
        print(f"ECHEC / CAUSE: regimes inconnus {inconnus}, attendu {sorted(connus)}")
        return 2

    print("BEAD anticitoyen-vram-brd, etape 1 (suite) — SmoothQuant + A4, "
          "isolation down_proj")
    print(f"  dossier      {DOSSIER}")
    print(f"  source HF    {SOURCE_HF} (calibration)")
    print(f"  corpus       {CORPUS.name}")
    print(f"  etalon a16   PPL {PPL_A16_ETALON}")
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
    if not SOURCE_HF.exists():
        print(f"ECHEC / CAUSE: source HF absente {SOURCE_HF}")
        return 2
    if not CORPUS.exists():
        print(f"ECHEC / CAUSE: corpus absent {CORPUS}")
        return 2
    sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if sha != CORPUS_SHA:
        print(f"ECHEC / CAUSE: corpus de sha {sha[:16]} au lieu de {CORPUS_SHA[:16]}")
        return 2

    from acvram.evaluate import perplexity

    besoin_calib = any(r.startswith("smooth-") for r in regimes)
    act_stats = _calibrer(a.device) if besoin_calib else None

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
            if _regime == "downproj-seul":
                handles, trouves = installer_hooks_genres(model, {"down_proj"}, "a4")
                if not trouves:
                    raise RuntimeError("downproj-seul : aucune couche trouvee")
                print(f"  {_regime} : {len(trouves)} down_proj hookes (A4), "
                      f"le reste en A16", flush=True)
            elif _regime == "moe-proj-seul":
                handles, trouves = installer_hooks_genres(
                    model, {"gate_proj", "up_proj", "down_proj"}, "a4")
                if not trouves:
                    raise RuntimeError("moe-proj-seul : aucune couche trouvee")
                print(f"  {_regime} : {len(trouves)} gate/up/down hookes (A4), "
                      f"q/k/v/o en A16", flush=True)
            else:
                alpha = float(_regime.split("a")[-1])
                handles, trouves, manques = installer_hooks_smoothquant(
                    model, act_stats, alpha)
                if manques:
                    raise RuntimeError(
                        f"{_regime} : {len(manques)} genres sans statistiques "
                        f"d'activation, calibration incomplete : {manques[:3]}")
                print(f"  {_regime} : {len(trouves)} projections repliees+hookees",
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
                             "seuil_respecte": ok, "releve": r.to_dict()}
        fichier_resultats.write_text(json.dumps(
            {"seuil_ppl": PPL_SEUIL, "ppl_a16_etalon": PPL_A16_ETALON,
             "resultats": resultats}, indent=2, ensure_ascii=False))

    print(f"\nFAIT / TESTE: {fichier_resultats} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
