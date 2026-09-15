#!/usr/bin/env python3
"""Bead anticitoyen-vram-brd — cible réelle, addendum 13/09 : PPL avec le
VRAI noyau MMA W4A4 (`nvfp4_gemm_grouped_mma`), pas un fake-quant simulé.

`outils/moe-experts-a4-qwen3-coder.py` (hooks sur `_grouped`) s'est révélé
inerte à window=2048 : ce prefill prend `_forward_prefill_grouped`, jamais
`_grouped` (voir revue/prediction-moe-experts-a4.md, addendum). poste4 :
`ACVRAM_MOE_MMA=1` fait passer ce chemin par le noyau MMA réel — mais
`_MOE_MMA` est figé à l'import de `acvram/engine/model.py`, donc un
régime par PROCESSUS, pas un hook togglable en cours de route.

Deux appels séparés :
  ACVRAM_MOE_MMA=0 python outils/moe-mma-reel-qwen3-coder.py --pour-de-vrai --regime a16
  ACVRAM_MOE_MMA=1 python outils/moe-mma-reel-qwen3-coder.py --pour-de-vrai --regime mma-reel

Seuil et prédiction : revue/prediction-moe-experts-a4.md (addendum).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
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
    ap.add_argument("--regime", required=True, choices=["a16", "mma-reel"])
    ap.add_argument("--sortie",
                    default=str(sorties() / "moe-mma-reel-qwen3-coder"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    regime = a.regime

    attendu_mma = os.environ.get("ACVRAM_MOE_MMA", "0") == "1"
    voulu_mma = (regime == "mma-reel")
    if attendu_mma != voulu_mma:
        print(f"ECHEC / CAUSE: regime {regime!r} exige ACVRAM_MOE_MMA="
              f"{'1' if voulu_mma else '0'}, or ACVRAM_MOE_MMA="
              f"{os.environ.get('ACVRAM_MOE_MMA', '0')!r}")
        return 2

    print(f"BEAD anticitoyen-vram-brd — noyau MMA reel, Qwen3-Coder-30B-A3B-nvfp4")
    print(f"  dossier      {DOSSIER}")
    print(f"  regime       {regime}  (ACVRAM_MOE_MMA={os.environ.get('ACVRAM_MOE_MMA', '0')})")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne.")
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

    from acvram import kernels
    from acvram.evaluate import perplexity

    if voulu_mma:
        ext = kernels.get_extension()
        if not (ext is not None and hasattr(ext, "nvfp4_gemm_grouped_mma")
                and ext.nvfp4_gemm_grouped_mma_disponible()):
            print("ECHEC / CAUSE: noyau nvfp4_gemm_grouped_mma indisponible "
                  "sur cette extension")
            return 2

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    fichier_resultats = sortie / "resultats.json"
    resultats = {}
    if fichier_resultats.exists():
        try:
            resultats = json.loads(fichier_resultats.read_text()).get("resultats", {})
        except (json.JSONDecodeError, OSError):
            resultats = {}

    t0 = time.time()
    r = perplexity(str(DOSSIER), str(CORPUS), device=a.device, **EVAL_KW)
    debit = r.tokens / r.seconds if r.seconds else 0.0
    print(f"  {regime} : PPL {r.perplexity}  débit {debit:.0f} j/s  "
          f"({time.time() - t0:.0f} s)", flush=True)
    resultats[regime] = {"ppl": r.perplexity, "debit_js": debit, "releve": r.to_dict()}

    if regime == "mma-reel" and "a16" in resultats:
        ppl_a16 = resultats["a16"]["ppl"]
        ecart = 100 * (r.perplexity / ppl_a16 - 1)
        seuil_ppl = round(ppl_a16 * (1 + SEUIL_RELATIF), 4)
        ok = r.perplexity <= seuil_ppl
        print(f"  mma-reel vs a16 mesure : ecart {ecart:+.3f} %  "
              f"seuil {seuil_ppl} {'RESPECTE' if ok else 'DEPASSE'}", flush=True)
        resultats[regime]["ecart_pct_vs_a16_mesure"] = ecart
        resultats[regime]["seuil_ppl"] = seuil_ppl
        resultats[regime]["seuil_respecte"] = ok
        debit_a16 = resultats["a16"].get("debit_js", 0.0)
        if debit_a16 and abs(debit - debit_a16) / debit_a16 < 0.05:
            print(f"  ATTENTION : débit quasi identique à a16 ({debit:.0f} vs "
                  f"{debit_a16:.0f} j/s) — le noyau MMA n'est peut-être pas pris",
                  flush=True)

    fichier_resultats.write_text(json.dumps(
        {"seuil_relatif": SEUIL_RELATIF, "resultats": resultats},
        indent=2, ensure_ascii=False))
    print(f"\nFAIT / TESTE: {fichier_resultats} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
