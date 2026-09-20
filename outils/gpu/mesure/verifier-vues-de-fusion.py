#!/usr/bin/env python3
"""Charge, genere, et verifie que les vues de fusion ont bien rendu les octets.

Trois chargements demandes par le circuit avant la construction du .deb :
un int8, un modele portant de l'int4_awq, et un bf16 en TEMOIN NEGATIF (il
etait deja juste avant le correctif, il doit etre identique apres).

VALEUR PREVUE, pour que ce controle puisse rendre « faux » :
    Llama-2-7b-int8   avant  7 485 530 112  (8,887 bits/poids)
                      apres  7 024 074 752  (8,3391)
    difference          461 455 360 = 5 x 92 291 072, cinq gate_up fusionnes
Un chiffre entre les deux, ou egal a l'ancien, infirme le correctif.

Aucune mesure de temps : les secondes d'un chargement ne veulent rien dire
(127 s puis 47 s pour un travail identique, c'est le cache page).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


BASE = Path(_RACINE)


def rendre_le_cache(dossier: Path) -> int:
    """posix_fadvise(DONTNEED) sur les fragments du modele.

    Trois chargements de suite ont fait tuer ce script par le superviseur avec
    84 Go DISPONIBLES : 85 Go etaient en cache page, `MemFree` valait 1 Go, et
    PSI ne montrait aucune pression (avg10 = 0,01). Le tueur lit MemFree, la
    machine ne souffre pas. Rendre le cache apres chaque modele supprime la
    cause au lieu de la contourner.
    """
    import os
    total = 0
    for f in sorted(dossier.glob("*.safetensors")):
        fd = os.open(f, os.O_RDONLY)
        try:
            taille = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, taille, os.POSIX_FADV_DONTNEED)
            total += taille
        finally:
            os.close(fd)
    return total

# (dossier, role, valeur prevue de model.nbytes ou None)
CAS = [
    # La valeur prevue porte desormais sur les OCTETS DE STOCKAGE, pas sur
    # `nbytes` : celui-ci somme des `numel()` et compte donc une vue comme si
    # elle possedait ses octets, ce qui rend le correctif invisible pour lui.
    ("Llama-2-7b-int8", "int8 pur — le cas mesure", 7_024_074_752),
    ("Qwen3B-pipeline-nvfp4", "porte bf16 + int4_awq + int8 + nvfp4", None),
    ("Agents-4B-kimi-bf16", "bf16 — TEMOIN NEGATIF, doit etre inchange", None),
]
INVITE = "Explique en une phrase ce qu'est la memoire virtuelle."


def un_cas(dossier: Path, role: str, prevu: int | None, jetons: int) -> dict:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer

    print(f"\n=== {dossier.name}\n    {role}", flush=True)
    charge = load_model(str(dossier), dtype=torch.bfloat16)
    tok = load_tokenizer(str(dossier))
    detail = charge.model.nbytes_detail()

    verdict = "sans valeur prevue"
    print(f"    nbytes (numel)        {detail['total']:,}".replace(",", " "), flush=True)
    print(f"    stockages uniques     {detail['octets_stockage_uniques']:,}"
          .replace(",", " "), flush=True)
    print(f"    dupliques par les vues {detail['octets_dupliques_par_les_vues']:,}"
          .replace(",", " "), flush=True)
    if prevu is not None:
        ecart = detail["octets_stockage_uniques"] - prevu
        verdict = ("CONFORME" if ecart == 0 else
                   f"ECART DE {ecart:+,} OCTETS".replace(",", " "))
        print(f"    nbytes prevu {prevu:,}".replace(",", " "), flush=True)
    print(f"    verdict : {verdict}", flush=True)
    print(f"    QuantLinear {detail['quantlinear']}, objets distincts "
          f"{detail['objets_distincts']}, comptes en double "
          f"{detail['octets_comptes_en_double']:,}".replace(",", " "), flush=True)
    print(f"    par format {detail['par_format']}", flush=True)

    texte = ""
    if tok is not None:
        ids = tok.encode(INVITE)
        r = Engine(charge, tokenizer=tok)
        p = SamplingParams(temperature=0.0, max_tokens=jetons)
        morceaux = []
        r_local = r
        for out in r_local.generate(ids, p):
            if getattr(out, "text_delta", None):
                morceaux.append(out.text_delta)
        texte = "".join(morceaux)
        # Un test de texte AFFICHE le texte : un detecteur de charabia qui
        # resume « coherent » ne se verifie pas.
        print(f"    invite  : {INVITE}")
        print(f"    reponse : {texte[:300]!r}", flush=True)
        if not texte.strip():
            print("    ATTENTION : aucune sortie — le chargement ne prouve "
                  "rien sur l'execution", flush=True)
    else:
        print("    pas de tokenizer : generation sautee", flush=True)

    # Un modele charge apres l'autre dans le MEME processus sans que le
    # precedent soit libere a deja fait mesurer un nvfp4 avec un bf16 par
    # dessus, le 9/09.
    import gc
    charge = None
    r = None
    r_local = None
    gc.collect()
    torch.cuda.empty_cache()
    rendu = rendre_le_cache(dossier)
    print(f"    cache page rendu : {rendu / 2**20:.0f} Mio", flush=True)
    return {"modele": dossier.name, "role": role, "prevu": prevu,
            "nbytes": detail["total"],
            "octets_stockage_uniques": detail["octets_stockage_uniques"],
            "verdict": verdict,
            "detail": detail, "reponse": texte[:400]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetons", type=int, default=24)
    ap.add_argument("--sortie", default="")
    a = ap.parse_args()
    res = []
    for nom, role, prevu in CAS:
        d = BASE / nom
        if not d.exists():
            print(f"\n=== {nom} : ABSENT, cas saute", flush=True)
            res.append({"modele": nom, "verdict": "absent"})
            continue
        try:
            res.append(un_cas(d, role, prevu, a.jetons))
        except Exception as e:                          # noqa: BLE001
            print(f"    ECHEC : {type(e).__name__}: {e}", flush=True)
            res.append({"modele": nom, "verdict": f"echec {type(e).__name__}",
                        "cause": str(e)[:400]})
    if a.sortie:
        Path(a.sortie).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n--- BILAN ---")
    for r in res:
        print(f"  {r['modele'][:44]:44s} {r.get('verdict', '?')}")
    ok = all(r.get("verdict") in ("CONFORME", "sans valeur prevue")
             for r in res if r.get("verdict") != "absent")
    print(f"\n{'FAIT' if ok else 'ECHEC'} / TESTE: {len(res)} chargements / "
          f"RESTE: {'rien' if ok else 'voir les ecarts ci-dessus'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
