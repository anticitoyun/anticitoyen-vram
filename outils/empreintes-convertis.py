#!/usr/bin/env python3
"""Empreinte de CONTENU des convertis acvram : 3 fenetres de 2 Mio par
safetensors (debut/milieu/fin), plus la taille. Discrimine deux modeles de
meme architecture et meme format, ce que la taille seule ne fait pas."""
import os, hashlib, sys
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

A = MODELES
FEN = 2 << 20

def empreinte(d):
    h = hashlib.sha256()
    fs = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    if not fs: return ""
    for f in fs:
        p = os.path.join(d, f)
        n = os.path.getsize(p)
        h.update(f"{f}:{n};".encode())
        with open(p, "rb") as fh:
            for off in (0, max(0, n // 2), max(0, n - FEN)):
                fh.seek(off); h.update(fh.read(FEN))
    return h.hexdigest()[:24]

noms = sorted(x for x in os.listdir(A) if os.path.isdir(os.path.join(A, x)))
for i, nom in enumerate(noms, 1):
    try: e = empreinte(os.path.join(A, nom))
    except Exception as ex: e = "ERREUR:" + str(ex)[:30]
    print(f"{e}\t{nom}", flush=True)
    print(f"[{i}/{len(noms)}] {nom}", file=sys.stderr, flush=True)
