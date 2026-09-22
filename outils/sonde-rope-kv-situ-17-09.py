#!/usr/bin/env python3
"""Sonde en situ (carte, ≈ 90 s) : ce que rope_kv écrit dans le cache sous
REJEU DE GRAPHE, comparé à ce que kv_write_int8 écrit au même pas.

Deux moteurs sur le même préfixe et les mêmes jetons forcés (teacher forcing,
graphes activés) : A = chemin actuel (rope_inplace + kv_write_int8), B =
`ACVRAM_ROPE_KV=1`. Après chaque pas de décodage, on lit dans les caches de la
couche 0 le créneau du jeton qui vient d'être écrit et l'on compare codes et
échelles : IDENTIQUE, DEMI-ENTIER (±1 code à un demi-entier exact),
DIFFÉRENT (valeurs autres : le noyau calcule faux), NUL (rien d'écrit : le
noyau n'est pas rejoué par le graphe), ou STALE (contenu d'un autre pas).
On imprime les 16 premiers pas puis le décompte sur `PAS` pas — la première
catégorie non IDENTIQUE dit où chercher.

    ACVRAM_KV_FORMAT=int8 ACVRAM_MOE_MMA=0 ACVRAM_MOE_DECODE_MMA=0 ACVRAM_NARROW_GEMM=1 \
        outils/carte.sh .venv/bin/python outils/sonde-rope-kv-situ-17-09.py [prefixe=256] [pas=64]

(le régime classé de la cellule F ; sans effet sur l'écriture du cache, mais
c'est le régime jugé). Ne pose pas ACVRAM_ROPE_KV : la sonde bascule le module.
"""
import os
import sys

import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ACVRAM_KV_FORMAT", "int8")
MODEL = os.environ.get("ACVRAM_MODELE_MESURE", _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4")
PREFIXE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
PAS = int(sys.argv[2]) if len(sys.argv) > 2 else 64

from acvram.engine import model as M                                   # noqa: E402
from acvram.engine.loader import load_model                            # noqa: E402
from acvram.engine.runner import Engine                                # noqa: E402
from acvram.engine.sampler import SamplingParams                       # noqa: E402
from acvram.server.chat import load_tokenizer                          # noqa: E402


def moteur(rope_kv: bool):
    M._ROPE_KV = rope_kv
    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=PREFIXE + PAS + 64)
    tok = load_tokenizer(MODEL)
    eng = Engine(loaded, tok, max_batch_size=1, max_model_len=PREFIXE + PAS + 64, enable_cuda_graphs=True)
    eng._eos = set()
    return eng, loaded


def decode(eng, ids, releves):
    """Force les jetons `ids[PREFIXE:]` ; après chaque pas, relève (codes k, échelle k, codes v) du créneau écrit."""
    c0 = next(iter(eng.model.caches.values()))
    etat = {"pos": PREFIXE - 1}
    seq_ref = {}

    def sample_force(logits, seqs):
        p = etat["pos"]
        cible = ids[p + 1] if p + 1 < len(ids) else ids[p]
        etat["pos"] = p + 1
        seq_ref["s"] = seqs[0]
        return torch.tensor([cible], device=logits.device, dtype=torch.long), torch.zeros(1, device=logits.device)

    eng._sample_only = sample_force
    eng.add_request(ids[:PREFIXE], SamplingParams(temperature=0.0, max_tokens=PAS), request_id="s0")
    while eng.running or eng.waiting:
        eng.step()
        s = seq_ref.get("s")
        if s is None or not s.blocks:
            continue
        # position du jeton ENTRÉ à ce pas (donc écrit dans le cache) : le
        # dernier échantillonné (output_ids[-1]) n'y est pas encore — Laure :
        # IndexError sur le bloc 16 au premier pas, n = 256 alors que 256
        # n'est écrit qu'au pas suivant
        n = len(s.prompt_ids) + len(s.output_ids) - 2
        if n < PREFIXE or n // 16 >= len(s.blocks):
            continue
        if releves and releves[-1][0] == n:
            continue
        blk, off = s.blocks[n // 16], n % 16
        releves.append((n, c0.k[blk, off].clone(), c0.k_scale[blk, off].clone(), c0.v[blk, off].clone()))


def classer(a, b):
    (n, ka, sa, va), (_, kb, sb, vb) = a, b
    if torch.equal(ka, kb) and torch.equal(sa, sb) and torch.equal(va, vb):
        return "IDENTIQUE"
    if not kb.any() and not vb.any():
        return "NUL"
    if torch.equal(sa, sb):
        d = (ka.int() - kb.int()).abs()
        dv = (va.int() - vb.int()).abs()
        if int(d.max()) <= 1 and int(dv.max()) <= 1 and int((d > 0).sum() + (dv > 0).sum()) <= 4:
            return "DEMI-ENTIER"
        return "DIFFÉRENT"
    return "DIFFÉRENT(échelle)"


def main():
    tok = load_tokenizer(MODEL)
    texte = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratchpad", "corpus-prive", "tranches-coder", "tranche0.txt"), errors="ignore").read()
    ids = tok.encode(texte)[:PREFIXE + PAS + 2]
    relA, relB = [], []
    engA, _ = moteur(False)
    print("A", engA.regime_ligne(), flush=True)
    decode(engA, ids, relA)
    del engA; torch.cuda.empty_cache()
    engB, _ = moteur(True)
    print("B", engB.regime_ligne(), flush=True)
    decode(engB, ids, relB)
    compte = {}
    for i, (a, b) in enumerate(zip(relA, relB)):
        cat = classer(a, b)
        # STALE : le créneau B ressemble à un autre pas de A
        if cat.startswith("DIFF"):
            for a2 in relA:
                if a2[0] != a[0] and torch.equal(a2[1], b[1]):
                    cat = f"STALE(pas {a2[0]})"; break
        compte[cat] = compte.get(cat, 0) + 1
        if i < 16 or not cat.startswith("IDENT"):
            print(f"pas {a[0]:5d} : {cat}" + ("" if cat == "IDENTIQUE" else
                  f"  |k diff max {int((a[1].int() - b[1].int()).abs().max())}, échelle A {float(a[2]):.5g} B {float(b[2]):.5g}"), flush=True)
    print("DÉCOMPTE", compte, flush=True)


if __name__ == "__main__":
    main()
