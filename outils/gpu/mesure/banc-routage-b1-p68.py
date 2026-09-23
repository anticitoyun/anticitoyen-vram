#!/usr/bin/env python3
"""Pièce 68 : où partent les 6,3 µs de `_route_fusee_kernel` à b = 1 ?

Quatre bras sur la forme servie (T = 1, E = 128, K = 8), chronométrés sous
graphe CUDA comme le décodage les sert — un lancement isolé hors graphe
mesurerait la latence de lancement de l'hôte, pas celle que le pas paie.

  vide   : noyau Triton qui ne fait rien, même grille, même num_warps
           → le PLANCHER de lancement, celui qu'aucune réécriture ne rend
  probs  : logits → probabilités (softmax + somme), sans sélection
  servi  : `_route_fusee_kernel` tel quel, num_warps=1 (le chemin servi)
  warps  : le même à num_warps=4 et 8 — TÉMOIN DE COÛT, jamais un candidat :
           il change l'ordre de `tl.sum` du softmax, donc il n'est pas au bit.

Prédictions, réfutations et alarme : revue/oceane-piece68-routage-b1-23-09.md
(vide 1,5-2,5 · probs 2,5-3,5 · servi 6,0-6,6 ; R1 si vide ≥ 4,5 ; R2 si probs
≥ 5,0 ; R3 si warps ne gagne pas 1 µs ; A1 si servi s'écarte de 20 % des 6,3).

Usage : outils/carte.sh python outils/gpu/mesure/banc-routage-b1-p68.py [--rep 300] [--json S]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../..")
from acvram.kernels import route_prep as RP                                  # noqa: E402


@triton.jit
def _vide_kernel(p, n):
    """Ne fait rien, mais le compilateur ne peut pas le supprimer."""
    t = tl.program_id(0)
    if t >= n:
        tl.store(p + t, 0)


@triton.jit
def _probs_kernel(logits_ptr, out_ptr, E, BE: tl.constexpr):
    """Le softmax de `_route_fusee_kernel`, sans sélection : mêmes charges,
    même réduction, même disposition."""
    t = tl.program_id(0)
    i = tl.arange(0, BE)
    masque = i < E
    lg = tl.load(logits_ptr + t * E + i, mask=masque, other=float("-inf")).to(tl.float32)
    m = tl.max(lg, 0)
    ex = tl.exp(lg - m)
    probs = ex / tl.sum(ex, 0)
    tl.store(out_ptr + t * E + i, tl.where(masque, probs, 0.0), mask=masque)


INTRA = 200                       # lancements DANS le graphe (voir _chrono)


@triton.jit
def _select_kernel(logits_ptr, topw_ptr, topi_ptr, eid_ptr, usage_ptr, E, scale,
                   K: tl.constexpr, BE: tl.constexpr, ATOMIQUE: tl.constexpr,
                   STORE_BOUCLE: tl.constexpr):
    """`_route_fusee_kernel` (softmax, sans biais ni valid) avec deux témoins :
    ATOMIQUE=0 retire les k `atomic_add` de la boucle, STORE_BOUCLE=0 sort les
    k `store` de la boucle et les remplace par trois stores vectorisés.

    Ce ne sont PAS des candidats : ATOMIQUE=0 ne compte plus l'usage. Ils
    bornent ce que chacun coûte, avant d'écrire une version au bit."""
    t = tl.program_id(0)
    i = tl.arange(0, BE)
    masque = i < E
    lg = tl.load(logits_ptr + t * E + i, mask=masque, other=float("-inf")).to(tl.float32)
    m = tl.max(lg, 0)
    ex = tl.exp(lg - m)
    probs = tl.where(masque, ex / tl.sum(ex, 0), 0.0)
    sel = tl.where(masque, probs, float("-inf"))
    jj = tl.arange(0, 32)
    mj = jj < K
    pw = tl.zeros((32,), dtype=tl.float32)
    ti = tl.zeros((32,), dtype=tl.int32)
    somme = 0.0
    for j in range(K):
        bv = tl.max(sel, 0)
        bi = tl.min(tl.where(sel == bv, i, BE), 0)
        pj = tl.sum(tl.where(i == bi, probs, 0.0), 0)
        somme += pj
        pw = tl.where(jj == j, pj, pw)
        ti = tl.where(jj == j, bi.to(tl.int32), ti)
        if STORE_BOUCLE:
            tl.store(topi_ptr + t * K + j, bi.to(tl.int32))
            tl.store(eid_ptr + t * K + j, bi.to(tl.int32))
        if ATOMIQUE:
            tl.atomic_add(usage_ptr + bi, 1)
        sel = tl.where(i == bi, float("-inf"), sel)
    if not STORE_BOUCLE:
        tl.store(topi_ptr + t * K + jj, ti, mask=mj)
        tl.store(eid_ptr + t * K + jj, ti, mask=mj)
    tl.store(topw_ptr + t * K + jj, pw * ((1.0 / somme) * scale), mask=mj)


def _chrono(fn, rep: int, intra: int = INTRA) -> tuple[float, float]:
    """Médiane et p90 en µs PAR LANCEMENT, sous graphe CUDA.

    FAUTE CORRIGÉE (23/09, première passe de la pièce 68) : avec un seul
    lancement dans le graphe, `Event … g.replay() … Event` mesurait la latence
    de rejeu — le bras `vide` rendait 4,77 µs et le bras `probs` sortait PLUS
    haut que le noyau complet, ce qui est impossible. Cette latence n'est pas
    ce que le pas paie : dans le pas, le noyau est un nœud parmi d'autres d'un
    graphe déjà lancé. On met donc `intra` lancements dans le graphe et on
    divise : le plancher de rejeu est amorti d'autant, et ce qui reste est le
    coût d'un nœud — la grandeur que `nsys` rapporte (6,3 µs, pièce 66)."""
    for _ in range(12):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(intra):
            fn()
    for _ in range(8):
        g.replay()
    torch.cuda.synchronize()
    ms = []
    for _ in range(rep):
        d, f = torch.cuda.Event(True), torch.cuda.Event(True)
        d.record(); g.replay(); f.record()
        torch.cuda.synchronize()
        ms.append(d.elapsed_time(f) * 1e3 / intra)
    ms.sort()
    return statistics.median(ms), ms[int(0.9 * len(ms)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--intra", type=int, default=INTRA,
                    help="lancements dans le graphe : amortit le plancher de rejeu")
    ap.add_argument("--json")
    ap.add_argument("-T", type=int, default=1)
    ap.add_argument("-E", type=int, default=128)
    ap.add_argument("-k", type=int, default=8)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("REFUS : carte requise"); return 65
    dev = "cuda"
    T, E, K = a.T, a.E, a.k
    torch.manual_seed(7)
    logits = (torch.randn(T, E, device=dev) * 2.0).to(torch.bfloat16)
    usage = torch.zeros(E, dtype=torch.int64, device=dev)
    sortie = torch.empty(T, E, dtype=torch.float32, device=dev)
    BE = 1
    while BE < E:
        BE *= 2
    BE = max(BE, 32)

    r = {"forme": {"T": T, "E": E, "k": K, "BE": BE}, "rep": a.rep, "intra": a.intra, "bras": {}}
    r["bras"]["vide"] = _chrono(lambda: _vide_kernel[(T,)](sortie, T, num_warps=1), a.rep, a.intra)
    r["bras"]["probs"] = _chrono(lambda: _probs_kernel[(T,)](logits, sortie, E, BE=BE, num_warps=1), a.rep, a.intra)
    topw = torch.empty(T, K, dtype=torch.float32, device=dev)
    topi = torch.empty(T, K, dtype=torch.int32, device=dev)
    eid = torch.empty(T * K, dtype=torch.int32, device=dev)
    for nom, at, sb in (("temoin_ref", 1, 1), ("sans_atomique", 0, 1), ("stores_groupes", 1, 0),
                        ("ni_l_un_ni_l_autre", 0, 0)):
        r["bras"][nom] = _chrono(
            lambda at=at, sb=sb: _select_kernel[(T,)](
                logits, topw, topi, eid, usage, E, 1.0, K=K, BE=BE,
                ATOMIQUE=at, STORE_BOUCLE=sb, num_warps=1), a.rep, a.intra)
    for w in (1, 4, 8):
        nom = "servi" if w == 1 else f"warps{w}"
        r["bras"][nom] = _chrono(
            lambda w=w: RP.route_fusee(logits, None, K, False, True, 1.0, None, usage, num_warps=w), a.rep, a.intra)

    print(f"forme T={T} E={E} k={K} BE={BE} · {a.rep} rejeux × {a.intra} lancements dans le graphe")
    for nom, (med, p90) in r["bras"].items():
        print(f"  {nom:8s} {med:6.2f} µs (p90 {p90:6.2f})")
    ref = r["bras"]["temoin_ref"][0]
    print(f"  → dans la sélection : atomiques {ref - r['bras']['sans_atomique'][0]:+.2f} µs · "
          f"stores en boucle {ref - r['bras']['stores_groupes'][0]:+.2f} · "
          f"les deux {ref - r['bras']['ni_l_un_ni_l_autre'][0]:+.2f}")
    v, p, s = (r["bras"][n][0] for n in ("vide", "probs", "servi"))
    print(f"  → lancement {v:.2f} · softmax {p - v:+.2f} · sélection {s - p:+.2f} µs")
    print(f"  → sélection sur 48 couches : {(s - p) * 48:.0f} µs/pas")
    verdicts = []
    if v >= 2.0:
        verdicts.append("R1 TENUE : le nœud vide coûte déjà ≥ 2 µs — le plancher, pas le calcul")
    if p - v >= 3.0:
        verdicts.append("R2 TENUE : le softmax (≥ 3 µs au-dessus du vide) domine, et il est intouchable au bit")
    gain_warps = s - min(r["bras"]["warps4"][0], r["bras"]["warps8"][0])
    if gain_warps <= 1.0:
        verdicts.append(f"R3 TENUE : même hors du bit, les warps ne rendent que {gain_warps:.2f} µs")
    if not 6.3 * 0.8 <= s <= 6.3 * 1.2:
        verdicts.append(f"A1 : servi {s:.2f} µs hors de 6,3 ± 20 % — objet mesuré douteux")
    for x in verdicts or ["aucune réfutation ne porte : la sélection est le gisement"]:
        print(f"  → {x}")
    r["verdicts"] = verdicts
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
