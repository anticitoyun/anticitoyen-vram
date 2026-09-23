#!/usr/bin/env python3
"""Pièce 92 : découpage de l'attention paginée de décodage (`_partiel_reduit_kernel`, compact) au banc, L2 FROID.

Un pas servi lit 48 caches distincts, un par couche : le KV d'une couche (b = 12, ctx 768 : 9,4 Mo) ne reste pas en
L2 entre deux pas, les poids l'en chassent. Le banc de la 17/09 (`outils/banc-attn-paginee-17-09.py`) relisait UN
cache, donc depuis L2 : il ne représente pas le service. Ici, 48 caches, la table au godet nblk du service
(`bucket_blocks`, graphs.py:642), 48 appels rejoués en UN graphe ; µs par couche = graphe / 48.

Variantes, sans toucher au noyau : C (substitution de `_tranches`), BN (`PAGES_PAR_TUILE`), warps (`WARPS_COMPACT`).
Justesse de chaque variante : écart max à la sortie servie (en 2⁻⁸ relatifs à la ligne) et distance à une
référence fp64 lue du même cache, rapportée à celle du servi (critère du 17/09 : ≤ 1,1 × sur 100 % des lignes).

    outils/carte.sh python outils/gpu/mesure/banc-attn-decoupe.py SORTIE.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
from acvram.kernels import attn_paginee as ap                            # noqa: E402
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache, bucket_blocks   # noqa: E402

if not os.path.realpath(ap.__file__).startswith(os.path.realpath(_REPO) + os.sep):
    raise SystemExit(f"acvram importé de {ap.__file__}, pas de l'arbre {_REPO}")

COUCHES, HQ, HKV, D = 48, 32, 4, 128                  # Coder-30B-A3B
CTX_B12 = [320, 448, 576, 704, 832, 960, 1088, 1216]  # milieux des tranches de la p91
GARDE = [(1, 300), (1, 2048), (4, 768)]
REPET = 30
_TRANCHES_SERVI = ap._tranches


def tranches_c(facteur: float = 1.0, une_tuile: bool = False):
    """`_tranches` du servi, C multiplié par `facteur` (plafond TRANCHES_MAX), ou une tuile par programme."""
    def f(n_pages, b, hkv, device):
        if une_tuile:
            ppt = ap.PAGES_PAR_TUILE
            c = min(ap.TRANCHES_MAX, -(-n_pages // ppt))
            pages = -(-n_pages // c)
            pages = -(-pages // ppt) * ppt
            return -(-n_pages // pages), pages * ap.PAGE
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        voulu = max(1, min(ap.TRANCHES_MAX, int(facteur * -(-2 * sms // max(1, b * hkv)))))
        ppt = max(ap.PAGES_PAR_TUILE, -(-n_pages // voulu))
        ppt = -(-ppt // ap.PAGES_PAR_TUILE) * ap.PAGES_PAR_TUILE
        return -(-n_pages // ppt), ppt * ap.PAGE
    return f


VARIANTES = {                                         # nom → (tranches, PAGES_PAR_TUILE, WARPS_COMPACT, réduction déroulée)
    "servi": (None, None, None, False),
    "C×2": (tranches_c(2.0), None, None, False),
    "w4": (None, None, 4, False),
    "dér": (None, None, None, True),
    "dér·w4": (None, None, 4, True),
    "dér·C×2": (tranches_c(2.0), None, None, True),
    "dér·C×2·w4": (tranches_c(2.0), None, 4, True),
    "dér·1tuile": (tranches_c(une_tuile=True), None, None, True),
    "dér·1tuile·w4": (tranches_c(une_tuile=True), None, 4, True),
}
# Première prise (f7bccbf0, prise-banc.log) : C×4, BN32 et leurs croisements, tous plus lents que le servi ; retirés.


def poser(v):
    tr, ppt, w, der = VARIANTES[v]
    ap._tranches = tr or _TRANCHES_SERVI
    ap.PAGES_PAR_TUILE = ppt or 4
    ap.WARPS_COMPACT = w or int(os.environ.get("ACVRAM_ATTN_WARPS_COMPACT", "8"))
    ap.REDUC_DEROULEE = der


def montage(b, ctx, g):
    nblk = -(-ctx // 16)
    n = bucket_blocks(nblk)
    caches, tables = [], []
    for _ in range(COUCHES):
        c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=b * nblk + 1,
                                       dtype="int8", device="cuda"))
        c.k.random_(-127, 128, generator=g); c.v.random_(-127, 128, generator=g)
        c.k_scale.uniform_(0.01, 0.05, generator=g); c.v_scale.uniform_(0.01, 0.05, generator=g)
        t = torch.zeros(b, n, dtype=torch.long, device="cuda")
        t[:, :nblk] = torch.randperm(b * nblk, device="cuda", generator=g).view(b, nblk) + 1   # blocs dispersés
        caches.append(c); tables.append(t)
    lens = torch.full((b,), ctx, dtype=torch.long, device="cuda")
    q = [torch.randn(b, HQ, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(COUCHES)]
    return caches, tables, lens, q


def pas(caches, tables, lens, q, scale):
    return [ap.paged_attention(q[i], c.k, c.k_scale, c.v, c.v_scale, tables[i], lens, HKV, scale, 0, compact=True)
            for i, c in enumerate(caches)]


def chrono_graphe(f):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPET):
        d, a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); g.replay(); a.record(); torch.cuda.synchronize()
        ts.append(d.elapsed_time(a))
    return statistics.median(ts) * 1000 / COUCHES      # µs par couche


def reference64(c, table, lens, q, scale):
    B = q.shape[0]
    n_rep = HQ // HKV
    ref = torch.zeros(B, HQ, D, dtype=torch.float64, device=q.device)
    for b in range(B):
        k, v = c.gather(table[b], int(lens[b]), torch.float64)
        for h in range(HQ):
            p = torch.softmax((k[:, h // n_rep] @ q[b, h].double()) * scale, 0)
            ref[b, h] = p @ v[:, h // n_rep]
    return ref


def main() -> int:
    sortie = sys.argv[1]
    g = torch.Generator(device="cuda").manual_seed(92)
    scale = 1 / math.sqrt(D)
    cellules = [(12, c) for c in CTX_B12] + GARDE
    res = {"sms": torch.cuda.get_device_properties(0).multi_processor_count, "cellules": []}
    print(f"{'b':>3s} {'ctx':>5s} " + " ".join(f"{v:>12s}" for v in VARIANTES) + "   (µs/couche, L2 froid, graphe)", flush=True)
    for b, ctx in cellules:
        caches, tables, lens, q = montage(b, ctx, g)
        ligne = {"b": b, "ctx": ctx, "variantes": {}}
        poser("servi")
        y0 = pas(caches, tables, lens, q, scale)[0].double()
        ref = reference64(caches[0], tables[0], lens, q[0], scale)
        d0 = ((y0 - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)).reshape(-1)
        for v in VARIANTES:
            poser(v)
            C, chunk = ap._tranches(tables[0].shape[1], b, HKV, q[0].device)
            y = pas(caches, tables, lens, q, scale)[0].double()
            ecart = ((y - y0).abs().amax(-1) / y0.abs().amax(-1).clamp(min=1e-6) / 2 ** -8).max().item()
            dv = ((y - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)).reshape(-1)
            us = chrono_graphe(lambda: pas(caches, tables, lens, q, scale))
            ligne["variantes"][v] = {"us": round(us, 2), "C": C, "chunk": chunk, "ecart_2m8": round(ecart, 3),
                                     "au_bit": bool(torch.equal(y, y0)),
                                     "dist_rapport_max": round(float((dv / d0.clamp(min=1e-12)).max()), 3),
                                     "part_le_1_1": round(float((dv <= 1.1 * d0).float().mean()), 4)}
        poser("servi")
        res["cellules"].append(ligne)
        print(f"{b:3d} {ctx:5d} " + " ".join(f"{ligne['variantes'][v]['us']:12.2f}" for v in VARIANTES)
              + "   au bit : " + ",".join(v for v in VARIANTES if ligne["variantes"][v]["au_bit"]), flush=True)
        del caches, tables, q
        torch.cuda.empty_cache()
    moy = {v: statistics.mean(l["variantes"][v]["us"] for l in res["cellules"] if l["b"] == 12) for v in VARIANTES}
    res["moyenne_lot_b12_us"] = {v: round(x, 2) for v, x in moy.items()}
    print("moyenne du lot b=12 (µs/couche) : " + " · ".join(f"{v} {x:.2f}" for v, x in moy.items()), flush=True)
    print("ms/pas (×48) : " + " · ".join(f"{v} {x * 48 / 1000:.3f}" for v, x in moy.items()), flush=True)
    json.dump(res, open(sortie, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
