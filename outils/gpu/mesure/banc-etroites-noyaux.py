"""Pièce 43 : à octets égaux, quel NOYAU sert le mieux les projections
d attention à M = 12 ? (poste1, 22/09 — TRT-LLM fait 1,81 To/s sur les mêmes
18,8 Mo/couche, nous 0,71-0,81 : l écart est le noyau, pas le format.)

Trois bras sur les formes de Coder (qkv [5120, 2048], o [2048, 4096]), même
poids de référence bf16, M = 12 :
  a) `gemm_etroit` Triton int8 par groupe de 128 — LE CHEMIN SERVI, référence
     numérique de tous les autres ;
  b) `torch._int_mm` (cuBLASLt int8×int8 → int32) + épilogue d échelles, poids
     int8 par CANAL — cuBLASLt exige M > 16 : le bras rembourre M 12 → 32 et
     le dit (les poids lus ne changent pas, seules 20 lignes d activation en
     plus, 40-80 Kio) ; déjà mesuré à M→32 le 21/09 (1,13 / 0,67 To/s,
     `verdict-m2-b12`), rejoué ici avec les octets et l écart ;
  c) `torch._scaled_mm` FP8 e4m3 (le chemin de TRT-LLM) : poids FP8 par
     canal, activation FP8 par jeton, mêmes octets qu int8 (1 o/poids) —
     **pas au bit** (autre format), étiqueté comme tel.
Sortie par bras : µs (médiane, p90) sous graphe, Go lus, To/s, part du
plancher (qkv 10,73 Mo → 6,8 µs, o 8,59 → 5,5 µs à 1,55 To/s), écart relatif
max contre (a) et rang de corrélation des sorties.

Prédit (écrit avant) : (b) 1,0-1,3 To/s sur qkv (M rembourré : la carte
travaille mieux) et 0,6-0,9 sur o (K = 4096 long, N court) — donc **mieux
que (a) sur qkv, pas sur o** ; (c) 1,3-1,8 To/s sur les deux (le chemin
trtllm), écart relatif 1-3 % contre (a). Réfuté si (b) et (c) ≤ 1,0 To/s
partout (le noyau n est pas le levier : il faudrait changer la forme du
problème) ; alarme si (c) > 2,0 To/s (au-delà de la bande utile mesurée :
relire les octets comptés).

Usage : outils/carte.sh python outils/gpu/mesure/banc-etroites-noyaux.py [--rep 200] [--json S]
        (≤ 5 min)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
import torch  # noqa: E402

FORMES = {"qkv": (5120, 2048), "o": (2048, 4096)}
M, G = 12, 128
PLANCHER = 1.55e12
M_CUBLASLT = 32                      # cuBLASLt (`torch._int_mm`) refuse M ≤ 16


def poids_ref(n: int, k: int, graine: int, dev):
    g = torch.Generator().manual_seed(graine)
    return (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16).to(dev)


def octets_int8(n: int, k: int) -> int:
    return n * k + (n * k // G) * 2 + (n * k // G)          # qweight + scales fp16 + zeros u8


def octets_canal(n: int, k: int) -> int:
    return n * k + n * 4                                     # poids + une échelle par ligne


def chrono(fn, rep: int) -> list[float]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)


def resume(ts: list[float], octets: int, ref=None, y=None) -> dict:
    med = ts[len(ts) // 2]
    d = {"us": round(med, 2), "us_p90": round(ts[int(0.9 * (len(ts) - 1))], 2),
         "mo": round(octets / 1e6, 2), "to_s": round(octets / (med * 1e-6) / 1e12, 3),
         "part_plancher": round(octets / (med * 1e-6) / PLANCHER, 3)}
    if ref is not None and y is not None:
        e = (y.float() - ref.float()).abs()
        d["ecart_rel_max"] = round(float(e.max() / ref.float().abs().max().clamp(min=1e-6)), 5)
        d["au_bit"] = bool(torch.equal(y, ref))
    return d


def bras_triton(w: torch.Tensor, x: torch.Tensor):
    from acvram.kernels import gemm_etroit as GE
    from acvram.quant.formats import quantize
    t = quantize(w, "int8", group_size=G).to(w.device)
    return (lambda: GE.gemm_etroit(x, t, compact=True)), octets_int8(*w.shape)


def bras_int_mm(w: torch.Tensor, x: torch.Tensor):
    """int8 par canal + `torch._int_mm`, M rembourré à 32 (cuBLASLt)."""
    n, k = w.shape
    s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (w.float() / s).round().clamp(-127, 127).to(torch.int8)
    sx = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    a = (x.float() / sx).round().clamp(-127, 127).to(torch.int8)
    a32 = torch.zeros(M_CUBLASLT, k, dtype=torch.int8, device=x.device)
    a32[:M] = a

    def f():
        acc = torch._int_mm(a32, q.t())
        return (acc[:M].float() * sx * s.t()).to(x.dtype)
    return f, octets_canal(n, k)


def bras_fp8(w: torch.Tensor, x: torch.Tensor):
    """FP8 e4m3 par canal × par jeton, `torch._scaled_mm` (chemin TRT-LLM)."""
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return None, 0
    n, k = w.shape
    smax = 448.0
    sw = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / smax
    qw = (w.float() / sw).clamp(-smax, smax).to(torch.float8_e4m3fn)
    sx = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / smax
    qx = (x.float() / sx).clamp(-smax, smax).to(torch.float8_e4m3fn)
    # `torch._scaled_mm(A [M, K], B [K, N])` : A rangée-majeure contiguë, B
    # COLONNE-majeure (stride(0) == 1). `qw` est [N, K] rangée-majeure, donc
    # `qw.t()` est déjà [K, N] colonne-majeure — la vue suffit. L ancien
    # `qw.t().contiguous().t()` rendait un [N, K] colonne-majeur dont le `.t()`
    # redevenait rangée-majeur : B mal disposé, plantage de stride avant toute
    # mesure (poste2 d964e2cc). Les échelles sont fp32, [M, 1] et [1, N].
    b = qw.t()
    sa, sb = sx.to(torch.float32).contiguous(), sw.t().to(torch.float32).contiguous()
    _garde_scaled_mm(qx, b, sa, sb)

    def f():
        return torch._scaled_mm(qx, b, scale_a=sa, scale_b=sb, out_dtype=x.dtype)
    return f, octets_canal(n, k)


def _garde_scaled_mm(a, b, sa, sb) -> None:
    """Nomme la disposition attendue AVANT l appel : sur carte, `_scaled_mm`
    rend une erreur de stride qui ne dit pas lequel des quatre tenseurs est en
    cause, et le banc meurt sans mesure. Vérifiable à sec par lecture, pas par
    exécution (aucun noyau FP8 sur processeur)."""
    m, k = a.shape
    k2, n = b.shape
    exigences = [
        (a.dtype == torch.float8_e4m3fn, f"A doit être float8_e4m3fn, pas {a.dtype}"),
        (a.stride(1) == 1, f"A doit être rangée-majeure (stride {a.stride()})"),
        (b.dtype == torch.float8_e4m3fn, f"B doit être float8_e4m3fn, pas {b.dtype}"),
        (k2 == k, f"B doit être [K, N] avec K = {k}, reçu {tuple(b.shape)}"),
        (b.stride(0) == 1, f"B doit être COLONNE-majeure, stride(0) = {b.stride(0)} "
                           f"(prendre `qw.t()` d un poids [N, K] contigu, pas `.contiguous().t()`)"),
        (k % 16 == 0, f"K = {k} doit être multiple de 16"),
        (sa.dtype == torch.float32 and tuple(sa.shape) == (m, 1), f"scale_a doit être fp32 [M, 1], reçu {sa.dtype} {tuple(sa.shape)}"),
        (sb.dtype == torch.float32 and tuple(sb.shape) == (1, n), f"scale_b doit être fp32 [1, N], reçu {sb.dtype} {tuple(sb.shape)}"),
    ]
    manques = [m_ for ok, m_ in exigences if not ok]
    if manques:
        raise RuntimeError("bras FP8 : disposition refusée par _scaled_mm — " + " ; ".join(manques))


def mesurer(rep: int) -> dict:
    dev = torch.device("cuda", torch.cuda.current_device())
    r = {"M": M, "M_cublaslt": M_CUBLASLT, "rep": rep, "plancher_to_s": PLANCHER / 1e12, "formes": {}}
    for nom, (n, k) in FORMES.items():
        w = poids_ref(n, k, hash(nom) % 1000, dev)
        x = (torch.randn(M, k, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).to(dev)
        lignes = {}
        f_a, oct_a = bras_triton(w, x)
        ref = f_a().clone()
        lignes["a_triton_int8_groupe"] = resume(chrono(f_a, rep), oct_a, ref, ref)
        for cle, fabrique in (("b_int_mm_int8_canal", bras_int_mm), ("c_scaled_mm_fp8", bras_fp8)):
            try:
                f, oct_ = fabrique(w, x)
                if f is None:
                    lignes[cle] = {"absent": "chemin indisponible dans cet environnement"}
                    continue
                y = f()
                lignes[cle] = resume(chrono(f, rep), oct_, ref, y)
            except Exception as exc:                       # noqa: BLE001
                lignes[cle] = {"erreur": f"{type(exc).__name__}: {exc}"[:160]}
        r["formes"][nom] = {"forme": [n, k], "lignes": lignes}
    r.update(verdict(r))
    return r


def verdict(r: dict) -> dict:
    gains = {}
    for nom, f in r["formes"].items():
        base = f["lignes"]["a_triton_int8_groupe"]["us"]
        for cle, v in f["lignes"].items():
            if cle.startswith("a_") or "us" not in v:
                continue
            g = base - v["us"]
            if g > gains.get(nom, {"g": 0.0})["g"]:
                gains[nom] = {"bras": cle, "g": round(g, 2), "pct": round(g / base, 3), "to_s": v["to_s"],
                              "ecart_rel_max": v.get("ecart_rel_max")}
    tos = [v["to_s"] for f in r["formes"].values() for c, v in f["lignes"].items() if not c.startswith("a_") and "to_s" in v]
    if tos and max(tos) > 2.0:
        v = f"ALARME : {max(tos)} To/s au-dessus de la bande utile — relire les octets comptés avant de publier"
    elif not tos or max(tos) <= 1.0:
        v = "RÉFUTÉ : aucun autre noyau ne dépasse 1,0 To/s sur ces formes — le noyau n est pas le levier"
    elif gains:
        total = sum(max(g["g"], 0.0) for g in gains.values()) * 48 / 1e3
        v = ("TENU : " + " ; ".join(f"{n} {g['bras']} −{g['g']} µs ({g['pct']:.0%}, {g['to_s']} To/s, écart {g['ecart_rel_max']})"
                                    for n, g in gains.items()) + f" → −{total:.2f} ms/pas (à qualifier : int8 canal et fp8 ne sont pas au bit)")
    else:
        v = "RÉFUTÉ : les autres noyaux existent mais ne gagnent pas de temps sur ces formes"
    return {"gains": gains, "verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = mesurer(a.rep)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    for nom, f in r["formes"].items():
        print(f"[noyaux] {nom} {f['forme']} · M={M}")
        for cle, v in f["lignes"].items():
            if "us" in v:
                print(f"    {cle:24s} {v['us']:7.2f} µs (p90 {v['us_p90']:.2f})  {v['mo']:6.2f} Mo  {v['to_s']:.3f} To/s  "
                      f"{v['part_plancher']:.0%} plancher  écart {v.get('ecart_rel_max', '—')}  au bit {v.get('au_bit', '—')}")
            else:
                print(f"    {cle:24s} {v.get('absent') or v.get('erreur')}")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
