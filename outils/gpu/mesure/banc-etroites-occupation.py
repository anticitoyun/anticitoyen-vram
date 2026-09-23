"""Pièce 35 : les projections étroites int8 à b=12 sont-elles bornées par
l OCCUPATION ? (Océane, 22/09 — verdict croisé Q(3), après le split-K réfuté)

En service : qkv [5120, 2048] 10,3 µs (1,02 To/s), o [2048, 4096] 11,8 µs
(0,71 To/s) contre 1,55 possible. Le split-K (partition de K) est réfuté au
banc (Manon b08a3d34) ; reste la forme du programme : un bloc de
`num_warps` warps, `num_stages` étages de pipeline, BM 16 × BN 64. Peu de
warps et peu d étages = peu de charges mémoire en vol par SM = bande
inatteignable, même avec assez de blocs.

Deux parties :
1. `--ptxas` : compile le noyau (Triton) pour chaque forme et lit le rapport
   `ptxas -v` du binaire produit : registres par fil, mémoire partagée,
   déversements (spills) ; occupation = min(blocs par SM permis par les
   registres, par la mémoire partagée, 32) — REGLES § 3 : lire la décision du
   compilateur AVANT de prendre la carte.
2. banc (carte) : balayage BLOCK_M ∈ {16, 32} — ici la tuile M est fixée à 16
   par le contrat `b ≤ 16` (BM 32 n est mesuré que s il est admis par le
   noyau) — × warps ∈ {2, 4, 8} × étages ∈ {2, 3} sur les deux formes, sous
   graphe : µs, To/s, et **égalité AU BIT avec le défaut** (la forme ne change
   ni l ordre des sommes en K ni celui des tranches : toute différence est un
   défaut, pas un arrondi).

Prédiction RÉVISÉE (22/09, lecture réelle de Manon : qkv à 4 warps / 2 étages
= **139 registres par fil**, 8 Kio partagés → 3 blocs/SM, **12 warps actifs
sur 64** : borné par les REGISTRES, pas par la mémoire partagée ni par le
nombre de blocs). Conséquence : **plus de warps par bloc n aide pas** (8
warps → 1 bloc/SM, 8 warps actifs : pire) ; ce qui aide, c est **moins de
warps par bloc** (2 warps → 7 blocs/SM, 14 warps actifs) et **moins
d étages** (chaque étage garde un tampon de plus en registres : 2 étages
au lieu de 3 rend 20-40 registres). Prédit : **(2, 2) et (2, 3) gagnent
≥ 10 % sur `o` (11,8 → ≤ 10,6 µs) et ≥ 5 % sur qkv** ; (8, ·) perd. Réfuté :
aucune forme ne gagne ≥ 5 % sur les deux (l occupation n est pas la cause :
reste la latence de la réduction finale ou le format) ; ou une forme
gagnante diffère du défaut d un seul bit (ce n est plus le même calcul, la
pièce change de nature) ; alarme : des déversements (`spills` > 0) sur une
forme gagnante — le gain viendrait d un registre spillé, pas de l occupation.

Usage : outils/carte.sh python outils/gpu/mesure/banc-etroites-occupation.py [--rep 200] [--json S]
        python outils/gpu/mesure/banc-etroites-occupation.py --ptxas       (à sec, sans carte)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
import torch  # noqa: E402

FORMES_POIDS = {"qkv": (5120, 2048), "o": (2048, 4096)}
B, G = 12, 128
PLANCHER = 1.55e12
BALAYAGE = [(w, e) for w in (2, 4, 8) for e in (2, 3)]
REGISTRES_SM, SHARED_SM, BLOCS_MAX = 65536, 228 * 1024, 32       # sm_120 : 64 K registres, 228 Kio partagés par SM


def occupation(registres: int, shared: int, warps: int) -> dict:
    """Blocs résidents par SM permis par les registres et par la mémoire
    partagée (REGLES § 3 : la décision du compilateur, pas la prédiction)."""
    fils = warps * 32
    par_reg = REGISTRES_SM // max(1, registres * fils) if registres else BLOCS_MAX
    par_shared = SHARED_SM // max(1, shared) if shared else BLOCS_MAX
    blocs = max(0, min(par_reg, par_shared, BLOCS_MAX))
    return {"blocs_par_sm": blocs, "warps_actifs": blocs * warps, "par_registres": par_reg, "par_shared": par_shared}


def lire_ptxas(texte: str) -> dict:
    """Registres/fil, mémoire partagée, déversements depuis un rapport `ptxas -v`."""
    reg = re.search(r"Used (\d+) registers", texte)
    shared = re.search(r"(\d+) bytes smem", texte) or re.search(r"smem=(\d+)", texte)
    spill_s = re.search(r"(\d+) bytes spill stores", texte)
    spill_l = re.search(r"(\d+) bytes spill loads", texte)
    return {"registres": int(reg.group(1)) if reg else 0, "shared": int(shared.group(1)) if shared else 0,
            "spill_stores": int(spill_s.group(1)) if spill_s else 0, "spill_loads": int(spill_l.group(1)) if spill_l else 0}


def poids_int8(n: int, k: int, graine: int, device):
    from acvram.quant.formats import INT8Tensor, quantize
    g = torch.Generator().manual_seed(graine)
    w = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16).to(device)
    t = quantize(w, "int8", group_size=G)
    if not isinstance(t, INT8Tensor):
        raise TypeError(f"quantize(int8) a rendu {type(t).__name__}")
    return t.to(device)


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


def infos_noyau(kernel, device) -> dict:
    """Registres/fil, déversements et mémoire partagée du DERNIER binaire
    compilé pour cet appareil. Triton 3.8 : `JITFunction.device_caches[dev]`
    = (cache signature → CompiledKernel, …) — `kernel.cache` n existe plus
    (22/09, Manon 82b070fc : un `getattr(…, {})` avalait l erreur et rendait
    0 registre, donc 32 blocs/SM partout, faux). Aucun repli muet : si la
    lecture échoue, l appelant reçoit une erreur nommée."""
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        raise RuntimeError("Triton sans `device_caches` : version inattendue, lecture des registres impossible")
    cle = device.index if hasattr(device, "index") and device.index is not None else 0
    entree = caches.get(cle) or next(iter(caches.values()), None)
    binaires = list((entree[0] if isinstance(entree, (tuple, list)) else entree or {}).values())
    if not binaires:
        raise RuntimeError("aucun binaire Triton compilé en cache : le noyau n a pas tourné sur cet appareil")
    c = binaires[-1]
    regs = int(getattr(c, "n_regs", 0) or 0)
    if regs <= 0:
        raise RuntimeError(f"registres lus à {regs} sur {type(c).__name__} : lecture fausse, ne pas publier d occupation")
    return {"registres": regs, "spills": int(getattr(c, "n_spills", 0) or 0),
            "shared": int(getattr(getattr(c, "metadata", None), "shared", 0) or 0)}


def ptxas_des_formes(formes) -> dict:
    """Compile le noyau pour chaque forme et lit `ptxas -v` (via TRITON_PRINT_AUTOTUNING
    non nécessaire : Triton expose `n_regs`, `n_spills`, `shared` sur le kernel compilé)."""
    from acvram.kernels import gemm_etroit as GE
    if not GE.disponible():
        return {"erreur": "Triton indisponible"}
    out = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for nom, (n, k) in FORMES_POIDS.items():
        t = poids_int8(n, k, 1, dev)
        x = torch.randn(B, k, dtype=torch.bfloat16, device=dev)
        for (w, e) in formes:
            GE.regler_forme((w, e))
            try:
                GE.gemm_etroit(x, t, compact=True)
            except Exception as exc:                                  # à sec : Triton refuse sans carte
                out[f"{nom}/{w}w{e}s"] = {"erreur": f"{type(exc).__name__}: {exc}"[:120]}
                continue
            infos = infos_noyau(GE._etroit_reduit_kernel, dev)
            out[f"{nom}/{w}w{e}s"] = {**infos, **occupation(infos["registres"], infos["shared"], w)}
    GE.regler_forme(None)
    return out


def mesurer(rep: int) -> dict:
    from acvram.kernels import gemm_etroit as GE
    dev = torch.device("cuda", torch.cuda.current_device())
    r = {"balayage": [f"{w}w{e}s" for w, e in BALAYAGE], "rep": rep, "b": B, "formes": {}}
    for nom, (n, k) in FORMES_POIDS.items():
        t = poids_int8(n, k, graine=hash(nom) % 1000, device=dev)
        x = (torch.randn(B, k, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).to(dev)
        octets = n * k + t.scales.numel() * 2 + t.zeros.numel()
        GE.regler_forme(None)
        ref = GE.gemm_etroit(x, t, compact=True).clone()
        lignes = {}
        for (w, e) in BALAYAGE:
            GE.regler_forme((w, e))
            try:
                y = GE.gemm_etroit(x, t, compact=True)
                ts = chrono(lambda: GE.gemm_etroit(x, t, compact=True), rep)
            except Exception as exc:
                lignes[f"{w}w{e}s"] = {"erreur": f"{type(exc).__name__}: {exc}"[:120]}
                continue
            med = ts[len(ts) // 2]
            lignes[f"{w}w{e}s"] = {"us": round(med, 2), "us_p90": round(ts[int(0.9 * (len(ts) - 1))], 2),
                                   "to_s": round(octets / (med * 1e-6) / 1e12, 3),
                                   "part_plancher": round(octets / (med * 1e-6) / PLANCHER, 3),
                                   "au_bit": bool(torch.equal(y, ref)),
                                   "ecart_max": float((y.float() - ref.float()).abs().max())}
        GE.regler_forme(None)
        r["formes"][nom] = {"forme": [n, k], "octets": octets, "defaut": "4w3s", "lignes": lignes}
    r.update(verdict(r))
    return r


def verdict(r: dict) -> dict:
    gains, pas_au_bit = {}, []
    for nom, f in r["formes"].items():
        base = f["lignes"].get("4w3s", {}).get("us")
        if base is None:
            continue
        for k, v in f["lignes"].items():
            if "us" not in v or k == "4w3s":
                continue
            if not v["au_bit"]:
                pas_au_bit.append(f"{nom}/{k} (écart {v['ecart_max']:.3g})")
            elif base - v["us"] > gains.get(nom, {"g": 0.0})["g"]:
                gains[nom] = {"forme": k, "g": base - v["us"], "pct": (base - v["us"]) / base, "to_s": v["to_s"]}
    if pas_au_bit:
        v = "INVALIDE : formes qui changent la sortie — " + ", ".join(pas_au_bit[:3]) + " (le contrat « au bit » est faux, défaut à nommer)"
    elif not gains or max(g["pct"] for g in gains.values()) < 0.05:
        v = "RÉFUTÉ : aucune forme ne gagne ≥ 5 % — l occupation n est pas la cause (reste la réduction finale ou le format)"
    else:
        total = sum(g["g"] for g in gains.values()) * 48 / 1e3
        v = ("TENU : " + " ; ".join(f"{n} {g['forme']} −{g['g']:.2f} µs ({g['pct']:.0%}, {g['to_s']} To/s)" for n, g in gains.items())
             + f" → −{total:.2f} ms/pas, au bit")
    return {"gains": gains, "verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ptxas", action="store_true", help="registres/shared/spills et occupation, sans chronométrer")
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.ptxas:
        r = {"ptxas": ptxas_des_formes(BALAYAGE)}
        for k, v in r["ptxas"].items():
            print(f"  {k:12s} {v}")
    else:
        r = mesurer(a.rep)
        for nom, f in r["formes"].items():
            print(f"[occupation] {nom} {f['forme']} · {f['octets'] / 1e6:.1f} Mo · défaut {f['defaut']}")
            for k, v in f["lignes"].items():
                if "us" in v:
                    print(f"    {k:6s} {v['us']:6.2f} µs (p90 {v['us_p90']:.2f})  {v['to_s']:.3f} To/s  {v['part_plancher']:.0%}  au bit {v['au_bit']}")
                else:
                    print(f"    {k:6s} {v['erreur']}")
        print(f"  verdict : {r['verdict']}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
