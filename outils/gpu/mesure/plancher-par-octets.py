#!/usr/bin/env python3
"""Le terme par bloc du plancher est-il l'ecriture des partiels ? Le test par D.

CE QUI EST A TRANCHER. Le plancher du noyau d'attention est modelise par
`9,65 us + 3,76 ns par bloc`, et ce terme par bloc a ete attribue aux 512
octets que chaque bloc ecrit dans part[] — 512 o / 3,76 ns = 136 Go/s.

POURQUOI CETTE ATTRIBUTION N'EST PAS ETABLIE. Le debit de 136 Go/s n'est pas
mesure : il est DEDUIT du temps observe. Or n'importe quel nombre d'octets
donne un debit plausible quand on l'ajuste ainsi — a la borne de la carte
(1050 Go/s) les memes 512 octets couteraient 0,49 ns, sept fois moins, et
l'ecart serait attribue a la dispersion de l'ecriture sans qu'on la mesure. Un
raisonnement qui ne peut pas echouer ne prouve rien.

Et si le plancher a ete mesure sur le noyau temoin qui ecrit TROIS FLOATS par
bloc, l'attribution est refutee par un facteur 43 : 12 o / 3,76 ns = 3,2 Go/s,
qu'aucune carte n'atteint.

LE TEST. Deux voies, et la seconde est meilleure — c'est Manon qui l'a
indiquee, en signalant que banc_fma n'ecrivait qu'un flottant par bloc.

  VOIE A, celle que j'avais ecrite : les cinq instanciations D de
  paged_attn_partial, donc cinq tailles d'ecriture. Defaut REEL, que j'avais
  moi-meme inscrit dans le domaine de validite : D change aussi la memoire
  partagee, l'occupation et le travail par warp — TROIS choses a la fois.

  VOIE B, retenue : banc_fma avec un parametre `flottants`. Le travail par
  bloc, la grille, les threads, l'occupation restent identiques ; SEULS LES
  OCTETS ECRITS BOUGENT. C'est le test propre, et la voie A ne sert plus que
  de controle sur le vrai noyau.

Si le terme par bloc est bien l'ecriture, il doit suivre PROPORTIONNELLEMENT :

    D     octets/bloc    terme attendu
     32       128          0,94 ns
     64       256          1,88
    128       512          3,76   <- le point mesure
    256      1024          7,52
    512      2048         15,04

PREDICTION ECRITE AVANT LA MESURE, et les trois issues :

    le terme suit a moins de 20 %   l'attribution est ETABLIE : le plancher est
                                    l'ecriture des partiels, et le borner borne
                                    le plancher
    le terme reste ~constant        l'attribution est REFUTEE : le cout est par
                                    BLOC et non par OCTET — distribution,
                                    ordonnancement, descripteur — et le plancher
                                    garde sa cause inconnue
    le terme croit mais moins vite  une part est l'ecriture, une part est fixe
    que les octets                  par bloc : les deux se separent par la pente

CE QUE JE N'ATTENDS PAS et qui serait le plus interessant : un terme qui
DECROIT avec D. Il faudrait alors que les gros blocs amortissent quelque chose
que les petits paient, et ce serait un mecanisme neuf.

DOMAINE DE VALIDITE. Le test compare cinq instanciations qui ne diffferent PAS
seulement par les octets ecrits : D change aussi la memoire partagee (3 648 o a
D=128, 11 328 a D=512), donc l'occupation en blocs par SM, et le travail par
warp (PER_LANE = D/32). Un ecart observe n'est donc attribuable a l'ecriture
que si le nombre de blocs RESIDENTS ne change pas dans la plage utilisee — ce
que le banc calcule et publie. C'est la limite de ce test, et elle est ecrite
avec lui.
"""
import argparse
import json
import math
import os
import sys

REGS_SM, WARPS_SM, SMEM_SM, SM = 65536, 48, 100 * 1024, 170
N_MESURES = 51


def blocs_residents(D: int, pa_warps: int = 4, reg: int = 40) -> int:
    thr = 32 * pa_warps
    smem = D * 4 + 3 * pa_warps * 4 + pa_warps * D * 4 + pa_warps * 4
    return min(REGS_SM // (reg * thr), WARPS_SM // pa_warps,
               SMEM_SM // max(1, smem), 32)


def mesurer_b(flottants: int, grilles: list[int]) -> dict:
    """VOIE B : banc_fma, seuls les octets ecrits par bloc varient."""
    import torch
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None:
        raise RuntimeError("extension indisponible")
    if not hasattr(ext, "banc_fma"):
        raise RuntimeError("ce binaire ne porte pas banc_fma")
    pts = []
    for blocs in grilles:
        gy = 32
        gz = max(1, blocs // (gy * 1))
        appel = lambda: ext.banc_fma(1, gy, gz, 128, 1, flottants)
        for _ in range(5):
            appel()
        torch.cuda.synchronize()
        temps = []
        for _ in range(N_MESURES):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); appel(); e1.record()
            torch.cuda.synchronize()
            temps.append(e0.elapsed_time(e1) * 1000.0)
        temps.sort()
        pts.append({"blocs": gy * gz, "us": temps[len(temps) // 2],
                    "p10": temps[int(0.10 * len(temps))],
                    "p90": temps[int(0.90 * len(temps))]})
    return {"voie": "B", "flottants": flottants,
            "octets_par_bloc": flottants * 4, **_ajuster(pts)}


def _ajuster(pts: list) -> dict:
    xs = [p["blocs"] for p in pts]
    ys = [p["us"] for p in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return {"points": pts, "intercept_us": my - b * mx,
            "ns_par_bloc": b * 1000.0}


def mesurer(D: int, grilles: list[int]) -> dict:
    """VOIE A, controle sur le vrai noyau : cinq instanciations D."""
    import torch
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None:
        raise RuntimeError("extension indisponible")
    d = "cuda"
    HKV, NB, N = 8, 1, 1
    pts = []
    for blocs in grilles:
        # blocs = BQ x HQ x C ; on fixe C=1 (une tranche) et HQ=8, on fait
        # varier BQ. Ainsi SEUL le nombre de blocs bouge, pas le travail par
        # bloc ni le nombre de partiels par paire.
        HQ = 8
        BQ = max(1, blocs // HQ)
        q = torch.randn(BQ, HQ, D, dtype=torch.bfloat16, device=d)
        kc = torch.randint(-100, 100, (NB, 16, HKV, D), dtype=torch.int8, device=d)
        vc = torch.randint(-100, 100, (NB, 16, HKV, D), dtype=torch.int8, device=d)
        ks = torch.ones(NB, 16, HKV, dtype=torch.float16, device=d)
        vs = torch.ones(NB, 16, HKV, dtype=torch.float16, device=d)
        tb = torch.zeros(BQ, N, dtype=torch.long, device=d)
        sl = torch.full((BQ,), 4, dtype=torch.long, device=d)
        appel = lambda: ext.paged_attention(q, kc, ks, vc, vs, tb, sl, HKV,
                                            D ** -0.5, 1, 0)
        for _ in range(5):
            appel()
        torch.cuda.synchronize()
        temps = []
        for _ in range(N_MESURES):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); appel(); e1.record()
            torch.cuda.synchronize()
            temps.append(e0.elapsed_time(e1) * 1000.0)
        temps.sort()
        pts.append({"blocs": BQ * HQ, "us": temps[len(temps) // 2],
                    "p10": temps[int(0.10 * len(temps))],
                    "p90": temps[int(0.90 * len(temps))]})
    return {"voie": "A", "D": D, "octets_par_bloc": D * 4,
            "blocs_residents": SM * blocs_residents(D), **_ajuster(pts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="32,64,128,256,512")
    ap.add_argument("--grilles", default="80,160,320,640,1280")
    ap.add_argument("--un-point", type=int)
    ap.add_argument("--voie", choices=("A", "B"), default="B")
    ap.add_argument("--flottants", default="32,64,128,256,512",
                    help="voie B : flottants ecrits par bloc")
    ap.add_argument("--sortie", default="/tmp/plancher-par-octets.json")
    a = ap.parse_args()
    grilles = [int(g) for g in a.grilles.split(",")]

    if a.un_point:
        print(json.dumps(mesurer(a.un_point, grilles) if a.voie == "A"
                         else mesurer_b(a.un_point, grilles)))
        return 0

    import subprocess
    releves = []
    valeurs = ([int(x) for x in a.dims.split(",")] if a.voie == "A"
               else [int(x) for x in a.flottants.split(",")])
    print(f"VOIE {a.voie} — "
          + ("cinq instanciations D du vrai noyau (controle)" if a.voie == "A"
             else "banc_fma, seuls les octets ecrits varient"))
    for D in valeurs:
        r = subprocess.run([sys.executable, __file__, "--un-point", str(D),
                            "--grilles", a.grilles, "--voie", a.voie],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"ECHEC a D={D} :\n{r.stderr[-1200:]}", file=sys.stderr)
            return 1
        d = json.loads(r.stdout.strip().splitlines()[-1])
        releves.append(d)
        print(f"  {'D' if a.voie == 'A' else 'flottants'}={D:>3} "
              f"({d['octets_par_bloc']:>4} o/bloc) : "
              f"plancher {d['intercept_us']:6.2f} us + "
              f"{d['ns_par_bloc']:6.2f} ns/bloc   "
              + (f"   ({d['blocs_residents']} blocs residents)"
                 if 'blocs_residents' in d else ""), flush=True)

    ref = next((d for d in releves if d["octets_par_bloc"] == 512), releves[0])
    print(f"\n{'D':>4} {'o/bloc':>7} {'ns/bloc':>9} {'attendu':>9} {'ecart':>8}"
          f" {'residents':>10}")
    verdict_suit = True
    for d in releves:
        att = ref["ns_par_bloc"] * d["octets_par_bloc"] / ref["octets_par_bloc"]
        ecart = (d["ns_par_bloc"] / att - 1) * 100 if att else float("nan")
        if abs(ecart) > 20:
            verdict_suit = False
        print(f"{d.get('D', d.get('flottants')):>4} {d['octets_par_bloc']:>7} {d['ns_par_bloc']:>9.2f} "
              f"{att:>9.2f} {ecart:>7.1f} % {d['blocs_residents']:>10}")

    # l'occupation a-t-elle change ? sans quoi l'ecart n'est pas attribuable
    res = {d.get("blocs_residents", 0) for d in releves}
    if len(res) > 1:
        print(f"\nRESERVE : les blocs residents varient sur la plage ({sorted(res)}) "
              "— un ecart n'est alors pas attribuable a la seule ecriture, D "
              "changeant aussi la memoire partagee et le travail par warp.")
    plats = [d["ns_par_bloc"] for d in releves]
    if verdict_suit:
        verdict = ("ATTRIBUTION ETABLIE : le terme par bloc suit les octets a "
                   "moins de 20 %. Borner le nombre de blocs borne le plancher.")
    elif max(plats) / max(1e-9, min(plats)) < 1.5:
        verdict = ("ATTRIBUTION REFUTEE : le terme est ~constant quel que soit "
                   "D. Le cout est par BLOC et non par OCTET — le plancher "
                   "garde sa cause inconnue.")
    else:
        verdict = ("MIXTE : le terme croit avec les octets mais moins vite. Une "
                   "part est l'ecriture, une part est fixe par bloc ; la pente "
                   "les separe.")
    print(f"\nVERDICT : {verdict}")
    json.dump({"releves": releves, "verdict": verdict},
              open(a.sortie, "w"), indent=1)
    print(f"TERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
