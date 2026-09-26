#!/usr/bin/env python3
"""Barriere de qualite de PA_WARPS : chaque reglage contre l'ETALON, jamais entre eux.

Passer de 4 a 16 warps change le nombre de participants a la reduction, donc
l'ORDRE DES SOMMES : chaque warp voit un sous-ensemble different de positions
(t += PA_WARPS), puis la reduction combine PA_WARPS accumulateurs avec
correction de maximum. Un gain de debit obtenu au prix de la justesse n'est pas
un gain.

L'etalon est `decode_attention_fixed` — dequantifier le cache en fp32 puis
SDPA, chemin independant du noyau et de son decoupage. DEUX DECOUPAGES COMPARES
ENTRE EUX NE PEUVENT PAS DIRE LEQUEL EST JUSTE : chaque reglage est donc
compare a l'etalon, et ce sont les ECARTS A L'ETALON qu'on met en regard.

DECOMPOSITION, et c'est ce qui rend le critere lisible. L'ecart a l'etalon a
deux sources :

    a) la quantification int8 du cache K/V — COMMUNE a tous les reglages
    b) l'ordre des sommes — la SEULE chose que PA_WARPS change

Le terme (a) domine et ne depend pas du reglage. Donc si les ecarts des trois
reglages tiennent dans un rapport faible, (b) est sous (a) et le reglage
n'affecte pas la qualite.

CRITERE ECRIT AVANT LA MESURE :

    ecart max de chaque reglage < 5e-3        (le seuil deja retenu par
                                              tests/test_graphs.py)
    rapport max/min des trois ecarts < 2     (l'ordre des sommes reste sous la
                                              quantification)
    aucun NaN, aucun inf                     (la correction de maximum survit
                                              a un warp sans position)

Un rapport superieur a 2 ne condamne pas le reglage : il dit que l'ordre des
sommes est devenu visible et qu'il faut l'examiner a plusieurs contextes avant
d'adopter la valeur.

DEUX CONTEXTES AU MOINS, parce que le nombre de positions PAR WARP en depend :
a 39 positions et 16 warps, un warp en voit 2 ou 3 ; a 1024 positions, il en
voit 64. L'accumulation d'erreur n'est pas la meme, et un reglage juste a
contexte court peut ne pas l'etre a contexte long.
"""
import argparse
import json
import math
import os
import subprocess
import sys

SEUIL_ABSOLU = 5e-3
RAPPORT_MAX = 2.0


def mesurer(modele: str, ctx: int) -> dict:
    import torch
    from acvram import kernels
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.layers import decode_attention_fixed

    charge = load_model(modele, dtype=torch.bfloat16, device_override="cuda:0")
    e = Engine(charge, None, max_batch_size=2, max_model_len=ctx + 64,
               enable_cuda_graphs=False)      # la capture figerait la grille
    e.add_request(list(range(1, ctx)), SamplingParams(temperature=0.0,
                                                      max_tokens=4))
    e.step()
    cache = charge.model.caches[0]
    if cache.k_scale is None:
        raise RuntimeError("cache non quantifie : la barriere n'a pas d'objet")
    dec = [s for s in e.running if not s.finished]
    for s in dec:
        e._grow(s)
    batch = e._build_batch(dec, prefill=False)
    tables, lens = batch.fixed_decode_views(torch.device("cuda:0"))
    hq = charge.spec.num_attention_heads
    hd = charge.spec.head_dim
    n_rep = hq // charge.spec.num_key_value_heads
    torch.manual_seed(0)                      # meme q pour tous les reglages
    q = torch.randn(len(dec), hq, hd, device="cuda:0")

    pagine = kernels.paged_attention(q, cache, tables, lens, n_rep, hd ** -0.5)
    if pagine is None:
        raise RuntimeError("noyau pagine indisponible")
    kk, vv = cache.gather_fixed(tables, torch.float32)
    etalon = decode_attention_fixed(q, kk, vv, lens, n_rep, hd ** -0.5)
    d = (pagine - etalon).abs()
    puissance = etalon.float().pow(2).mean().item()
    bruit = d.float().pow(2).mean().item()
    return {
        "ctx": ctx, "positions": int(lens.max().item()),
        "pa_warps": os.environ.get("ACVRAM_PA_WARPS", "defaut"),
        "ecart_max": d.max().item(),
        "ecart_moyen": d.mean().item(),
        "snr_db": (10 * math.log10(puissance / bruit) if bruit > 0
                   else float("inf")),
        "nan": bool(torch.isnan(pagine).any().item()),
        "inf": bool(torch.isinf(pagine).any().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--warps", default="4,8,16")
    ap.add_argument("--ctx", default="40,1024")
    ap.add_argument("--un-point", type=int)      # sous-processus
    ap.add_argument("--sortie", default="/tmp/barriere-pa-warps.json")
    a = ap.parse_args()

    if a.un_point:
        print(json.dumps(mesurer(a.modele, a.un_point)))
        return 0

    releves = []
    for w in a.warps.split(","):
        for ctx in [int(c) for c in a.ctx.split(",")]:
            env = dict(os.environ, ACVRAM_PA_WARPS=w.strip())
            r = subprocess.run(
                [sys.executable, __file__, "--modele", a.modele,
                 "--un-point", str(ctx)], env=env, capture_output=True,
                text=True, timeout=1800)
            if r.returncode != 0:
                print(f"ECHEC a PA_WARPS={w} ctx={ctx}:\n{r.stderr[-1500:]}",
                      file=sys.stderr)
                return 1
            d = json.loads(r.stdout.strip().splitlines()[-1])
            releves.append(d)
            print(f"  PA_WARPS={d['pa_warps']:>3} ctx={ctx:>5} "
                  f"({d['positions']} positions) : ecart max {d['ecart_max']:.3e}"
                  f"  moyen {d['ecart_moyen']:.3e}  SNR {d['snr_db']:6.2f} dB"
                  f"{'  NaN!' if d['nan'] else ''}"
                  f"{'  inf!' if d['inf'] else ''}", flush=True)

    print()
    verdict = 0
    for ctx in [int(c) for c in a.ctx.split(",")]:
        pts = [d for d in releves if d["ctx"] == ctx]
        pires = [d["ecart_max"] for d in pts]
        rapport = max(pires) / min(pires) if min(pires) > 0 else float("inf")
        depasse = [d["pa_warps"] for d in pts if d["ecart_max"] >= SEUIL_ABSOLU]
        casse = [d["pa_warps"] for d in pts if d["nan"] or d["inf"]]
        print(f"ctx {ctx} : ecarts {min(pires):.3e} a {max(pires):.3e}, "
              f"rapport {rapport:.2f}")
        if casse:
            print(f"  REFUSE : NaN ou inf a PA_WARPS={casse}")
            verdict = 2
        elif depasse:
            print(f"  REFUSE : ecart au-dela de {SEUIL_ABSOLU:.0e} a "
                  f"PA_WARPS={depasse}")
            verdict = 2
        elif rapport >= RAPPORT_MAX:
            print(f"  A EXAMINER : rapport {rapport:.2f} >= {RAPPORT_MAX} — "
                  "l'ordre des sommes est devenu visible au-dessus de la "
                  "quantification du cache. Ne pas adopter sans un troisieme "
                  "contexte.")
            verdict = max(verdict, 1)
        else:
            print(f"  ACCEPTE : les trois reglages sont a la meme distance de "
                  f"l'etalon (rapport {rapport:.2f} < {RAPPORT_MAX}), donc "
                  "l'ordre des sommes reste sous la quantification du cache.")
    json.dump(releves, open(a.sortie, "w"), indent=1)
    print(f"\nTERMINE — releve dans {a.sortie}")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
