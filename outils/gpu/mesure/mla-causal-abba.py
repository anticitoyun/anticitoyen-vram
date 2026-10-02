"""zzs (poste5, 01/10) : troncature causale du cœur MLA au préfill — mur du préfill et jetons gloutons, en SERVICE, ABBA.
A = ACVRAM_MLA_CAUSAL=0 (témoin : toutes les clés puis masque), B = défaut ; MÊME arbre, un serveur neuf par bras, lancé et
arrêté par le harnais serveur-bras.sh (mla-causal-abba.sh). Prédiction et critère : revue/poste5-zzs-scelle-01-10.md.

  bras URL NOM SORTIE.json : contre un `acvram serve --no-prefix-cache` prêt. Lit la ligne de régime dans /metrics (preuve
           que l'interrupteur a pris DANS le serveur), puis pour L = 8 192 et 32 768 : une chauffe (invite hors mesure,
           autotune et allocateur), puis REPS invites DISTINCTES (mêmes ids d'un bras à l'autre), chacune deux fois :
           1) flux, max_tokens=1 : mur du préfill = temps jusqu'au premier fragment (texte vide compris, pièce 210) ;
           2) sans flux, max_tokens=64, logprobs=0 : les 64 jetons gloutons (textes par pas) et leurs logprobs.
           Le serveur ne rend pas les ids : deux ids distincts au même texte ne se distinguent que par leur logprob, d'où
           la comparaison des logprobs en plus des textes.
  comparer A1.json B1.json B2.json A2.json : régimes (A porte mla_causal=0(temoin), B non, rien d'autre ne diffère :
           sinon rc 5), témoins A1 = A2 et B1 = B2 en jetons (sinon rc 6, aucun verdict), logprobs finis (sinon rc 8),
           jetons A contre B (premier pas divergent), puis mur du préfill (médiane des 2 × REPS mesures par chemin)
           contre les bandes scellées : 8 k −18 à −26 %, 32 k −35 à −48 %, FAUX si le gain à 32 k est < 20 %.
"""
import json
import math
import os
import statistics
import sys
import time

LONGUEURS = (8192, 32768)
REPS = int(os.environ.get("MLA_ABBA_REPS", "3"))
N_GEN = 64
VOCAB_APPROX = int(os.environ.get("BANC_VOCAB", "150000"))
BANDES = {8192: (-26.0, -18.0, None), 32768: (-48.0, -35.0, -20.0)}   # (bas, haut, FAUX si Δ > ce seuil)
TEMOIN = "mla_causal=0(temoin)"


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


def _cles(L: int) -> list:
    """Invites mesurées pour L : distinctes entre elles, entre longueurs, et de la chauffe (k = base)."""
    base = 100 * LONGUEURS.index(L)
    return [base + 1 + r for r in range(REPS)]


def bras(url: str, nom: str, sortie: str) -> int:
    import httpx
    url = url.rstrip("/")
    c = httpx.Client(timeout=900.0)
    regime = c.get(f"{url}/metrics").json().get("regime_ligne")
    if not regime:
        sys.exit("mla-causal-abba : /metrics sans regime_ligne — l'interrupteur ne peut pas être prouvé")

    def mur_prefill(ids):
        corps = {"model": nom, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "ignore_eos": True, "stream": True}
        t = time.perf_counter()
        with c.stream("POST", f"{url}/v1/completions", json=corps) as r:
            r.raise_for_status()
            for ligne in r.iter_lines():
                if not ligne.startswith("data: ") or ligne == "data: [DONE]":
                    continue
                d = json.loads(ligne[6:])
                if "error" in d:
                    raise RuntimeError(f"erreur serveur : {d['error']}")
                if d.get("choices"):
                    return (time.perf_counter() - t) * 1e3
        raise RuntimeError("flux sans fragment de jeton")

    def gloutons(ids):
        corps = {"model": nom, "prompt": ids, "max_tokens": N_GEN, "temperature": 0.0, "ignore_eos": True,
                 "logprobs": 0}
        r = c.post(f"{url}/v1/completions", json=corps)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        lp = ch.get("logprobs") or {}
        return lp.get("tokens") or [], lp.get("token_logprobs") or []

    res = {"url": url, "regime": regime, "reps": REPS, "n_gen": N_GEN, "mesures": {}}
    for L in LONGUEURS:
        mur_prefill(invite(100 * LONGUEURS.index(L), L))          # chauffe, jetée
        m = []
        for k in _cles(L):
            ids = invite(k, L)
            ms = mur_prefill(ids)
            toks, lps = gloutons(ids)
            m.append({"k": k, "mur_prefill_ms": ms, "jetons": toks, "logprobs": lps})
            print(f"bras L={L} k={k} : préfill {ms:.1f} ms, {len(toks)} jetons", flush=True)
        res["mesures"][str(L)] = m
    with open(sortie, "w") as f:
        json.dump(res, f)
    print(f"bras terminé : {res['regime']}", flush=True)
    return 0


def _divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def _jetons(x):
    return {L: [s["jetons"] for s in m] for L, m in x["mesures"].items()}


def _dlp(a, b):
    v = [abs(p - q) for p, q in zip(a, b) if p is not None and q is not None]
    return max(v) if v else None


def comparer(a1: str, b1: str, b2: str, a2: str) -> int:
    A1, B1, B2, A2 = (json.load(open(p)) for p in (a1, b1, b2, a2))
    for nom, x in (("A1", A1), ("A2", A2)):
        if TEMOIN not in x["regime"].split():
            print(f"RÉGIME : {nom} sans « {TEMOIN} » — l'interrupteur n'a pas pris : {x['regime']}"); return 5
    for nom, x in (("B1", B1), ("B2", B2)):
        if TEMOIN in x["regime"].split():
            print(f"RÉGIME : {nom} porte « {TEMOIN} » — bras inversés : {x['regime']}"); return 5
    # regime.py:737 : A porte aussi la variable hors défaut, et B seul peut porter « défaut » (aucune variable posée)
    neutres = {TEMOIN, "ACVRAM_MLA_CAUSAL=0", "défaut"}
    sans = {" ".join(t for t in x["regime"].split() if t not in neutres) for x in (A1, B1, B2, A2)}
    if len(sans) != 1:
        print("RÉGIME : les bras diffèrent par autre chose que mla_causal :"); [print(f"  {s}") for s in sorted(sans)]
        return 5
    print(f"RÉGIMES : A = B + « {TEMOIN} », rien d'autre ({next(iter(sans))[:160]})")
    for nom, x, y in (("A1/A2", A1, A2), ("B1/B2", B1, B2)):
        if _jetons(x) != _jetons(y):
            print(f"TÉMOIN {nom} ≠ : l'instrument n'est pas déterministe, aucun verdict d'équivalence ni de temps"); return 6
    nf = [(n, L, s["k"]) for n, x in (("A1", A1), ("B1", B1), ("B2", B2), ("A2", A2)) for L, m in x["mesures"].items()
          for s in m if any(v is not None and not math.isfinite(v) for v in s["logprobs"])]
    if nf:
        print(f"NaN/inf dans les logprobs : {nf} (risque vLLM #27491 nommé au scellé) — aucun verdict"); return 8
    print("TÉMOINS A1 = A2 et B1 = B2 au jeton près ; logprobs tous finis")
    tout = True
    for L in A1["mesures"]:
        for sa, sb, sa2 in zip(A1["mesures"][L], B1["mesures"][L], A2["mesures"][L]):
            d = _divergence(sa["jetons"], sb["jetons"])
            dl, dt = _dlp(sa["logprobs"], sb["logprobs"]), _dlp(sa["logprobs"], sa2["logprobs"])
            tout = tout and d is None and dl in (None, 0.0)
            print(f"  L={L} k={sa['k']} : {len(sa['jetons'])} jetons, "
                  + ("IDENTIQUES" if d is None else f"PREMIÈRE DIVERGENCE au pas {d}")
                  + f" ; max |Δ logprob| A/B {dl} (témoin A1/A2 {dt})")
    print("E1(b) : jetons ET logprobs IDENTIQUES A/B sur toutes les invites" if tout else
          "E1(b) TOMBE : divergence ou logprob différent (voir ci-dessus) — E2, décision du chef")
    rc = 0
    for L in A1["mesures"]:
        bas, haut, faux_si = BANDES[int(L)]
        ta = [s["mur_prefill_ms"] for x in (A1, A2) for s in x["mesures"][L]]
        tb = [s["mur_prefill_ms"] for x in (B1, B2) for s in x["mesures"][L]]
        ma, mb = statistics.median(ta), statistics.median(tb)
        d = (mb / ma - 1) * 100
        ecart = (max(ta) - min(ta)) / ma * 100
        lecture = "dans la bande" if bas <= d <= haut else ("au-delà (mieux)" if d < bas else "en deçà")
        faux = faux_si is not None and d > faux_si
        rc = rc or (9 if faux else 0)
        print(f"  L={L} mur du préfill : A {ma:.1f} ms, B {mb:.1f} ms, Δ {d:+.1f} % (prédit {bas:+.0f} à {haut:+.0f} %) "
              f"— {lecture}{f' — FAUX (gain < {-faux_si:.0f} %)' if faux else ''} ; étendue témoin A {ecart:.1f} %")
    return rc


if __name__ == "__main__":
    if sys.argv[1:2] == ["bras"] and len(sys.argv) == 5:
        sys.exit(bras(*sys.argv[2:5]))
    if sys.argv[1:2] == ["comparer"] and len(sys.argv) == 6:
        sys.exit(comparer(*sys.argv[2:6]))
    sys.exit(__doc__)
