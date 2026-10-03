#!/usr/bin/env python3
"""Preuve carte S1 (kv31b levier 2 étape 1), comparaison à sec des bras A1, A2 (seul tenant) et B (morceaux 4 096).

Lit `scratchpad/poste6-s1-<bras>/{completion.json,metrics.json,serveur.log}` et juge P1-P5 du scellé
`poste6-s1-morceaux-scelle-carte-30-09.md`. Ne montre jamais le texte généré (REGLES § 6) : ids par sha256 des jetons,
logprobs par écart maximal. Usage : s1-morceaux-comparer.py [dossier-scratchpad]  → rc 0 si P1-P5 tenus, 1 sinon.

Le témoin reprise et P3 se lisent sur le logprob du MÊME candidat (`ecart_candidats`, 02/10) : jusqu'au 02/10 22 h, quand
le premier jeton basculait, le script soustrayait les logprobs des deux jetons CHOISIS — deux jetons différents. Sur
l'invite dense de S1 (deux candidats de tête à 0,01-0,08 l'un de l'autre) il affichait 0,0299 et 0,00403 pour des écarts
réels de 0,041 et 0,014 (revue/poste6-bf16-reduction-verdict-carte-02-10.md). Test : tests/test_s1_comparateur_candidat.py.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

BRAS = ("A1", "A2", "B")
MORCEAU_ATTENDU = {"A1": 0, "A2": 0, "B": 1}


def lire(base: Path, bras: str, fichier: str = "completion.json") -> dict:
    d = base / f"poste6-s1-{bras}"
    if not (d / fichier).exists():
        return {}
    c = json.loads((d / fichier).read_text(encoding="utf-8"))
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    log = (d / "serveur.log").read_text(encoding="utf-8", errors="replace")
    ch = c["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    vals = [v for v in (lp.get("token_logprobs") or [])]
    tops = lp.get("top_logprobs") or []
    # top-10 aplati en (rang, valeur) — les clés (texte des jetons) ne sont pas affichées, seulement hachées.
    plat = []
    for pos in tops:
        if pos is None:
            plat.append(None)
            continue
        plat.append(sorted(((hashlib.sha256(k.encode()).hexdigest()[:12], float(v)) for k, v in pos.items()),
                           key=lambda kv: (-kv[1], kv[0])))
    regime = next((l for l in log.splitlines() if l.startswith("[acvram] régime")), "")
    # `couches_exilées=8/60` du régime (état final), pas la somme des « N MLP de plus » (chaque plan remplace le précédent :
    # 15 puis 8 lisaient 23 le 01/10 pour 8 exilés) ; repli sur la somme si la ligne de régime manque
    m_ex = re.search(r"couches_exilées=(\d+)/", regime)
    exil = int(m_ex.group(1)) if m_ex else sum(int(x) for x in re.findall(r"plan réajusté : (\d+) MLP de plus en RAM hôte", log))
    return {
        "n": len(toks),
        "sha_ids": hashlib.sha256("\x1f".join(toks).encode()).hexdigest()[:16],
        "cles": [hashlib.sha256(k.encode()).hexdigest()[:12] for k in toks],        # même hachage que les clés de `tops`
        "vals": vals, "tops": plat,
        "prompt_tokens": c.get("usage", {}).get("prompt_tokens"),
        "morceaux": int(m.get("engine", {}).get("prefill_morceaux", -1)),
        "exil": exil, "regime_morceaux": "(morceaux@4096)" in regime, "fini": ch.get("finish_reason"),
    }


def ecart(x: dict, y: dict) -> tuple[float, int]:
    """(Δ max sur token_logprobs + top-10, nombre de valeurs comparées) ; inf si ids ou rangs diffèrent."""
    if x["sha_ids"] != y["sha_ids"] or x["n"] != y["n"]:
        return float("inf"), 0
    d, n = 0.0, 0
    for a, b in zip(x["vals"], y["vals"]):
        if (a is None) != (b is None):
            return float("inf"), n
        if a is not None:
            d = max(d, abs(a - b)); n += 1
    for pa, pb in zip(x["tops"], y["tops"]):
        if (pa is None) != (pb is None):
            return float("inf"), n
        if pa is None:
            continue
        if [k for k, _ in pa] != [k for k, _ in pb]:
            return float("inf"), n
        for (_, a), (_, b) in zip(pa, pb):
            d = max(d, abs(a - b)); n += 1
    return d, n


def ecart_candidats(ref: dict, autre: dict) -> tuple[float, int, int | None]:
    """(Δ max du logprob du MÊME candidat, valeurs comparées, position de divergence des jetons choisis ou None).

    Position par position tant que les deux suites conditionnent sur le même texte : jusqu'à la première position où les
    jetons choisis divergent, INCLUSE — au-delà, les distributions ne portent plus sur le même contexte et ne se comparent pas.
    Candidats comparés : ceux du top de la référence présents dans le top de l'autre. Le jeton CHOISI par la référence doit
    s'y trouver, sinon inf : il n'y a alors rien de comparable, et soustraire le logprob du jeton choisi par l'autre
    comparerait deux jetons différents (la faute d'avant)."""
    d, n = 0.0, 0
    commun = min(len(ref["cles"]), len(autre["cles"]))
    for i in range(commun):
        ka, kb = ref["cles"][i], autre["cles"][i]
        ta = dict((ref["tops"][i] if i < len(ref["tops"]) else None) or ())
        tb = dict((autre["tops"][i] if i < len(autre["tops"]) else None) or ())
        ta.setdefault(ka, ref["vals"][i])
        tb.setdefault(kb, autre["vals"][i])
        if ta[ka] is None or tb.get(ka) is None:
            return float("inf"), n, (i if ka != kb else None)
        for k, a in ta.items():
            if a is not None and tb.get(k) is not None:
                d = max(d, abs(a - tb[k])); n += 1
        if ka != kb:
            return d, n, i
    return d, n, (commun if len(ref["cles"]) != len(autre["cles"]) else None)


def _bascule(pos: int | None) -> str:
    return "" if pos is None else f", jetons choisis divergents à la position {pos} (même candidat comparé jusque-là)"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    base = Path(argv[0] if argv else Path(__file__).resolve().parents[3] / "scratchpad")
    r = {b: lire(base, b) for b in BRAS}
    ok = True
    for b in BRAS:
        x = r[b]
        print(f"{b}: jetons {x['n']} sha_ids {x['sha_ids']} prompt_tokens {x['prompt_tokens']} prefill_morceaux {x['morceaux']} "
              f"régime(morceaux@4096)={x['regime_morceaux']} exil_MLP {x['exil']} fin={x['fini']}")
    # la chauffe compte ses propres morceaux (2 essais à 10 240 le 01/10) : B > 0, A = 0
    p1 = r["A1"]["morceaux"] == 0 and r["A2"]["morceaux"] == 0 and r["B"]["morceaux"] > 0 and r["B"]["regime_morceaux"] and not r["A1"]["regime_morceaux"]
    d2, n2 = ecart(r["A1"], r["A2"])
    d3, n3, v3 = ecart_candidats(r["A1"], r["B"])
    p2 = d2 == 0.0 and n2 > 0
    # REGLES § 4 (chef 01/10) : le seuil des morceaux est 2 × l'écart du témoin reprise (même requête rejouée sur le serveur
    # A1, REQUETES=2 → completion-2.json : K/V de l'invite relus du cache), jamais « au bit » (SDPA par blocs de clés)
    rep = lire(base, "A1", "completion-2.json")
    dr, nr, vr = ecart_candidats(r["A1"], rep) if rep else (float("nan"), 0, None)
    seuil3 = 2 * dr if rep else (0.0 if p2 else 2 * d2)
    # un témoin incomparable (inf) ne rend pas P3 « tenu » par défaut : un seuil infini ne juge rien
    p3 = n3 > 0 and seuil3 != float("inf") and d3 <= seuil3
    pt = {r[b]["prompt_tokens"] for b in BRAS}
    p4 = len(pt) == 1 and (r["A1"]["prompt_tokens"] or 0) >= 5120
    p5 = len({r[b]["exil"] for b in BRAS}) == 1
    print(f"P1 prise (B>0, A=0, régime) : {'tenu' if p1 else 'FAUX'}")
    print(f"P2 témoin A1/A2 : Δmax {d2:.3g} sur {n2} valeurs : {'tenu (au bit)' if p2 else 'FAUX — seuil P3 = 2×Δ = ' + format(seuil3, '.3g')}")
    print(f"témoin reprise A1 (requête rejouée) : Δ {dr:.3g} sur {nr} valeurs" + _bascule(vr)
          + ("" if rep else " — ABSENT (REQUETES=2 non passé) : seuil = témoin A1/A2"))
    print(f"P3 B/A1 : Δ {d3:.3g} sur {n3} valeurs{_bascule(v3)} ≤ 2 × témoin reprise = {seuil3:.3g} : {'tenu' if p3 else 'FAUX'}")
    print(f"P4 prompt_tokens {sorted(pt)} ≥ 5120 et égaux : {'tenu' if p4 else 'FAUX'}")
    print(f"P5 exil MLP identique ({[r[b]['exil'] for b in BRAS]}) : {'tenu' if p5 else 'FAUX — comparaison contaminée'}")
    ok = p1 and p2 and p3 and p4 and p5
    print("VERDICT :", "P1-P5 tenus — morceaux dans 2 × le témoin reprise (REGLES § 4)" if ok else "au moins un seuil FAUX, voir ci-dessus")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
