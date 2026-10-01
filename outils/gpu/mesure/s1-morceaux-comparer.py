#!/usr/bin/env python3
"""Preuve carte S1 (kv31b levier 2 étape 1), comparaison à sec des bras A1, A2 (seul tenant) et B (morceaux 4 096).

Lit `scratchpad/poste6-s1-<bras>/{completion.json,metrics.json,serveur.log}` et juge P1-P5 du scellé
`poste6-s1-morceaux-scelle-carte-30-09.md`. Ne montre jamais le texte généré (REGLES § 6) : ids par sha256 des jetons,
logprobs par écart maximal. Usage : s1-morceaux-comparer.py [dossier-scratchpad]  → rc 0 si P1-P5 tenus, 1 sinon.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

BRAS = ("A1", "A2", "B")
MORCEAU_ATTENDU = {"A1": 0, "A2": 0, "B": 1}


def lire(base: Path, bras: str) -> dict:
    d = base / f"poste6-s1-{bras}"
    c = json.loads((d / "completion.json").read_text(encoding="utf-8"))
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
    exil = sum(int(x) for x in re.findall(r"plan réajusté : (\d+) MLP de plus en RAM hôte", log))
    regime = next((l for l in log.splitlines() if l.startswith("[acvram] régime")), "")
    return {
        "n": len(toks),
        "sha_ids": hashlib.sha256("\x1f".join(toks).encode()).hexdigest()[:16],
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


def main() -> int:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "scratchpad")
    r = {b: lire(base, b) for b in BRAS}
    ok = True
    for b in BRAS:
        x = r[b]
        print(f"{b}: jetons {x['n']} sha_ids {x['sha_ids']} prompt_tokens {x['prompt_tokens']} prefill_morceaux {x['morceaux']} "
              f"régime(morceaux@4096)={x['regime_morceaux']} exil_MLP {x['exil']} fin={x['fini']}")
    p1 = all(r[b]["morceaux"] == MORCEAU_ATTENDU[b] for b in BRAS) and r["B"]["regime_morceaux"] and not r["A1"]["regime_morceaux"]
    d2, n2 = ecart(r["A1"], r["A2"])
    d3, n3 = ecart(r["A1"], r["B"])
    p2 = d2 == 0.0 and n2 > 0
    seuil3 = 0.0 if p2 else 2 * d2
    p3 = n3 > 0 and d3 <= seuil3
    pt = {r[b]["prompt_tokens"] for b in BRAS}
    p4 = len(pt) == 1 and (r["A1"]["prompt_tokens"] or 0) >= 5120
    p5 = len({r[b]["exil"] for b in BRAS}) == 1
    print(f"P1 prise (B=1, A=0, régime) : {'tenu' if p1 else 'FAUX'}")
    print(f"P2 témoin A1/A2 : Δmax {d2:.3g} sur {n2} valeurs : {'tenu (au bit)' if p2 else 'FAUX — seuil P3 = 2×Δ = ' + format(seuil3, '.3g')}")
    print(f"P3 B/A1 : Δmax {d3:.3g} sur {n3} valeurs (seuil {seuil3:.3g}) : {'tenu' if p3 else 'FAUX'}")
    print(f"P4 prompt_tokens {sorted(pt)} ≥ 5120 et égaux : {'tenu' if p4 else 'FAUX'}")
    print(f"P5 exil MLP identique ({[r[b]['exil'] for b in BRAS]}) : {'tenu' if p5 else 'FAUX — comparaison contaminée'}")
    ok = p1 and p2 and p3 and p4 and p5
    print("VERDICT :", "P1-P5 tenus — morceaux 4096 au bit du seul tenant sur carte" if ok else "au moins un seuil FAUX, voir ci-dessus")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
