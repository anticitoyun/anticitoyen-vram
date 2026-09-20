#!/usr/bin/env python3
"""Durée des prises de carte, lue dans le journal du verrou — la septième ligne
d'un verdict (« durée : prévue X / tenue Y »), jamais estimée.

    duree-prises.py [--depuis HH:MM] [--journal CHEMIN] [--n N]

Pour chaque prise rendue depuis l'heure donnée : début, commande, type,
tenue (s), TIMEOUT s'il y a eu échéance ; puis le total tenu, le nombre de
prises, la plus longue, et les trous > 60 s entre deux prises (le temps qui
n'est pas de la carte). Le journal est `$ACVRAM_VERROU.journal`, par défaut
celui de la carte 0 (`outils/carte.sh:161`).
"""
import argparse
import os
import re
import sys
from datetime import datetime

_LIGNE = re.compile(r"^(\S+) (prise|rendue|TIMEOUT)\s+(\S+)\s+(.*?)\s+(mesure|etat|service)(?:\s+tenue=(\d+)s)?\s*$")


def lire(lignes):
    """[{debut, fin, pid, nom, type, tenue, timeout}] pour chaque prise rendue,
    dans l'ordre du journal ; une prise sans « rendue » est en cours."""
    en_cours, prises = {}, []
    for l in lignes:
        m = _LIGNE.match(l.rstrip("\n"))
        if not m:
            continue
        quand, quoi, pid, nom, typ, tenue = m.groups()
        t = datetime.fromisoformat(quand)
        if quoi == "prise":
            en_cours[pid] = {"debut": t, "pid": pid, "nom": nom.strip(), "type": typ, "timeout": False}
        elif quoi == "TIMEOUT" and pid in en_cours:
            en_cours[pid]["timeout"] = True
        elif quoi == "rendue":
            p = en_cours.pop(pid, {"debut": None, "pid": pid, "nom": nom.strip(), "type": typ, "timeout": False})
            p["fin"] = t
            p["tenue"] = int(tenue) if tenue is not None else (t - p["debut"]).seconds if p["debut"] else None
            prises.append(p)
    return prises, list(en_cours.values())


def resume(prises, depuis=None):
    sel = [p for p in prises if p["fin"] and (depuis is None or p["fin"].time() >= depuis)]
    trous = []
    for a, b in zip(sel, sel[1:]):
        if a["fin"] and b["debut"]:
            g = (b["debut"] - a["fin"]).total_seconds()
            if g > 60:
                trous.append((a["fin"], b["debut"], int(g)))
    total = sum(p["tenue"] or 0 for p in sel)
    plus_longue = max(sel, key=lambda p: p["tenue"] or 0) if sel else None
    return sel, total, plus_longue, trous


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depuis", help="HH:MM (heure locale) : ne compter que les prises rendues après")
    ap.add_argument("--journal", default=os.environ.get("ACVRAM_VERROU", "/tmp/acvram-carte-0.lock") + ".journal")
    ap.add_argument("--n", type=int, default=0, help="n'imprimer que les N dernières prises (0 = toutes)")
    ns = ap.parse_args(argv[1:])
    depuis = datetime.strptime(ns.depuis, "%H:%M").time() if ns.depuis else None
    try:
        with open(ns.journal) as f:
            prises, en_cours = lire(f)
    except OSError as e:
        print(f"journal illisible : {e}", file=sys.stderr)
        return 2
    sel, total, plus_longue, trous = resume(prises, depuis)
    for p in sel[-ns.n:] if ns.n else sel:
        print(f"{p['debut'].strftime('%H:%M:%S') if p['debut'] else '??:??:??'}  {p['type']:<7} {p['nom'][:32]:<32} "
              f"tenue={p['tenue']}s{'  TIMEOUT' if p['timeout'] else ''}")
    print(f"prises : {len(sel)} ; tenue totale : {total} s ({total / 60:.1f} min) ; "
          f"plus longue : {plus_longue['tenue'] if plus_longue else 0} s"
          f"{' (TIMEOUT)' if plus_longue and plus_longue['timeout'] else ''}")
    for a, b, g in trous:
        print(f"trou : {a.strftime('%H:%M:%S')} → {b.strftime('%H:%M:%S')} = {g} s hors carte")
    for p in en_cours:
        print(f"en cours : {p['nom']} depuis {p['debut'].strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
