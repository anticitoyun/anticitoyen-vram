"""Compare deux dumps de interroger.py par sha256 du texte. Usage : python comparer.py <a.json> <b.json> <sortie.json>"""
import hashlib
import json
import sys

A, B, SORTIE = sys.argv[1:4]
a, b = json.load(open(A, encoding="utf-8")), json.load(open(B, encoding="utf-8"))
res, tenues = [], 0
for nom in a:
    sha_a = hashlib.sha256(a[nom].encode()).hexdigest()
    sha_b = hashlib.sha256(b.get(nom, "").encode()).hexdigest()
    ok = sha_a == sha_b
    tenues += ok
    res.append({"nom": nom, "a": a[nom], "b": b.get(nom), "sha_a": sha_a, "sha_b": sha_b, "tenu": ok})
    print(nom, "TENU" if ok else "FAUX", sha_a[:16], sha_b[:16])
json.dump({"tenues": tenues, "sur": len(res), "invites": res}, open(SORTIE, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"PORTE {tenues}/{len(res)}")
sys.exit(0 if tenues == len(res) else 1)
