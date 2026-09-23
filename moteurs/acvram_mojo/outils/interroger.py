"""Interroge un serveur OpenAI (/v1/chat/completions, glouton) sur les invites de tests/invites.json,
écrit {nom: texte} en JSON. Usage : python interroger.py <invites.json> <port> <sortie.json>
"""
import json
import sys
import urllib.request

INVITES, PORT, SORTIE = sys.argv[1:4]

res = {}
for inv in json.load(open(INVITES, encoding="utf-8")):
    corps = json.dumps({
        "model": "defaut", "messages": inv["messages"], "temperature": 0.0,
        "top_p": 1.0, "max_tokens": 128, "seed": 0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=corps,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        rep = json.load(r)
    res[inv["nom"]] = rep["choices"][0]["message"]["content"]
    print(inv["nom"], "ok", len(res[inv["nom"]]), "car.")

json.dump(res, open(SORTIE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
