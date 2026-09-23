"""Fixture du test tokeniseur : texte rendu et ids d'invite du moteur PYTHON, chemin conversation
(`render_chat` puis `Tokenizer.encode`, app.py:1020). À sec : aucune carte.

Usage : CUDA_VISIBLE_DEVICES= PYTHONPATH=<racine> python moteurs/acvram_rust/outils/ids_python.py <modèle> <invites.json> <sortie.json>
"""
import hashlib
import json
import sys

from acvram.server.chat import load_tokenizer, render_chat

modele, invites, sortie = sys.argv[1:4]
tok = load_tokenizer(modele)
res = []
for inv in json.load(open(invites, encoding="utf-8")):
    texte = render_chat(tok, inv["messages"], True)
    ids = tok.encode(texte)
    res.append({"nom": inv["nom"], "texte": texte, "ids": ids,
                "sha256_ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
json.dump({"modele": modele.rstrip("/").split("/")[-1], "gabarit": tok.gabarit_effectif, "invites": res},
          open(sortie, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok", len(res), "gabarit", tok.gabarit_effectif)
