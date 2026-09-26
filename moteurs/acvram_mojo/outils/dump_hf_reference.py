"""Dump de référence HF bf16 (transformers, poids source non convertis) pour la porte KL de l'étape 1
Mojo : pour chaque invite de tests/invites.json, décodage glouton 8 jetons, logits complets par position.
Format attendu par outils/gpu/mesure/kl-api.py : {ids, cibles, logits[8, vocab]} par invite, en .pt.

Usage (à sec ou sous carte.sh) :
  PYTHONPATH=<racine> python dump_hf_reference.py <dossier_modele_hf> <invites.json> <dossier_sortie>
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELE, INVITES, SORTIE = sys.argv[1:4]
N_JETONS = 8
os.makedirs(SORTIE, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODELE)
mod = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.bfloat16).to("cuda:0")
mod.eval()

for inv in json.load(open(INVITES, encoding="utf-8")):
    texte = tok.apply_chat_template(inv["messages"], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(texte, add_special_tokens=False)
    entree = torch.tensor([ids], device="cuda:0")
    cibles, logits = [], []
    with torch.no_grad():
        for _ in range(N_JETONS):
            sortie = mod(entree)
            lg = sortie.logits[0, -1, :].float().cpu()
            jeton = int(lg.argmax())
            cibles.append(jeton)
            logits.append(lg)
            entree = torch.cat([entree, torch.tensor([[jeton]], device="cuda:0")], dim=1)
    torch.save({"ids": ids, "cibles": cibles, "logits": torch.stack(logits)},
               os.path.join(SORTIE, f"{inv['nom']}.pt"))
    print(inv["nom"], "ok", len(ids), "+", N_JETONS)
