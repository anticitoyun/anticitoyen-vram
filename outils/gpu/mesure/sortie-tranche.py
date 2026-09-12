"""Le decoupage change-t-il ce que le modele DIT ?

L'ecart numerique brut ne tranche pas : un maximum relatif par element explose
des qu'un element vaut ~0, et le seuil du protocole (0,1 %) porte sur une
grandeur agregee, pas sur un maximum ponctuel. On mesure donc deux choses qui
ont un sens :

  1. la norme relative de l'ecart de sortie du noyau — agregee, insensible aux
     elements nuls ;
  2. LA SEQUENCE GENEREE a temperature 0, sous les deux reglages. C'est le
     critere qui decide : si le modele produit les memes jetons, l'ecart
     d'arrondi n'a aucune consequence observable.

Un processus par reglage : le graphe CUDA capture la grille, donc changer le
reglage a chaud ne changerait rien — defaut deja paye aujourd'hui.
"""
import os, sys, zlib, hashlib, torch
reglage = sys.argv[3]
if reglage != "auto":
    os.environ["ACVRAM_PA_CHUNK"] = reglage
else:
    os.environ.pop("ACVRAM_PA_CHUNK", None)
if torch.cuda.device_count() != 1:
    raise SystemExit("REFUS : une seule carte doit etre visible")
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels

chemin, lm = sys.argv[1], int(sys.argv[2])
# DU VRAI TEXTE, TOKENISE. Les ids pseudo-aleatoires (crc32) conviennent pour
# un DEBIT — le cout d'un pas ne depend pas du sens — mais pas pour juger la
# QUALITE : sur du charabia le modele part en repetition, et deux boucles
# degenerees divergent entre candidats quasi equiprobables sans que cela dise
# quoi que ce soit du noyau. Premiere version faite ainsi, resultat jete.
TEXTE = open("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw", encoding="utf-8",
             errors="ignore").read()
L = load_model(chemin, dtype=torch.bfloat16, max_model_len=8192)
eng = Engine(L, None, max_batch_size=1, max_model_len=8192)
from acvram.server.chat import load_tokenizer
tok = load_tokenizer(chemin)
if tok is None:
    raise SystemExit("REFUS : pas de tokenizer, on ne fabrique pas d'invite")
debut = 20000
prompt = tok.encode(TEXTE[debut:debut + lm * 6])[:lm]
if len(prompt) < lm // 2:
    raise SystemExit(f"REFUS : invite de {len(prompt)} jetons pour {lm} demandes")
# generate() DIFFUSE : chaque GenerationOutput porte le SEUL jeton du pas.
# Affecter au lieu d'accumuler ne gardait que le dernier — et comparer deux
# sequences par leur dernier jeton aurait rendu « identiques » deux textes
# differents, ou l'inverse.
jetons = []
for out in eng.generate(prompt, SamplingParams(temperature=0.0, max_tokens=128)):
    jetons.extend(int(t) for t in (out.token_ids or []))
h = hashlib.sha256(",".join(map(str, jetons)).encode()).hexdigest()[:16]
print(f"SORTIE\t{lm}\t{reglage}\t{len(jetons)}\t{h}\t{kernels._SO_HASH}")
print(f"# {lm} mots · {reglage} · {len(jetons)} jetons · sha {h}")
print("IDS " + ",".join(map(str, jetons)))
