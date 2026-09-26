"""Le tampon partiel se stabilise-t-il, et les chiffres tiennent-ils ?

L'ancien cache ajoutait une entree DEFINITIVE par forme (BQ, HQ, C, D) : 16
valeurs de C a chunk 512, mais 128 avec la tranche adaptative — 135 Mo au lieu
de 2,2. Le remede garde un seul tampon, au pire cas vu.

DEUX CHOSES A PROUVER, pas une :
  1. les octets retenus se STABILISENT quand on balaie les longueurs ;
  2. le temps du noyau NE BOUGE PAS — un remede memoire qui coute du debit
     n'est pas un remede.
"""
import os, sys, zlib, torch
# Pièce 211 : racine dérivée de __file__ — sans ceci, l'import acvram retombe sur l'installation
# editable et la garde a86fa1dd refuse depuis un worktree (constat poste5, 25/09, comme la 168).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if torch.cuda.device_count() != 1:
    raise SystemExit("REFUS : une seule carte doit etre visible")
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels
import acvram.engine.model as M

chemin = sys.argv[1]
ext = kernels.get_extension()
if ext is None or not hasattr(ext, "paged_attn_tampon_octets"):
    raise SystemExit("REFUS : le binaire ne porte pas le temoin de tampon")
MOTS = open("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw", encoding="utf-8",
            errors="ignore").read().split()
L = load_model(chemin, dtype=torch.bfloat16, max_model_len=8192)
eng = Engine(L, None, max_batch_size=1, max_model_len=8192)
print(f"# binaire sha {kernels._SO_HASH}")

captures = []
vrai = kernels.paged_attention
def sonde(q, cache, tables, seq_lens, n_rep, scale, q_len=1, window=0):
    captures.append((q, cache, tables, seq_lens, n_rep, scale, q_len, window))
    return vrai(q, cache, tables, seq_lens, n_rep, scale, q_len=q_len, window=window)
M.kernels.paged_attention = sonde

vus = []
derniere = None
for lm in [200, 400, 700, 1100, 1600, 2200, 3000, 3800, 300, 3000]:
    captures.clear()
    d = (7 * 4001 + lm) % max(1, len(MOTS) - lm - 50)
    prompt = [(zlib.crc32(w.encode()) % 150000) + 10 for w in MOTS[d:d + lm]]
    for _ in eng.generate(prompt, SamplingParams(temperature=0.0, max_tokens=4)):
        pass
    if captures:
        derniere = captures[0]
    o = ext.paged_attn_tampon_octets()
    nb = int(captures[0][2].shape[1]) if captures else -1
    vus.append(o)
    print(f"{lm:5d} mots · table {nb:4d} blocs · tampon {o/1e6:7.3f} Mo", flush=True)

croissances = sum(1 for a, b in zip(vus, vus[1:]) if b > a)
print(f"# croissances du tampon : {croissances} sur {len(vus)-1} transitions")
print(f"# maximum retenu : {max(vus)/1e6:.3f} Mo "
      f"(ancien cache cumule : 135,3 Mo)")

# LE TEMPS N'A PAS BOUGE ? Meme protocole que le balayage : etape 3, chunk 64.
# `captures` peut etre VIDE : un pas rejoue par un graphe CUDA n'appelle pas la
# sonde Python — c'est ce que disent les lignes « table -1 » ci-dessus. On
# garde donc la derniere capture REELLE au lieu de supposer qu'il y en a une.
if not derniere:
    raise SystemExit("REFUS : aucun appel n'a traverse la sonde, rien a "
                     "chronometrer (tous les pas rejoues par graphe)")
q, cache, tables, seq_lens, n_rep, scale, q_len, window = derniere
appel = lambda: vrai(q, cache, tables, seq_lens, n_rep, scale, q_len=q_len, window=window)
os.environ["ACVRAM_PA_CHUNK"] = "64"
for _ in range(20):
    appel()
torch.cuda.synchronize()
t = []
for _ in range(51):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); appel(); e1.record(); torch.cuda.synchronize()
    t.append(e0.elapsed_time(e1) * 1000.0)
t.sort()
print(f"# noyau chunk 64, contexte {int(seq_lens.max().item())} : "
      f"{t[len(t)//2]:.2f} us [{t[5]:.2f} - {t[-6]:.2f}]  "
      f"(reference du balayage : 25,66 us a contexte 3007)")
