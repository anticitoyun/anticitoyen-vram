"""La tranche adaptative tient-elle quand le lot grandit ?

BQ = B x QL est la premiere dimension de grille du noyau. Toutes nos mesures
l'ont laissee a 1 : le gain de la tranche adaptative est donc etabli a lot 1
et SUPPOSE ailleurs. On le mesure a BQ = 1, 4, 8, 12.

BRAS COMMUN : chunk 512 est mesure a chaque valeur de BQ. Sans lui, comparer
cette manche a une autre reviendrait a comparer deux moteurs — c'est le bras
d'appariement, pas un bras d'interet.

LE BIAIS QU'IL FAUT NEUTRALISER, ET IL N'EST PAS EVIDENT
    Pour fabriquer un lot on replique les arguments captures a B = 1. Mais
    repliquer les TABLES DE BLOCS ferait lire a toutes les sequences LES MEMES
    pages du cache KV : le L2 servirait B fois la meme chose et le noyau
    paraitrait invariant au lot par pur effet de cache. On DECALE donc les
    indices de blocs pour que chaque sequence lise des pages distinctes, et on
    refuse si le cache n'en a pas assez.

PREDICTION ECRITE AVANT
    Le travail par bloc ne depend pas de BQ : la grille passe de (1, HQ, C) a
    (BQ, HQ, C), donc le nombre de blocs est multiplie par BQ, et le terme par
    bloc mesure ce matin (3,76 ns) devrait suivre. J'attends donc :
      - un temps qui croit avec BQ, a peu pres proportionnellement au-dela du
        point ou la carte est remplie ;
      - un RAPPORT adaptatif/512 a peu pres CONSTANT en BQ : la tranche agit
        sur C, le lot sur BQ, et les deux dimensions sont independantes.
    Ce que chaque issue voudra dire :
      rapport constant        -> le gain se transporte au lot, rien a refaire ;
      rapport qui DECROIT     -> le gain etait un effet de sous-occupation a
                                 lot 1 : a lot eleve la carte est deja pleine
                                 et decouper ne sert plus a rien ;
      rapport qui CROIT       -> inattendu, et c'est cette croissance qu'il
                                 faudrait expliquer avant d'en profiter.
    Et le point sans hypothese : on releve aussi le nombre de blocs effectif et
    le temps du bras 512 lui-meme. Si CE bras ne croit pas avec BQ, la
    replication n'a pas pris et rien de ce tableau ne vaut.
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
LM = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
LOTS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,8,12").split(",")]
MOTS = open("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw", encoding="utf-8",
            errors="ignore").read().split()
L = load_model(chemin, dtype=torch.bfloat16, max_model_len=8192)
eng = Engine(L, None, max_batch_size=1, max_model_len=8192)
print(f"# binaire sha {kernels._SO_HASH}")

cap = []
vrai = kernels.paged_attention
def sonde(q, cache, tables, seq_lens, n_rep, scale, q_len=1, window=0):
    if not cap:
        cap.append((q, cache, tables, seq_lens, n_rep, scale, q_len, window))
    return vrai(q, cache, tables, seq_lens, n_rep, scale, q_len=q_len, window=window)
M.kernels.paged_attention = sonde

d = (7 * 4001 + LM) % max(1, len(MOTS) - LM - 50)
prompt = [(zlib.crc32(w.encode()) % 150000) + 10 for w in MOTS[d:d + LM]]
for _ in eng.generate(prompt, SamplingParams(temperature=0.0, max_tokens=4)):
    pass
if not cap:
    raise SystemExit("REFUS : le noyau pagine n'a pas ete appele")
q, cache, tables, seq_lens, n_rep, scale, q_len, window = cap[0]
nblk = tables.shape[1]
dispo = cache.k.shape[0]
print(f"# contexte {int(seq_lens.max())} · {nblk} blocs par sequence · "
      f"{dispo} blocs dans le cache")

def lot(B):
    """Replique a B sequences en DECALANT les pages, pour ne pas mesurer le L2."""
    if B * nblk > dispo:
        return None
    t = torch.cat([tables + i * nblk for i in range(B)], 0).contiguous()
    return (q.repeat(B, 1, 1).contiguous(), t, seq_lens.repeat(B).contiguous())

def chrono(appel, n=31):
    for _ in range(10):
        appel()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); appel(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]

print(f"{'BQ':>3} {'blocs':>6} {'512 us':>9} {'auto us':>9} {'rapport':>8}")
for B in LOTS:
    a = lot(B)
    if a is None:
        print(f"{B:3d} REFUS : {B*nblk} blocs demandes, {dispo} dans le cache "
              f"— on ne replique pas des pages, on n'aurait mesure que le L2")
        continue
    qb, tb, sb = a
    appel = lambda: vrai(qb, cache, tb, sb, n_rep, scale, q_len=q_len, window=window)
    os.environ["ACVRAM_PA_CHUNK"] = "512"
    t512 = chrono(appel)
    os.environ.pop("ACVRAM_PA_CHUNK", None)
    tauto = chrono(appel)
    print(f"{B:3d} {B*nblk:6d} {t512:9.2f} {tauto:9.2f} {t512/tauto:7.2f}x",
          flush=True)
    print(f"RESULTAT\t{B}\t{B*nblk}\t{t512:.3f}\t{tauto:.3f}\t{t512/tauto:.4f}",
          flush=True)
