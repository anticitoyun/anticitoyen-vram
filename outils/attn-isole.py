"""Temps GPU d'UN appel isole de paged_attention.

POURQUOI LE MONTAGE PRECEDENT ETAIT FAUX
    300 appels enchaines, tampons de sortie partages : le GPU serialise, chaque
    appel attend l'ecriture du precedent. Le chiffre obtenu — 49,5 us, dont
    46,5 « fixes » — etait une latence de file. La bisection l'a montre : un
    noyau qui ecrit trois flottants et sort coutait AUTANT que le noyau complet
    (49,4 / 49,7 / 49,3 / 49,5 us). Et comme le balayage de grille venait du
    meme montage, la sous-parallelisation n'est PAS refutee : elle est non
    testee.

CE MONTAGE-CI
    Un seul appel entre deux evenements CUDA, puis synchronisation. Les
    evenements mesurent le temps GPU ENTRE LES MARQUEURS : la synchronisation
    qui suit n'entre pas dans l'intervalle, contrairement a un chronometre mur
    ou elle ajoutait 9 us par appel. Aucune file ne se forme, donc aucune
    serialisation artificielle.

LE TEMOIN EST LA BISECTION ELLE-MEME
    Si le montage est bon, ACVRAM_PA_ETAPE=0 (indices + trois flottants) doit
    couter BEAUCOUP MOINS que l'etape 3. S'il rend encore le meme temps, le
    montage est encore faux et RIEN de ce qu'il mesure ne vaut.
"""
import os, sys, time, zlib, torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels

chemin = sys.argv[1]
LMOTS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "350,3000").split(",")]
N_MESURES = 51
CORPUS = "/mnt/AI_GENERATOR/corpus/wiki.test.raw"
MOTS = open(CORPUS, encoding="utf-8", errors="ignore").read().split()

L = load_model(chemin, dtype=torch.bfloat16, max_model_len=8192)
eng = Engine(L, None, max_batch_size=1, max_model_len=8192)
par = SamplingParams(temperature=0.0, max_tokens=8)

captures = []
vrai = kernels.paged_attention
def sonde(q, cache, tables, seq_lens, n_rep, scale, q_len=1, window=0):
    if not captures:
        captures.append((q, cache, tables, seq_lens, n_rep, scale, q_len, window))
    return vrai(q, cache, tables, seq_lens, n_rep, scale, q_len=q_len, window=window)
import acvram.engine.model as M
M.kernels.paged_attention = sonde

etape = os.environ.get("ACVRAM_PA_ETAPE", "3")
chunk = os.environ.get("ACVRAM_PA_CHUNK", "512")
# L'ETAPE REDUITE NE DOIT PAS ALIMENTER LE MOTEUR. A l'etape 0 le noyau sort
# avant d'ecrire `out` — qui, quand C == 1, est rendu tel quel a l'appelant :
# un `torch::empty` jamais initialise. Laisser le moteur GENERER dans cet etat
# lui fait consommer de la memoire non initialisee couche apres couche, et la
# generation finit en acces memoire illegal — observe deux fois, une fois dans
# la capture de graphe, une fois dans un matmul. La bisection est un instrument
# de mesure, pas un mode de fonctionnement : on genere en etape 3, et on ne
# bascule que pour les appels CHRONOMETRES.
# Le noyau relit getenv A CHAQUE APPEL (pas de static), donc la bascule depuis
# Python suffit — verifie dans acvram_kernels.cu.
os.environ.pop("ACVRAM_PA_ETAPE", None)          # generation : noyau complet

def bascule(v):
    if v == "3":
        os.environ.pop("ACVRAM_PA_ETAPE", None)
    else:
        os.environ["ACVRAM_PA_ETAPE"] = v
print(f"# etape={etape} chunk={chunk} · un appel isole, evenements CUDA, "
      f"mediane de {N_MESURES}")

for lm in LMOTS:
    captures.clear()
    bascule("3")                  # la generation qui suit doit etre correcte
    d = (7 * 4001) % max(1, len(MOTS) - lm - 50)
    # `hash()` sur str est SALE PAR PROCESSUS : deux executions ne
    # construiraient pas la meme invite, donc ne toucheraient pas les memes
    # blocs de cache — au moment precis ou l'on rejoue pour comparer.
    prompt = [(zlib.crc32(w.encode()) % 150000) + 10 for w in MOTS[d:d + lm]]
    for _ in eng.generate(prompt, par):
        pass
    if not captures:
        print(f"{lm:5d} REFUS : le noyau n'a pas ete appele")
        continue
    q, cache, tables, seq_lens, n_rep, scale, q_len, window = captures[0]
    n_ctx = int(seq_lens.max().item())
    n_blocs = int(tables.shape[1])
    ch = int(chunk)
    C_demande = (n_blocs * 16 + ch - 1) // ch   # DEMANDE, pas observe
    appel = lambda: vrai(q, cache, tables, seq_lens, n_rep, scale,
                         q_len=q_len, window=window)
    # PARTICIPATION OBSERVEE : a l'etape 0 chaque bloc s'annonce par un
    # atomique. Une colonne « grille » reconstruite en Python dirait ce qui a
    # ete DEMANDE et corroborerait une fausse refutation ; ceci dit ce qui a
    # TOURNE. Le compteur est remis a zero, un appel, puis relu.
    bascule(etape)                # a partir d'ici seulement, l'etape mesuree
    parts = None
    if etape == "0" and hasattr(kernels.get_extension(), "paged_attn_participants"):
        ext = kernels.get_extension()
        ext.paged_attn_participants(True)          # remise a zero
        appel()
        torch.cuda.synchronize()
        parts = ext.paged_attn_participants(True)

    for _ in range(20):
        appel()
    torch.cuda.synchronize()

    temps = []
    for _ in range(N_MESURES):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        appel()                       # UN SEUL appel entre les marqueurs
        e1.record()
        torch.cuda.synchronize()      # hors intervalle : n'entre pas dans la mesure
        temps.append(e0.elapsed_time(e1) * 1000.0)
    temps.sort()
    med = temps[len(temps) // 2]
    p10, p90 = temps[len(temps) // 10], temps[-1 - len(temps) // 10]
    att = 32 * C_demande
    if parts is None:
        obs = "participants non observes (etape != 0)"
    elif parts == att:
        obs = f"participants {parts} = grille demandee"
    else:
        obs = f"participants {parts} < grille demandee {att} — GRILLE INERTE"
    print(f"{lm:5d} mots · contexte {n_ctx:5d} · table {n_blocs:4d} blocs · "
          f"C demande {C_demande:3d} · GPU {med:7.2f} us "
          f"[{p10:6.2f} – {p90:6.2f}] · {obs}")
    # Ligne machine, pour que l'orchestrateur n'ait pas a decouper la ligne
    # humaine : un separateur qui change casserait le TEMOIN sans le dire, et
    # un temoin casse se lit « pas de separation » — donc « arret », donc un
    # resultat inverse. Champs : lm etape chunk n_ctx n_blocs C med p10 p90
    # participants attendus.
    print("RESULTAT\t%d\t%s\t%s\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s\t%d"
          % (lm, etape, chunk, n_ctx, n_blocs, C_demande, med, p10, p90,
             "NA" if parts is None else parts, att), flush=True)
