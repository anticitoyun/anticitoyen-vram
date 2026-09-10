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
import os, sys, time, torch
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
print(f"# etape={etape} chunk={chunk} · un appel isole, evenements CUDA, "
      f"mediane de {N_MESURES}")

for lm in LMOTS:
    captures.clear()
    d = (7 * 4001) % max(1, len(MOTS) - lm - 50)
    prompt = [(abs(hash(w)) % 150000) + 10 for w in MOTS[d:d + lm]]
    for _ in eng.generate(prompt, par):
        pass
    if not captures:
        print(f"{lm:5d} REFUS : le noyau n'a pas ete appele")
        continue
    q, cache, tables, seq_lens, n_rep, scale, q_len, window = captures[0]
    n_ctx = int(seq_lens.max().item())
    n_blocs = int(tables.shape[1])
    ch = int(chunk)
    C = (n_blocs * 16 + ch - 1) // ch
    appel = lambda: vrai(q, cache, tables, seq_lens, n_rep, scale,
                         q_len=q_len, window=window)
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
    print(f"{lm:5d} mots · contexte {n_ctx:5d} · table {n_blocs:4d} blocs · "
          f"C={C:3d} · grille {32 * C:5d} blocs · "
          f"GPU {med:7.2f} us  [{p10:6.2f} – {p90:6.2f}]")
