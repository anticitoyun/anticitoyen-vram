"""UN bras de la validation : debit de decodage a chunk fixe, plus son temoin.

POURQUOI UN PROCESSUS PAR BRAS. La premiere version alternait les deux bras
DANS le meme processus, pour qu'aucune difference de chargement ne s'y glisse.
C'etait juste, et c'est exactement ce qui a tue la mesure : le decodage passe
par un GRAPHE CUDA capture une fois, et un graphe enregistre la grille
`dim3 g1(BQ, HQ, C)` AU MOMENT DE LA CAPTURE. Changer ACVRAM_PA_CHUNK ensuite
ne rejoue pas un nouveau lancement, il rejoue celui qui a ete capture. Les deux
bras executaient le meme graphe : +0,1 % aux deux contextes, avec une etendue
de 0,2 % sur le bras inerte contre 7,1 % sur l'autre — la signature d'un bras
qui n'a jamais tourne.

    Le montage choisi pour eliminer une difference parasite avait elimine
    la difference etudiee.

Le chunk est donc pose AVANT le chargement, et l'alternance se fait dehors.

LE TEMOIN. Un bras doit prouver qu'il a pris. Apres les passages, on
chronometre UN appel isole du noyau avec les arguments reellement captures :
si les deux bras rendent le meme temps de noyau, le reglage n'a pas pris et le
debit ne dit rien. Ce controle manquait, et son absence a rendu une mesure
entiere ininterpretable.
"""
import os, sys, statistics as st, zlib, torch

chemin, lm, reglage, npass = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
if reglage != "auto":
    os.environ["ACVRAM_PA_CHUNK"] = reglage      # AVANT tout chargement
else:
    os.environ.pop("ACVRAM_PA_CHUNK", None)

if torch.cuda.device_count() != 1:
    raise SystemExit(f"REFUS : {torch.cuda.device_count()} cartes visibles")
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels
import acvram.engine.model as M

JETONS = 128
MOTS = open("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw", encoding="utf-8",
            errors="ignore").read().split()
L = load_model(chemin, dtype=torch.bfloat16, max_model_len=8192)
eng = Engine(L, None, max_batch_size=1, max_model_len=8192)

captures = []
vrai = kernels.paged_attention
def sonde(q, cache, tables, seq_lens, n_rep, scale, q_len=1, window=0):
    if not captures:
        captures.append((q, cache, tables, seq_lens, n_rep, scale, q_len, window))
    return vrai(q, cache, tables, seq_lens, n_rep, scale, q_len=q_len, window=window)
M.kernels.paged_attention = sonde

debits = []
for i in range(npass):
    d = (i * 4001 + lm) % max(1, len(MOTS) - lm - 50)
    prompt = [(zlib.crc32(w.encode()) % 150000) + 10 for w in MOTS[d:d + lm]]
    t0, j0 = eng.stats.decode_seconds, eng.stats.decode_tokens   # compteurs CUMULATIFS
    n = 0
    for _ in eng.generate(prompt, SamplingParams(temperature=0.0, max_tokens=JETONS)):
        n += 1
    dt, dj = eng.stats.decode_seconds - t0, eng.stats.decode_tokens - j0
    if n < JETONS or dt <= 0:
        raise SystemExit(f"REFUS : {n} jetons sur {JETONS}, {dt:.4f} s comptes")
    debits.append(dj / dt)

# TEMOIN : le reglage a-t-il pris ?
noyau = float("nan")
if captures:
    q, cache, tables, seq_lens, n_rep, scale, q_len, window = captures[0]
    appel = lambda: vrai(q, cache, tables, seq_lens, n_rep, scale,
                         q_len=q_len, window=window)
    for _ in range(10):
        appel()
    torch.cuda.synchronize()
    ts = []
    for _ in range(31):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); appel(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort(); noyau = ts[len(ts) // 2]

med = st.median(debits)
print(f"PASSAGE\t{lm}\t{reglage}\t{med:.3f}\t{min(debits):.3f}\t{max(debits):.3f}"
      f"\t{noyau:.3f}\t{kernels._SO_HASH}", flush=True)
print(f"# {lm} mots · reglage {reglage} · {med:.2f} j/s "
      f"[{min(debits):.2f}-{max(debits):.2f}] · noyau isole {noyau:.2f} us", flush=True)
