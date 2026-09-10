"""La tranche adaptative se transporte-t-elle au debit du moteur ?

PREDICTION ECRITE AVANT (docs/ATTENTION-DECODAGE-ETAT.md) : le noyau isole
gagne x3,1 et pese 49,5 % du temps GPU, donc Amdahl donne +50 % de debit.
Trois issues, et ce qu'elles voudront dire, sont inscrites la-bas AVANT cette
mesure — y compris celle ou le gain est nul.

ABBA, PAS A PUIS B. La carte derive en temperature : +3 W en douze passages
suffisent a fabriquer un ecart dans le sens de l'ordre de passage. Alterner ne
corrige rien, ABBA si — chaque bras voit la meme derive moyenne.

    A = ACVRAM_PA_CHUNK=512   l'ancien reglage, impose explicitement
    B = tranche adaptative    (variable absente : le noyau choisit)

Le noyau relit getenv A CHAQUE APPEL, donc les deux bras tournent dans le MEME
processus, sur le MEME modele charge une seule fois : aucune difference de
chargement, de cache disque ou d'etat de la carte ne peut s'y glisser.
"""
import os, sys, statistics as st, torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"REFUS : {torch.cuda.device_count()} cartes visibles, "
                     f"CUDA_VISIBLE_DEVICES=0 obligatoire")
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels
import zlib

chemin = sys.argv[1]
LMOTS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "350,3000").split(",")]
CYCLES = int(sys.argv[3]) if len(sys.argv) > 3 else 3      # ABBA par cycle
JETONS = 128
CORPUS = "/mnt/AI_GENERATOR/corpus/wiki.test.raw"
MOTS = open(CORPUS, encoding="utf-8", errors="ignore").read().split()

L = load_model(chemin, dtype=torch.bfloat16, max_model_len=8192)
eng = Engine(L, None, max_batch_size=1, max_model_len=8192)
print(f"# arbre {os.path.dirname(os.path.dirname(os.path.abspath(kernels.__file__)))}")
print(f"# binaire sha {kernels._SO_HASH or 'INCONNU'}")

def bras(nom, lm, i):
    if nom == "A":
        os.environ["ACVRAM_PA_CHUNK"] = "512"
    else:
        os.environ.pop("ACVRAM_PA_CHUNK", None)
    # INVITE DIFFERENTE A CHAQUE PASSAGE : une invite rejouee touche les memes
    # blocs de cache et le prefill n'est plus paye — le debit mesure devient
    # celui d'un cache chaud. crc32 et non hash(), qui est sale par processus.
    d = (i * 4001 + lm) % max(1, len(MOTS) - lm - 50)
    prompt = [(zlib.crc32(w.encode()) % 150000) + 10 for w in MOTS[d:d + lm]]
    par = SamplingParams(temperature=0.0, max_tokens=JETONS)
    # LES COMPTEURS DU MOTEUR SONT CUMULATIFS : `decode_seconds += ...` sur
    # toute la vie de l'Engine. Lire la valeur absolue apres chaque bras
    # donnerait la moyenne depuis le debut — donc un ecart entre bras qui
    # S'AMENUISE a chaque passage, et un ABBA qui ne peut plus rien separer.
    # On prend le DELTA.
    t0 = eng.stats.decode_seconds
    j0 = eng.stats.decode_tokens
    n = 0
    for _ in eng.generate(prompt, par):
        n += 1
    dec = eng.stats.decode_seconds - t0
    jetons = eng.stats.decode_tokens - j0
    if dec <= 0 or jetons <= 0:
        raise SystemExit("REFUS : le moteur n'a compte ni temps ni jetons de "
                         "decodage sur ce passage")
    if n < JETONS:
        raise SystemExit(f"REFUS : {n} jetons sur {JETONS} — generation "
                         f"interrompue, le debit porterait sur autre chose")
    return jetons / dec

for lm in LMOTS:
    A, B = [], []
    for i in range(CYCLES):
        A.append(bras("A", lm, 4 * i + 0))
        B.append(bras("B", lm, 4 * i + 1))
        B.append(bras("B", lm, 4 * i + 2))
        A.append(bras("A", lm, 4 * i + 3))
        print(f"  cycle {i+1}/{CYCLES} ({lm} mots) fait", flush=True)
    a, b = st.median(A), st.median(B)
    ea = (max(A) - min(A)) / a * 100
    eb = (max(B) - min(B)) / b * 100
    print(f"{lm:5d} mots · chunk 512 {a:7.2f} j/s (etendue {ea:.1f} %) · "
          f"adaptatif {b:7.2f} j/s (etendue {eb:.1f} %) · "
          f"gain {(b/a - 1)*100:+.1f} %", flush=True)
    print(f"RESULTAT\t{lm}\t{a:.3f}\t{b:.3f}\t{(b/a-1)*100:.2f}\t{ea:.1f}\t{eb:.1f}",
          flush=True)
