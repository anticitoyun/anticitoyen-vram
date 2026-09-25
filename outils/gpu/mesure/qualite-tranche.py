"""Le decoupage change-t-il le RESULTAT du noyau pagine ?

POURQUOI PAS `acvram eval`. La perplexite passe par `model(batch, ...)`, le
forward dense : elle n'appelle jamais paged_attention. Lancee sur les deux
reglages elle rend quatre fois le meme chiffre au milliesme — non parce que le
decoupage est neutre, mais parce que l'instrument ne peut pas le voir. Une
barriere aveugle a ce qu'elle teste rend toujours « conforme ».

CE QUI EST COMPARE ICI : la sortie du noyau lui-meme, sur les MEMES arguments
captures dans une vraie generation, entre chunk 512 et la tranche adaptative.
Hors graphe, donc le reglage prend a chaque appel.

    classe A   identique au bit pres
    classe B   bits differents : l'ecart doit rester au niveau du bruit
               d'arrondi bf16, et il est CHIFFRE, pas suppose.

Un decoupage different change l'ordre des sommes : la classe B est attendue.
Ce qui compte est l'amplitude, comparee a une reference qui ne doit rien au
decoupage — l'ecart entre deux appels IDENTIQUES, qui donne le bruit du noyau.
"""
import os, sys, torch
# Pièce 211 : racine dérivée de __file__ — sans ceci, l'import acvram retombe sur l'installation
# editable et la garde a86fa1dd refuse depuis un worktree (constat poste5, 25/09, comme la 168).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if torch.cuda.device_count() != 1:
    raise SystemExit(f"REFUS : {torch.cuda.device_count()} cartes visibles")
os.environ.pop("ACVRAM_PA_CHUNK", None)
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels
import acvram.engine.model as M
import zlib

chemin = sys.argv[1]
LMOTS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "350,3000").split(",")]
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

def ecart(a, b):
    d = (a.float() - b.float()).abs()
    n = b.float().abs().clamp_min(1e-6)
    return d.max().item(), (d / n).max().item()

for lm in LMOTS:
    captures.clear()
    d = (7 * 4001 + lm) % max(1, len(MOTS) - lm - 50)
    prompt = [(zlib.crc32(w.encode()) % 150000) + 10 for w in MOTS[d:d + lm]]
    for _ in eng.generate(prompt, SamplingParams(temperature=0.0, max_tokens=8)):
        pass
    if not captures:
        print(f"{lm:5d} REFUS : le noyau pagine n'a pas ete appele")
        continue
    # REFERENCE INDEPENDANTE : l'attention DENSE, qui ne decoupe rien. Comparer
    # les deux decoupages entre eux ne dit pas lequel est juste — seul un tiers
    # qui ne partage pas leur defaut peut l'arbitrer. Si la tranche adaptative
    # n'est pas plus loin de la reference que 512, elle ne degrade rien.
    from acvram.engine.layers import decode_attention_fixed
    pires = (0.0, 0.0, 0.0, 0.0)
    ecarts_ref = []
    for args in captures[:48]:            # une passe de couches
        q, cache, tables, seq_lens, n_rep, scale, q_len, window = args
        def appel():
            return vrai(q, cache, tables, seq_lens, n_rep, scale,
                        q_len=q_len, window=window).clone()
        os.environ["ACVRAM_PA_CHUNK"] = "512"
        a1, a2 = appel(), appel()          # BRUIT : deux appels IDENTIQUES
        os.environ.pop("ACVRAM_PA_CHUNK", None)
        b1 = appel()                       # tranche adaptative
        br_abs, br_rel = ecart(a1, a2)
        ec_abs, ec_rel = ecart(b1, a1)
        try:
            kk, vv = cache.gather_fixed(tables, q.dtype)
            ref = decode_attention_fixed(q, kk, vv, seq_lens, n_rep, scale,
                                         window=window)
            n = ref.float().norm().clamp_min(1e-9)
            ecarts_ref.append(
                (((a1.float() - ref.float()).norm() / n).item(),
                 ((b1.float() - ref.float()).norm() / n).item(),
                 ((b1.float() - a1.float()).norm() / n).item()))
        except Exception as exc:               # pas de reference : on le DIT
            ecarts_ref.append((float("nan"),) * 3)
        pires = (max(pires[0], br_abs), max(pires[1], br_rel),
                 max(pires[2], ec_abs), max(pires[3], ec_rel))
    br_abs, br_rel, ec_abs, ec_rel = pires
    classe = "A (identique au bit pres)" if ec_abs == 0.0 else "B (bits differents)"
    verdict = ("CONFORME" if ec_abs <= max(br_abs, 0.0) or ec_rel <= 2e-2
               else "A EXAMINER")
    print(f"{lm:5d} mots · {len(captures)} appels · classe {classe}")
    print(f"      bruit du noyau (deux appels identiques) : abs {br_abs:.3e} "
          f"rel {br_rel:.3e}")
    print(f"      ecart 512 -> adaptatif                  : abs {ec_abs:.3e} "
          f"rel {ec_rel:.3e}   {verdict}")
    if ecarts_ref and ecarts_ref[0][0] == ecarts_ref[0][0]:   # non NaN
        r512 = max(e[0] for e in ecarts_ref)
        rauto = max(e[1] for e in ecarts_ref)
        entre = max(e[2] for e in ecarts_ref)
        mieux = "adaptatif AU MOINS AUSSI PROCHE" if rauto <= r512 * 1.05 else "adaptatif PLUS LOIN"
        print(f"      distance a l'attention dense (norme relative, pire couche) :")
        print(f"        chunk 512 {r512:.6e}   adaptatif {rauto:.6e}   -> {mieux}")
        # CE QUI DECIDE : l'ecart entre les deux decoupages compare a l'erreur
        # DEJA ACCEPTEE (cache int8 contre attention dense). S'il est deux
        # ordres de grandeur en dessous, changer de decoupage ne deplace rien
        # que la quantification n'ait deja deplace bien davantage.
        print(f"        ecart 512 <-> adaptatif   {entre:.6e}   soit "
              f"{entre / max(r512, 1e-12) * 100:.2f} % de l'erreur deja acceptee")
    else:
        print(f"      REFERENCE DENSE INDISPONIBLE — l'ecart ci-dessus "
              f"n'est arbitre par rien")
    print(f"RESULTAT\t{lm}\t{br_abs:.6e}\t{br_rel:.6e}\t{ec_abs:.6e}\t{ec_rel:.6e}\t{verdict}")
