"""Le plancher suit-il les BLOCS ou les VAGUES ?

PREDICTION ECRITE AVANT LA MESURE
---------------------------------
Le plancher du lancement croit avec le nombre de blocs : 15,36 us a 1504
blocs, 21,47 a 3008, 33,73 a 6016. Ni proportionnel aux blocs (x4 -> x2,2) ni
proportionnel aux vagues (x3 -> x2,2). Deux mecanismes possibles, et ils se
distinguent par la FORME de la courbe, pas par sa pente :

  (A) VAGUES — le GPU ne peut heberger que N blocs residents ; au-dela, un
      bloc attend qu'un autre finisse. Le temps doit alors monter EN MARCHES :
      plat tant qu'on reste dans une vague, SAUT NET au franchissement.
      => on doit voir une discontinuite entre `resident` et `resident + 1`.

  (B) DEBIT DE DISTRIBUTION — le repartiteur (GigaThread) place les blocs a
      cadence finie. Le temps monte alors CONTINUMENT avec le nombre de
      blocs, sans marche, et la pente donne directement les blocs par
      microseconde.

  (C) les deux : marches VISIBLES posees sur une pente.

Ce que chaque issue voudra dire est ecrit ici AVANT :
  marche nette, plat entre    -> (A) vagues. Le plancher est une capacite, et
                                 il se calcule : ceil(blocs / residents).
  pente continue, aucune marche -> (B) debit. Le plancher est un DEBIT et se
                                 mesure en blocs/us ; la sous-linearite vient
                                 alors d'un recouvrement entre distribution et
                                 execution, pas d'une capacite.
  marches sur une pente        -> (C), et il faudra separer les deux termes.
  temps PLAT sur tout le balayage -> l'instrument ne voit rien : ARRET, le
                                 travail par bloc est trop gros et masque le
                                 plancher.

TRAVAIL PAR BLOC CONSTANT. C'est la condition de l'experience : `banc_fma`
enchaine K FMA dependantes, donc le travail ne depend QUE de K, jamais de la
grille. Seul le nombre de blocs varie.
"""
import os, sys, torch
# Pièce 211 : racine dérivée de __file__ — sans ceci, l'import acvram retombe sur l'installation
# editable et la garde a86fa1dd refuse depuis un worktree (constat poste5, 25/09, comme la 168).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if torch.cuda.device_count() != 1:
    raise SystemExit("REFUS : une seule carte doit etre visible")
from acvram import kernels

ext = kernels.get_extension()
if ext is None or not hasattr(ext, "banc_fma"):
    raise SystemExit("REFUS : le binaire ne porte pas banc_fma")
print(f"# binaire sha {kernels._SO_HASH}")

props = torch.cuda.get_device_properties(0)
SM = props.multi_processor_count
K = int(sys.argv[1]) if len(sys.argv) > 1 else 64      # travail par bloc, fixe
THREADS = 128
print(f"# {props.name} · {SM} SM · K={K} FMA chainees par bloc · "
      f"{THREADS} fils par bloc")

def temps(nblocs, n=31):
    appel = lambda: ext.banc_fma(nblocs, 1, 1, THREADS, K)
    for _ in range(10):
        appel()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); appel(); e1.record(); torch.cuda.synchronize()
        t.append(e0.elapsed_time(e1) * 1000.0)
    t.sort()
    return t[len(t) // 2], t[3], t[-4]

# Balayage GROSSIER d'abord : ou sont les coudes ?
print("# --- balayage large ---")
larges = [SM, 2 * SM, 4 * SM, 6 * SM, 8 * SM, 12 * SM, 16 * SM, 24 * SM,
          32 * SM, 48 * SM]
for nb in larges:
    m, a, b = temps(nb)
    print(f"{nb:6d} blocs ({nb/SM:5.1f} par SM) · {m:8.2f} us [{a:.2f}-{b:.2f}]",
          flush=True)

# Balayage FIN autour de chaque multiple de SM : une marche s'y verrait.
print("# --- balayage fin autour des multiples de SM ---")
for base in (SM, 2 * SM, 4 * SM):
    for d in (-2, -1, 0, 1, 2):
        nb = base + d
        if nb < 1:
            continue
        m, a, b = temps(nb)
        print(f"{nb:6d} blocs (= {base//SM}xSM {d:+d}) · {m:8.2f} us "
              f"[{a:.2f}-{b:.2f}]", flush=True)
