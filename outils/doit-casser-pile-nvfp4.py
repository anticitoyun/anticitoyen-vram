"""Sabote le CHEMIN de la pile (comme le noyau de GLM-4.7 le faisait), pas la
donnee : la reference bf16 reste saine, donc le controle peut rendre faux."""
import sys, torch
sys.path.insert(0,'/tmp/poste1-acvram'); sys.path.insert(0,'/tmp/poste1-acvram/tests')
import test_fusion_nvfp4 as T
vrai_stack = T.stack_nvfp4_linears
class PileFausse:
    """L'echelle du premier bloc appliquee au second — le defaut du 6/09."""
    def __init__(s, f): s.f = f
    def __call__(s, x):
        y = s.f(x).clone()
        y[:, 768:] = y[:, 768:] / 40.0
        return y
def stack_sabote(lins):
    return PileFausse(vrai_stack(lins))
T.stack_nvfp4_linears = stack_sabote
ko = ok = 0
for n in (1,4,8,9,16,64):
    try:
        T.test_pile_juste_a_tout_nombre_de_jetons(n)
        print(f"  n={n:3d} PASSE  <- AVEUGLE"); ok+=1
    except AssertionError as e:
        print(f"  n={n:3d} rend faux : {str(e).splitlines()[0][:100]}"); ko+=1
print(f"\n{ko}/6 rendent faux, {ok} aveugles")
