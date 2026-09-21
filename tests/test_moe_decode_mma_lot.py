"""Garde de lot du chemin MMA au décodage (Sage, revue/sage-mma-lot-15-09.md) :
sous ACVRAM_MOE_DECODE_MMA_MIN_T (défaut 5 : godets 8, 12 et 16 en MMA), le
GEMV ; au-dessus, la MMA. Courbe de Laure, 15/09, MMA/GEMV : b=1 +56 % ms,
b=2 +36, b=3 +27, b=4 +24, b=6 +18, b=12 −6,4 % ms / −13,2 % J. Le test lit
le seuil de l'env et casse si la garde disparaît (t=1, 4 → GEMV, 8, 12 → MMA)."""
import os
import subprocess
import sys

CODE = """
import os, torch
import acvram.engine.model as M
seuil = M._MOE_DECODE_MMA_MIN_T
appels = []
class Faux(M.MoEBlock):
    def __init__(self): pass
    def _forward_grouped_mma(self, x, topw, topi): appels.append("mma"); return torch.zeros(1)
    def _forward_grouped(self, x, topw, topi): appels.append("gemv"); return torch.zeros(1)
    def _forward_prefill_grouped(self, x, topw, topi): appels.append("prefill"); return torch.zeros(1)
    def _route(self, x): return torch.zeros(x.shape[0], 8), torch.zeros(x.shape[0], 8, dtype=torch.long)
    def _compter_routage(self, topi): pass
    def _shared_out(self, x): return 0
b = Faux(); b._stack_state = "oui"; b.shared = None; b.top_k = 8
b._stacks = {"gate_proj": ("nvfp4",)}
class X:
    def __init__(self, t): self.shape = (t, 8); self.is_cuda = True; self.dtype = torch.bfloat16
    def __getitem__(self, i): return self
for t in (1, 4, seuil - 1, seuil, 8, 12):
    M.MoEBlock.forward(b, X(t))
print(seuil, ",".join(appels))
"""


def _lance(env_sup):
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_MOE_DECODE_MMA_MIN_T"}
    env.update(env_sup)
    r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip()


def test_garde_de_lot_defaut_5():
    """Défaut 5 (0.6.6, cellule b=5 de Laure) : t=1 et 4 (godet 4) en GEMV, 5, 8 et 12 en MMA."""
    assert _lance({}) == "5 gemv,gemv,gemv,mma,mma,mma"


def test_garde_lit_le_seuil():
    assert _lance({"ACVRAM_MOE_DECODE_MMA_MIN_T": "2"}) == "2 gemv,mma,gemv,mma,mma,mma"
    assert _lance({"ACVRAM_MOE_DECODE_MMA_MIN_T": "1"}).split(" ")[1].split(",")[0] == "mma"   # t=1 en MMA
