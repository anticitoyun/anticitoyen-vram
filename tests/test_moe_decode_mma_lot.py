"""Garde de lot du chemin MMA au décodage (poste7, revue/poste7-mma-lot-15-09.md) :
sous ACVRAM_MOE_DECODE_MMA_MIN_T (défaut 6, provisoire), le GEMV ; au-dessus,
la MMA. poste3, 15/09 : à b=1 la MMA fait 7,00 ms/pas contre 4,48 (−36 % de
débit, +14 % de J). Le test lit le seuil de l'env et casse si la garde
disparaît (t=1 → GEMV, t=12 → MMA)."""
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
for t in (1, seuil - 1, seuil, 12):
    M.MoEBlock.forward(b, X(t))
print(seuil, ",".join(appels))
"""


def _lance(env_sup):
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_MOE_DECODE_MMA_MIN_T"}
    env.update(env_sup)
    r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip()


def test_garde_de_lot_defaut_6():
    assert _lance({}) == "6 gemv,gemv,mma,mma"


def test_garde_lit_le_seuil():
    assert _lance({"ACVRAM_MOE_DECODE_MMA_MIN_T": "2"}) == "2 gemv,gemv,mma,mma"
    assert _lance({"ACVRAM_MOE_DECODE_MMA_MIN_T": "1"}).split(" ")[1].split(",")[0] == "mma"   # t=1 en MMA
