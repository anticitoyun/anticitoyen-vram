"""Pièce 187 : la tranche du GEMV int8 (`ACVRAM_INT8_TRANCHE` pour N ≤ 16, `ACVRAM_INT8_TRANCHE_PREFILL` au-delà) ne
change pas la sortie : chaque sortie (r, n) garde son accumulateur et son ordre, quelle que soit la tranche. La tranche
est lue une fois par processus (acvram_kernels.cu) : un sous-processus par valeur, sorties comparées par empreinte à la
tranche 16. Poids réels du mixte (couche 0, q/k/v, porte, sortie, par canal vus en g128) si l'alias est sur disque,
sinon poids tirés. Bras cassant (prise 187) : ordre des mots inversé pour NV ≤ 12 dans le noyau → ROUGE."""
import functools
import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIAS = "Qwen3.8-27B-unsloth-mixte-i8c"
NS = (2, 5, 9, 12, 13, 16, 17, 31, 33, 64, 78, 80)

ENFANT = r'''
import hashlib, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.getcwd(), "outils"))
from acvram import kernels
from acvram.quant.formats import INT8Tensor, _quantize_int8
ext = kernels.get_extension()
dev = torch.device("cuda:0")
ts = []
chemin = sys.argv[1]
if chemin:
    from safetensors import safe_open
    with safe_open(os.path.join(chemin, "acvram-00000.safetensors"), "pt", device="cuda:0") as h:
        for nom in ("linear_attn.qkv", "linear_attn.gate", "linear_attn.out"):
            p = "model.layers.0." + nom + ".weight."
            q = h.get_tensor(p + "qweight")
            t = INT8Tensor(qweight=q, scales=h.get_tensor(p + "scales"), zeros=h.get_tensor(p + "zeros"),
                           group_size=q.shape[1], shape=tuple(q.shape))
            ts.append((nom, kernels.vue_g128(t)))
else:
    torch.manual_seed(187)
    ts.append(("tire", _quantize_int8(torch.randn(6144, 5120, device=dev) * 0.02, group_size=128)))
res = {}
for nom, t in ts:
    for n in json.loads(sys.argv[2]):
        g = torch.Generator(device="cpu").manual_seed(n)
        x = (torch.randn(n, t.qweight.shape[1], generator=g) * 0.5).to(dev, torch.bfloat16)
        for fp32 in (False, True):
            y = ext.int8_gemv(t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(), x, t.group_size, fp32)
            res[f"{nom}/{n}/{int(fp32)}"] = hashlib.sha256(y.float().cpu().numpy().tobytes()).hexdigest()[:16]
print("RES" + json.dumps(res))
'''


def _modele():
    sys.path.insert(0, os.path.join(RACINE, "outils"))
    try:
        from racine_modeles import racine_modeles
        c = os.path.join(racine_modeles(), ALIAS)
        return c if os.path.isfile(os.path.join(c, "acvram-00000.safetensors")) else ""
    except Exception:
        return ""


@functools.lru_cache(maxsize=None)
def _empreintes(dec, pre, chemin):
    env = dict(os.environ, ACVRAM_INT8_TRANCHE=str(dec), ACVRAM_INT8_TRANCHE_PREFILL=str(pre), PYTHONPATH=RACINE)
    r = subprocess.run([sys.executable, "-c", ENFANT, chemin, json.dumps(NS)], cwd=RACINE, env=env,
                       capture_output=True, text=True, timeout=600)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("RES")]
    assert r.returncode == 0 and lignes, r.stderr[-2000:]
    return json.loads(lignes[-1][3:])


@pytest.mark.parametrize("dec,pre", [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 8)])
def test_tranche_au_bit_contre_16(dec, pre):
    chemin = _modele()
    ref = _empreintes(16, 16, chemin)
    autre = _empreintes(dec, pre, chemin)
    assert len(ref) >= 2 * len(NS)
    diff = sorted(k for k in ref if ref[k] != autre[k])
    assert not diff, diff


def _tranches(**env):
    e = {k: v for k, v in os.environ.items() if k not in ("ACVRAM_INT8_TRANCHE", "ACVRAM_INT8_TRANCHE_PREFILL")}
    e.update(env, PYTHONPATH=RACINE)
    r = subprocess.run([sys.executable, "-c", "from acvram import kernels; print('TR', tuple(kernels.get_extension().int8_tranches()))"],
                       cwd=RACINE, env=e, capture_output=True, text=True, timeout=600)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("TR")]
    assert r.returncode == 0 and lignes, r.stderr[-2000:]
    return eval(lignes[-1][3:])


def test_defaut_tranche_6_et_temoins():
    """Pièce 187 : le défaut est 6/6 (casse s'il revient à 16) ; 16 reste le témoin ; le préfill suit le décodage
    sauf s'il est posé ; une valeur hors liste retombe sur le défaut."""
    assert _tranches() == (6, 6)
    assert _tranches(ACVRAM_INT8_TRANCHE="16") == (16, 16)
    assert _tranches(ACVRAM_INT8_TRANCHE="16", ACVRAM_INT8_TRANCHE_PREFILL="6") == (16, 6)
    assert _tranches(ACVRAM_INT8_TRANCHE="7") == (6, 6)
