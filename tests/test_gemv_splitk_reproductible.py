"""Pièce 67 (23/09) : le GEMV Marlin par lot en split-K (`ACVRAM_GEMV_SPLITK`, mb_splitk : réduction du dernier
bloc) est REPRODUCTIBLE AU BIT d un appel à l autre, à S auto (=1 → gate·up S=4, down S=2) comme à S=8 forcé, sur
la forme servie à b=1 (8 paires, K 2048, N 768 / 2048) ; et à ≤ 2⁻⁷·max|y| par ligne du noyau S=1 (autre ordre de
somme, pas au bit). `mb_splitk` lit la variable UNE fois par processus : chaque S tourne dans un sous-processus."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
RACINE = Path(__file__).resolve().parents[1]
SONDE = r'''
import json, sys, torch
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
from test_moe_tensor_decodage import _charger, _piles, E, K, I
_, MP, ext, banc = _charger(); dev = torch.device("cuda", 0); marlin = _piles(MP, banc, dev)
g = torch.Generator(dev).manual_seed(67)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev, generator=g)
eid = torch.tensor([7, 0, 3, 5, 12, 33, 64, 127], dtype=torch.int32, device=dev); tok = torch.zeros(8, dtype=torch.int32, device=dev)
mg, mu, md = marlin["gate_proj"], marlin["up_proj"], marlin["down_proj"]
def passe():
    y = ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok, x, mg[3], mg[4], 0)
    d = ext.nvfp4_gemv_marlin(md[0], md[1], md[2], eid, torch.arange(8, dtype=torch.int32, device=dev), y.to(torch.bfloat16).contiguous(), md[3], md[4])
    return y.float(), d.float()
ref = passe(); ok = all(torch.equal(a, b) for _ in range(20) for a, b in zip(ref, passe()))
torch.save({"y": ref[0].cpu(), "d": ref[1].cpu()}, sys.argv[1])
print(json.dumps({"S": __import__("os").environ.get("ACVRAM_GEMV_SPLITK", "0"), "reproductible_20x": ok}))
'''


def _sonde(S, sortie):
    env = dict(os.environ, ACVRAM_GEMV_SPLITK=str(S), PYTHONPATH=str(RACINE))
    out = subprocess.run([sys.executable, "-c", SONDE, str(sortie)], env=env, capture_output=True, text=True, timeout=600, cwd=RACINE)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1]), torch.load(sortie)


@pytest.mark.parametrize("S", [1, 4, 8])                          # 1 = S auto servi (gate·up 4, down 2), 4 et 8 forcés
def test_splitk_reproductible_au_bit_et_pres_du_noyau_serie(S, tmp_path):
    r0, y0 = _sonde(0, tmp_path / "s0.pt")          # noyau d avant (S=1 partout)
    rS, yS = _sonde(S, tmp_path / f"s{S}.pt")
    assert r0["reproductible_20x"] and rS["reproductible_20x"], (r0, rS)
    for k in ("y", "d"):
        borne = 2.0 ** -7 * y0[k].abs().amax(1, keepdim=True)
        hors = int(((y0[k] - yS[k]).abs() > borne).sum())
        assert hors == 0, f"{k} : {hors} valeurs hors 2⁻⁷·max contre S=1"
        assert torch.isfinite(yS[k]).all()
