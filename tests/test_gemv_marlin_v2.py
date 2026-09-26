"""Pièce 130 (24/09) : GEMV sur disposition Marlin v2 (`nvfp4_gemv_marlin2`, E = 1, M = 1).
* AU BIT de v1 (`nvfp4_gemv_marlin`) à split-K égal, pour tpb = 1, 2, 4 : même ordre d'accumulation par colonne.
* Juge 2⁻⁷·max par ligne contre le GEMV naturel (`nvfp4_matmul`, disposition naturelle) — l'ordre fp32 diffère, le
  bit est impossible par construction (v1 non plus, acvram_kernels.cu : commentaire du GEMV Marlin).
* K = 17 408 (> 11 264, borne de v1) en UN lancement ; reproductible au bit ; défaut hors du chemin."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def test_v1_pose_sous_processus():
    """Pièce 156 : v2 est le défaut (test_defaut_marlin_156) ; ACVRAM_GEMV_MARLIN_V2=0 rend v1."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_GEMV_MARLIN")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ACVRAM_GEMV_MARLIN_V2"] = "0"
    r = subprocess.run([sys.executable, "-c", "import acvram.kernels as k; print(k._GEMV_MARLIN_V2)"], env=env,
                       capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert r.stdout.strip().splitlines()[-1] == "False", r.stdout + r.stderr


def _poids(n, k, graine):
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import quantize_nvfp4
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin2") or MP.charger(compiler=False) is None:
        pytest.skip("extension sans nvfp4_gemv_marlin2 ou port Marlin absent")
    g = torch.Generator(device="cuda").manual_seed(graine)
    t = quantize_nvfp4(torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02)
    w, s, gs = MP.preparer_pile(t.qweight[None], t.block_scale[None], t.global_scale.reshape(1).float())
    x = torch.randn(1, k, device="cuda", generator=g, dtype=torch.bfloat16)
    return kernels, ext, t, w.contiguous(), s.contiguous(), gs.contiguous(), x


@carte
@pytest.mark.parametrize("n,k", [(5120, 6144), (2048, 5120), (12288, 5120)])
@pytest.mark.parametrize("tpb", [1, 2, 4])
def test_v2_au_bit_de_v1_a_split_egal(n, k, tpb):
    kernels, ext, t, w, s, gs, x = _poids(n, k, 130 + n % 97)
    S = int(ext.nvfp4_gemv_marlin_splitk(k, n, 1))
    z = torch.zeros(1, dtype=torch.int32, device="cuda")
    y1 = ext.nvfp4_gemv_marlin(w, s, gs, z, z, x, k, n)
    y2 = ext.nvfp4_gemv_marlin2(w, s, gs, x, k, n, tpb, S)
    assert torch.equal(y1, y2), f"v2 ≠ v1 au bit (S = {S}, tpb = {tpb}) : {int((y1 != y2).sum())} colonnes"


@carte
@pytest.mark.parametrize("n,k", [(5120, 17408), (34816, 5120), (5120, 6144)])
@pytest.mark.parametrize("tpb,S", [(2, 0), (4, 0), (1, 1), (4, 2)])
def test_v2_contre_naturel_et_reproductible(n, k, tpb, S):
    kernels, ext, t, w, s, gs, x = _poids(n, k, 7 + n % 89)
    ref = kernels.nvfp4_matmul(x, t).float()
    y = ext.nvfp4_gemv_marlin2(w, s, gs, x, k, n, tpb, S)
    hors = int(((y - ref).abs() > 2 ** -7 * ref.abs().amax(-1, keepdim=True)).sum())
    assert hors == 0, f"{hors} colonnes hors 2⁻⁷·max contre le GEMV naturel"
    assert torch.equal(y, ext.nvfp4_gemv_marlin2(w, s, gs, x, k, n, tpb, S)), "v2 non reproductible"


def test_tpb_par_forme_choix(monkeypatch):
    """142 famille 24B : TPB = 0 (défaut) → 2 si N ≥ 49 152 (gate‖up 65 536 : +15,8 % à TPB 1, +1,5 % à TPB 2), sinon 1
    (Qwen3.8, N ≤ 34 816 : TPB 1 le meilleur, 130) ; un TPB explicite prime."""
    from acvram import kernels
    monkeypatch.setattr(kernels, "_GEMV_MARLIN_TPB", 0)
    assert [kernels._tpb_marlin(n) for n in (65536, 49152, 49088, 34816, 5120)] == [2, 2, 1, 1, 1]
    monkeypatch.setattr(kernels, "_GEMV_MARLIN_TPB", 4)
    assert kernels._tpb_marlin(65536) == 4 and kernels._tpb_marlin(5120 + 64) == 1        # 81 tuiles : 4 ne divise pas


@carte
@pytest.mark.parametrize("n,k", [(65536, 5120), (49152, 5120), (34816, 5120), (5120, 32768), (5120, 4096)])
def test_tpb_par_forme_au_bit_de_tpb1(n, k):
    """Le TPB choisi par forme, à S = 0 (automatique, le défaut servi), rend la sortie AU BIT de TPB 1 : à N ≥ 49 152 les
    deux lancent ≥ 384 blocs (MB_BLOCS_MIN), S = 1 des deux côtés ; en dessous, TPB 1 est choisi. Cassant (seuil abaissé,
    TPB 2 aux petites N : S différent) → rouge sur (5 120, 32 768) et (5 120, 4 096)."""
    kernels, ext, t, w, s, gs, x = _poids(n, k, 142 + n % 89)
    tpb = kernels._tpb_marlin(n) if kernels._GEMV_MARLIN_TPB == 0 else pytest.skip("ACVRAM_GEMV_MARLIN_TPB posé")
    y1 = ext.nvfp4_gemv_marlin2(w, s, gs, x, k, n, 1, 0)
    yt = ext.nvfp4_gemv_marlin2(w, s, gs, x, k, n, tpb, 0)
    assert torch.equal(y1, yt), f"TPB {tpb} ≠ TPB 1 au bit : {int((y1 != yt).sum())} colonnes (n = {n})"
