"""C14-b, geste (3) (revue/chantier-c14b-19-09 § Fait le 20/09) : `mla_prep_batch` regrillé — la
tuile k_b[h][r0..r0+64] chargée une fois en shared par uint4 coalescés, les B créneaux découpés en
groupes de MLAP_BG=4 : grille nh·NR·⌈B/4⌉ + B = 492 blocs à b=12 (172 avant : un bloc par SM, chaque
fil lisant ses 32 bf16 un par un, 22 µs par couche pour 2,6 Mo), 172 inchangés à b=1. Même
arithmétique par (b, r) : même fil (r, quart), même ordre de somme sur n, mêmes deux shuffles ; le
noyau d'avant reste sous `temoin=True` (`mla_prep_batch_temoin_kernel`) et le test carte compare
les deux AU BIT (torch.equal) à B ∈ {1, 5, 12}, avec et sans RoPE ; à sec, la source porte la grille.
"""
import os
import re

import pytest
import torch

NH, NOPE, ROPE, RANK = 20, 128, 64, 512
W = RANK + ROPE
DT = torch.bfloat16
CU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "acvram", "kernels", "acvram_kernels.cu")


def test_source_cu_porte_la_grille_et_le_temoin():
    """`mla_prep_batch` prend `temoin` (False) ; la grille est nh·NR·GB + B avec GB = ⌈B/MLAP_BG⌉,
    MLAP_BG = 4 ; le témoin garde nh·NR + B ; les deux noyaux gardent le fil (r, quart), l'ordre de
    somme sur n et les deux shuffles."""
    src = open(CU, encoding="utf-8").read()
    assert re.search(r"constexpr int MLAP_FILS = 256, MLAP_LIGNES = 64, MLAP_BG = 4;", src)
    prep = src.split("std::vector<torch::Tensor> mla_prep_batch(")[1].split("\n}\n")[0]
    assert "const int GB = (B + MLAP_BG - 1) / MLAP_BG;" in prep
    assert "mla_prep_batch_kernel<<<(unsigned)(nh * NR * GB + B), MLAP_FILS, shm, stream>>>" in prep
    assert "mla_prep_batch_temoin_kernel<<<(unsigned)(nh * NR + B), MLAP_FILS, shm, stream>>>" in prep
    assert 'py::arg("eps"), py::arg("temoin") = false' in src
    # le noyau regrillé garde le fil (r, quart) et les deux shuffles du témoin
    for nom in ("mla_prep_batch_kernel(", "mla_prep_batch_temoin_kernel("):
        corps = src.split("__global__ void __launch_bounds__(MLAP_FILS) " + nom)[1].split("\n}\n")[0]
        assert "const int r = tid >> 2, quart = tid & 3;" in corps
        assert "acc += __shfl_xor_sync(0xffffffffu, acc, 1);" in corps
        assert "acc += __shfl_xor_sync(0xffffffffu, acc, 2);" in corps
        assert "for (int n = quart; n < nope; n += 4)" in corps


def _ext_carte():
    if not torch.cuda.is_available():
        pytest.skip("carte requise")
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "mla_prep_batch"):
        pytest.skip("extension sans mla_prep_batch")
    return ext


@pytest.mark.parametrize("B", [1, 5, 12])
@pytest.mark.parametrize("rope", [True, False])
def test_prep_regrille_au_bit_contre_temoin_sur_carte(B, rope):
    """`mla_prep_batch` (tuile k_b en shared, groupes de 4 créneaux) = `temoin=True` (grille
    d'avant) AU BIT, q_eff et k_new, à B=1 (un groupe), 5 (groupe partiel) et 12 (trois groupes)."""
    ext = _ext_carte()
    torch.manual_seed(7 + B)
    q = (torch.randn(B, NH, NOPE + ROPE, device="cuda") * 0.5).to(DT).contiguous()
    kvp = (torch.randn(B, W, device="cuda") * 0.3).to(DT).contiguous()
    lens = torch.randint(0, 2000, (B,), device="cuda", dtype=torch.int64)
    k_b = (torch.randn(NH, RANK, NOPE, device="cuda") * 0.05).to(DT).contiguous()
    w_norm = (torch.rand(RANK, device="cuda") + 0.5).to(DT).contiguous()
    if rope:
        ang = torch.arange(2048, device="cuda", dtype=torch.float32)[:, None] * (0.01 * torch.arange(ROPE // 2, device="cuda")[None, :])
        cos32, sin32 = torch.cos(ang).to(DT).float().contiguous(), torch.sin(ang).to(DT).float().contiguous()
    else:
        cos32 = sin32 = None
    a = ext.mla_prep_batch(q, kvp, lens, cos32, sin32, k_b, w_norm, NOPE, ROPE, RANK, 1e-5)
    t = ext.mla_prep_batch(q, kvp, lens, cos32, sin32, k_b, w_norm, NOPE, ROPE, RANK, 1e-5, True)
    assert torch.equal(a[0], t[0]), f"q_eff diverge : {(a[0] - t[0]).abs().max().item():.3e}"
    assert torch.equal(a[1], t[1]), "k_new diverge"
    # témoin cassant : le témoin n'est pas une copie du nouveau — un q d'un créneau modifié change ce créneau seul
    q2 = q.clone(); q2[B - 1] += 1
    a2 = ext.mla_prep_batch(q2, kvp, lens, cos32, sin32, k_b, w_norm, NOPE, ROPE, RANK, 1e-5)
    assert not torch.equal(a2[0][B - 1], a[0][B - 1]) and torch.equal(a2[0][:B - 1], a[0][:B - 1])
