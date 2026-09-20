"""Port Marlin (acvram/kernels/marlin_port), parties Python à sec : id du
type kFE2M1f (= vLLM), alignement des jetons par blocs d'expert (contrat de
moe_align_block_size), traitement des échelles S0E5M3 (formes, zéro sous 2,
monotone), permutation des échelles (bijection). L'extension CUDA se
compile et se juge sur carte (outils/banc-marlin-p1-18-09.py)."""
import pytest

pytestmark = pytest.mark.a_sec          # Triton interprété : carte visible → ignoré (T4 20/09)
import torch

from acvram.kernels import marlin_port as MP


def test_l_id_de_kfe2m1f_est_celui_de_vllm():
    assert MP._kfe2m1f_id() == 562949953487106      # vllm.scalar_type.scalar_types.float4_e2m1f.id


def test_aligner_blocs_suit_le_contrat_de_moe_align_block_size():
    ids = torch.tensor([[2, 3, 0], [1, 2, 0], [1, 3, 0], [1, 2, 3]])
    s, e, n = MP.aligner_blocs(ids, 4, 4)
    assert s.tolist() == [2, 5, 8, 12, 3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12]
    assert e.tolist() == [0, 1, 2, 3] and n.tolist() == [16]
    # expert sans jeton : aucun bloc ; expert à 5 jetons sur bloc 4 : deux blocs, 3 sentinelles
    ids = torch.tensor([[1, 1, 1, 1, 1, 3]])
    s, e, n = MP.aligner_blocs(ids, 4, 4)
    assert e.tolist() == [1, 1, 3] and n.tolist() == [12]
    assert s.tolist()[:8] == [0, 1, 2, 3, 4, 6, 6, 6] and s.tolist()[8:] == [5, 6, 6, 6]
    assert MP.choisir_block_size(2048, 8, 128) == 64 and MP.choisir_block_size(12, 8, 128) == 8


def test_echelles_s0e5m3_formes_et_seuil():
    torch.manual_seed(0)
    K, N = 256, 128
    s = (torch.rand(K // 16, N) * 4).to(torch.bfloat16)          # [K/16, N] ≥ 0
    p = MP.permuter_echelles(s, K, N, 16)
    assert p.shape == (K // 16, N) and torch.equal(p.flatten().sort().values, s.flatten().sort().values)
    f = MP.facteur_nvfp4(p)
    assert f >= 1.0 and f.is_integer() and (int(f) & (int(f) - 1)) == 0      # puissance de 2
    t = MP.traiter_echelles_nvfp4(p, f)
    assert t.dtype == torch.float8_e4m3fn and t.shape == (K // 16, N)
    g = MP.traiter_echelle_globale(torch.tensor([1.5]), f)
    assert g.item() == 1.5 * 2.0 ** (126 - 7) / f
    # une échelle nulle reste nulle, une échelle sous 2/(f·2⁷) tombe à zéro
    s0 = torch.zeros(K // 16, N, dtype=torch.bfloat16)
    assert int(MP.traiter_echelles_nvfp4(s0, 1.0).view(torch.uint8).sum()) == 0


@pytest.mark.parametrize("b,k,E,bloc", [(12, 8, 128, 8), (1, 8, 128, 8), (4, 3, 4, 4), (12, 8, 128, 16)])
def test_l_aligneur_capturable_egale_aligner_blocs(b, k, E, bloc):
    """Un lancement Triton, sorties de taille fixe (graphe) = la version torch."""
    if not (torch.cuda.is_available() or __import__("os").environ.get("TRITON_INTERPRET") == "1"):
        pytest.skip("Triton")
    g = torch.Generator().manual_seed(b + k)
    topk = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(b)])
    s1, e1, n1 = MP.aligner_blocs(topk, bloc, E)
    tampons = MP.aligner_blocs_capturable(topk.reshape(-1).to(torch.int32), bloc, E)
    s2, e2, n2 = MP.aligner_blocs_capturable(topk.reshape(-1).to(torch.int32), bloc, E, tampons)   # rejoué sur les mêmes tampons
    P = int(n1)
    assert int(n2) == P and torch.equal(s1[:P], s2[:P]) and torch.equal(e1[:P // bloc], e2[:P // bloc])
    assert bool((s2[P:] == b * k).all())                          # au-delà : sentinelle
