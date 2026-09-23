"""Pièce 62 (23/09) : l aligneur CUDA porté de vLLM (`marlin_port.aligner_blocs_cuda`) rend EXACTEMENT les mêmes
sorties que l aligneur Triton capturable du 18/09 (`aligner_blocs_capturable`) — sorted_ids (paires triées par
expert, sentinelle G), expert_ids par bloc sur les blocs utiles, num_post — sur des routages tirés au sort à
godets 2/8/16, avec des experts à 0, 1 et 12 paires. Carte requise ; skip sans port compilé."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _mp():
    from acvram.kernels import marlin_port as MP
    ops = MP.charger(compiler=False)
    if ops is None or not hasattr(ops, "moe_align_block_size"):
        pytest.skip("port Marlin sans moe_align_block_size (recompiler : outils/banc-marlin-p1-18-09.py --compiler-seulement)")
    return MP


@pytest.mark.parametrize("t", [2, 8, 16])
def test_aligneur_cuda_egal_a_triton(t):
    MP = _mp()
    E, k = 128, 8
    dev = torch.device("cuda", 0)
    g = torch.Generator().manual_seed(62 + t)
    p = 1.0 / (torch.arange(E, dtype=torch.float32) + 1) ** 1.7
    eid = torch.stack([torch.multinomial(p, k, replacement=False, generator=g) for _ in range(t)]).reshape(-1).to(torch.int32)
    eid[: k] = 7                                             # un expert à ≥ 8 paires (jusqu à 12 quand t ≥ 2)
    eid = eid.to(dev)
    bloc = MP.choisir_block_size(t, k, E)
    s1, e1, n1 = MP.aligner_blocs_capturable(eid, bloc, E)
    s2, e2, n2 = MP.aligner_blocs_cuda(eid, bloc, E)
    assert int(n1) == int(n2), (int(n1), int(n2))
    n = int(n1)
    assert torch.equal(e1[: n // bloc], e2[: n // bloc]), "experts par bloc différents"
    # Pièce 63 : la place d une paire dans les blocs de SON expert est libre (vLLM : atomiques ; un expert sur
    # deux blocs se répartit différemment d un appel à l autre) — l invariant lu par la GEMM : paire → expert.
    def expert_de_chaque_paire(s, e):
        pos = torch.nonzero(s[:n] < eid.numel()).flatten()
        m = torch.full((eid.numel(),), -2, dtype=torch.int32, device=dev)
        m[s[pos]] = e[pos // bloc]
        return m
    assert torch.equal(expert_de_chaque_paire(s1, e1), expert_de_chaque_paire(s2, e2)), "paire → expert différent"
    assert torch.equal(expert_de_chaque_paire(s2, e2), eid), "paire → expert ≠ eid"
    assert bool((s1[n:] == eid.numel()).all()) and bool((s2[n:] == eid.numel()).all()), "sentinelle G attendue après num_post"


def test_aligneur_cuda_sur_tampons_fixes_capturable():
    MP = _mp()
    E, k, t = 128, 8, 16
    dev = torch.device("cuda", 0)
    eid = (torch.arange(t * k, device=dev, dtype=torch.int32) * 7) % E
    bloc = MP.choisir_block_size(t, k, E)
    P = -(-(t * k + E * (bloc - 1)) // bloc) * bloc
    tampons = (torch.empty(P, dtype=torch.int32, device=dev), torch.empty(P // bloc, dtype=torch.int32, device=dev), torch.empty(1, dtype=torch.int32, device=dev))
    s, e, n = MP.aligner_blocs_cuda(eid, bloc, E, tampons)
    assert s.data_ptr() == tampons[0].data_ptr() and e.data_ptr() == tampons[1].data_ptr() and n.data_ptr() == tampons[2].data_ptr()
    gph = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        MP.aligner_blocs_cuda(eid, bloc, E, tampons)
    torch.cuda.current_stream().wait_stream(st)
    with torch.cuda.graph(gph):
        MP.aligner_blocs_cuda(eid, bloc, E, tampons)
    ref = (tampons[0].clone(), tampons[1].clone(), int(tampons[2]))
    tampons[0].zero_(); tampons[1].zero_(); tampons[2].zero_()
    gph.replay(); torch.cuda.synchronize()
    assert torch.equal(tampons[0], ref[0]) and torch.equal(tampons[1], ref[1]) and int(tampons[2]) == ref[2]
