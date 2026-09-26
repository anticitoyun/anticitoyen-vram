"""Pièce 147 (poste6, 24/09) : `depaqueter_marlin(noyau="cuda")` (extension, lignes entières) rend AU BIT ce que rendent
les noyaux triton et torch, sur toutes les formes denses des modèles mesurés en 145 (gemma4 31B, Qwen3.8-27B), une pile MoE et
une vue à échelle globale par colonne (pile q/k/v de la 134). Bras cassant : un octet de code ou d'échelle changé doit se voir.
À sec : la variable est déclarée ; `noyau="cuda"` sans extension lève. Le défaut (ACVRAM_PROJ_MARLIN=0) n'appelle jamais ce chemin."""
import pytest
import torch

from acvram.kernels import marlin_port as MP

FORMES = [  # (E, K, N) — gemma4 31B : q, k/v, o, gate_up fusé, down ; Qwen3.8-27B : q, k/v, o, gate_up fusé, down ; pile MoE
    (1, 5376, 8192), (1, 5376, 4096), (1, 8192, 5376), (1, 5376, 43008), (1, 21504, 5376),
    (1, 5120, 6144), (1, 5120, 1024), (1, 6144, 5120), (1, 5120, 34816), (1, 17408, 5120),
    (8, 2048, 768),
]


def _entrees(E, K, N, graine, par_colonne=False):
    g = torch.Generator(device="cuda").manual_seed(graine)
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (E, K // 16, 2 * N), device="cuda", dtype=torch.int32, generator=g)
    s = torch.randint(0, 256, (E, K // 16, N), device="cuda", dtype=torch.uint8, generator=g)
    s[torch.rand(s.shape, device="cuda", generator=g) < 0.03] = 0                  # échelles annulées par le repack
    gm = (torch.rand(N if par_colonne else E, device="cuda", generator=g) + 0.5) * 2.0 ** 108
    return w, s, gm


def test_variable_declaree():
    from acvram.regime import VARIABLES
    v = [x for x in VARIABLES if x.nom == "DEPAQUETAGE"]
    assert len(v) == 1 and v[0].defaut == "auto" and v[0].lu_a == ("acvram.kernels.marlin_port", "_DEPAQUETAGE")


def test_cuda_sans_extension_leve(monkeypatch):
    monkeypatch.setattr(MP, "get_extension", lambda: None, raising=False)
    import acvram.kernels as K
    monkeypatch.setattr(K, "get_extension", lambda: None)
    w = torch.zeros(1, 4, 128, dtype=torch.int32); s = torch.zeros(1, 4, 64, dtype=torch.uint8); g = torch.ones(1)
    with pytest.raises(RuntimeError):
        MP.depaqueter_marlin(w, s, g, 64, 64, noyau="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
@pytest.mark.parametrize("E,K,N", FORMES)
def test_cuda_au_bit_du_triton_et_du_torch(E, K, N):
    assert MP._depaqueter_cuda_disponible(), "extension sans depaqueter_marlin_cuda"
    w, s, g = _entrees(E, K, N, E * 1000003 + K * 7 + N)
    cu = MP.depaqueter_marlin(w, s, g, K, N, noyau="cuda")
    tr = MP.depaqueter_marlin(w, s, g, K, N, noyau="triton")
    assert torch.equal(cu.view(torch.int16), tr.view(torch.int16)), f"cuda ≠ triton : {int((cu.view(torch.int16) != tr.view(torch.int16)).sum())} bf16"
    if K * N * E <= 2048 * 768 * 8 or (E == 1 and K <= 5376 and N <= 8192):
        to = MP.depaqueter_marlin(w, s, g, K, N, noyau="torch")                  # juge, fp32 transitoire borné
        assert torch.equal(cu.view(torch.int16), to.view(torch.int16)), "cuda ≠ torch"
    assert (cu.float() == 0).float().mean().item() < 0.5                          # pas un tenseur vide : du signal partout
    del cu, tr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
def test_vue_echelle_par_colonne_au_bit():
    K, N = 5376, 8192
    w, s, g = _entrees(1, K, N, 42, par_colonne=True)
    cu = MP.depaqueter_marlin(w, s, g, K, N, noyau="cuda")
    tr = MP.depaqueter_marlin(w, s, g, K, N, noyau="triton")
    assert torch.equal(cu.view(torch.int16), tr.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
def test_bras_cassant_un_octet_change_se_voit():
    """Un code ou une échelle modifiés d'un octet changent la sortie CUDA — sinon le test au bit ne contrôlerait rien."""
    K, N = 2048, 768
    w, s, g = _entrees(1, K, N, 7)
    ref = MP.depaqueter_marlin(w, s, g, K, N, noyau="cuda").clone()
    w2 = w.clone(); w2.view(torch.uint8).view(-1)[12345] ^= 0x0F
    assert not torch.equal(ref.view(torch.int16), MP.depaqueter_marlin(w2, s, g, K, N, noyau="cuda").view(torch.int16))
    s2 = s.clone(); s2.view(-1)[4321] = (int(s2.view(-1)[4321]) + 17) % 256
    assert not torch.equal(ref.view(torch.int16), MP.depaqueter_marlin(w, s2, g, K, N, noyau="cuda").view(torch.int16))
    # le juge torch voit les mêmes écarts aux mêmes places
    to = MP.depaqueter_marlin(w2, s, g, K, N, noyau="torch")
    cu2 = MP.depaqueter_marlin(w2, s, g, K, N, noyau="cuda")
    assert torch.equal(to.view(torch.int16), cu2.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
def test_vue_de_pile_sans_copie_au_bit():
    """Tranche de colonnes d'une pile q/k/v (vue non contiguë, pas de ligne 2N_total) = copie contiguë, au bit."""
    K, N = 5376, 16384
    w, s, g = _entrees(1, K, N, 99, par_colonne=True)
    d, n_lig = 8192, 4096
    wv, sv, gv = w[0][:, 2 * d:2 * (d + n_lig)], s[0][:, d:d + n_lig], g[d:d + n_lig].contiguous()   # vues 2-D comme _marlin_parent
    assert not wv.is_contiguous()
    cu = MP.depaqueter_marlin(wv, sv, gv, K, n_lig, noyau="cuda")
    ref = MP.depaqueter_marlin(wv.contiguous(), sv.contiguous(), gv, K, n_lig, noyau="triton")
    assert torch.equal(cu.view(torch.int16), ref.view(torch.int16))


def test_le_defaut_n_appelle_pas_le_depaquetage(monkeypatch):
    """PROJ_MARLIN=0 : `_marlin_seul` n'est jamais atteint — depaqueter_marlin ne doit pas être appelé par nvfp4_matmul."""
    import acvram.kernels as K
    appels = []
    monkeypatch.setattr(MP, "depaqueter_marlin", lambda *a, **k: appels.append(1))
    monkeypatch.setattr(K, "_PROJ_MARLIN", False, raising=False)
    from acvram.quant.formats import quantize
    t = quantize(torch.randn(256, 128) * 0.05, "nvfp4")
    x = torch.randn(64, 128).to(torch.bfloat16)
    K.nvfp4_matmul(x, t)
    assert not appels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
def test_pile_par_colonne_refusee_au_cuda_et_auto_evite_cuda():
    """Pièce 250 : une pile E > 1 à g [E, N] (209, facteur par ligne d'expert) n'est pas servie par l'extension CUDA
    (elle lirait g[e] comme scalaire, sans erreur) — refus explicite en ``noyau="cuda"``, et ``auto`` passe par un
    autre noyau, au bit du juge torch. Cassant : sans la garde, ``cuda`` rendrait une sortie fausse silencieuse."""
    E, K, N = 4, 1024, 512
    w, s, _ = _entrees(E, K, N, 11)
    g = (torch.rand(E, N, device="cuda") + 0.5) * 2.0 ** 108
    with pytest.raises(ValueError, match="E > 1"):
        MP.depaqueter_marlin(w, s, g, K, N, noyau="cuda")
    avant = dict(MP.DEPAQUETAGES)
    au = MP.depaqueter_marlin(w, s, g, K, N, noyau="auto")
    assert MP.DEPAQUETAGES["cuda"] == avant.get("cuda", 0), "auto a pris le noyau CUDA sur une pile à g [E, N]"
    to = MP.depaqueter_marlin(w, s, g, K, N, noyau="torch")
    assert torch.equal(au.view(torch.int16), to.view(torch.int16))
