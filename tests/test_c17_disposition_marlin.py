"""C17 (chantier-c17-mma2-lit-marlin-19-09) — à sec : l'arithmétique d'index du noyau
mma2<MARLIN> (marlin_port/disposition.py, mot à mot celle du .cu) reconstitue la pile
naturelle depuis la disposition Marlin : quartets ÉGAUX (100 %), échelles égales hors celles
que le repack annule (s·facteur·2⁷ < 2) ; témoins cassants : un quartet permuté, un decal faux."""
import math

import pytest
import torch

from acvram.kernels.marlin_port import disposition as D
from acvram.kernels.marlin_port import facteur_nvfp4, permuter_echelles, traiter_echelles_nvfp4, GROUP_SIZE


def _pile(N, K, seed):
    g = torch.Generator().manual_seed(seed)
    qw = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.uint8)
    # échelles E4M3 ≥ 0 : normales surtout, quelques sous-normales (annulées par le repack)
    e = torch.randint(1, 12, (N, K // 16), generator=g)
    m = torch.randint(0, 8, (N, K // 16), generator=g)
    bs = ((e << 3) | m).to(torch.uint8)
    bs[torch.rand(N, K // 16, generator=g) < 0.002] = 3                  # sous-normal E4M3 (E = 0, M = 3)
    bs[0, 0] = 0x7E                                                      # 448 : facteur = 1, les sous-normaux sont annulés
    return qw, bs


@pytest.mark.parametrize("N,K", [(128, 256), (256, 2048), (2048, 768)])
def test_quartets_reconstitues_egaux_et_echelles_hors_annulees(N, K):
    qw, bs = _pile(N, K, 7)
    w_m = D.pile_naturelle_vers_marlin(qw, K, N)
    assert tuple(w_m.shape) == (K // 16, 2 * N)
    rec = D.fragments_b_depuis_marlin(w_m, K, N)
    assert torch.equal(rec, qw), "quartets : la reconstitution diffère de la pile naturelle"
    # échelles : même chaîne que preparer_pile (permuter_echelles + traiter_echelles_nvfp4)
    scales = bs.view(torch.float8_e4m3fn).to(torch.bfloat16)               # [N, K/16]
    facteur = facteur_nvfp4(scales)
    s_m = traiter_echelles_nvfp4(permuter_echelles(scales.T, K, N, GROUP_SIZE), facteur).view(torch.uint8)
    rec_s = D.echelles_ue4m3_depuis_marlin(s_m, D.decal_echelles(facteur), K, N)
    annulees = (scales.float() * facteur * 128) < 2
    assert torch.equal(rec_s[~annulees], bs[~annulees]), "échelles non annulées : reconversion ≠ UE4M3 d'origine"
    assert (rec_s[annulees] == 0).all()                                     # celles-là restent nulles
    assert facteur == 1.0 and 0 < annulees.float().mean() < 0.01             # le cas existe (Coder : 0,0064 %) et reste rare


def test_temoins_cassants():
    N, K = 128, 256
    qw, bs = _pile(N, K, 11)
    w_m = D.pile_naturelle_vers_marlin(qw, K, N)
    # un quartet permuté dans une tuile : la reconstitution doit différer
    casse = w_m.clone(); casse[0, 0] ^= 0xF0
    assert not torch.equal(D.fragments_b_depuis_marlin(casse, K, N), qw)
    # un decal faux (facteur ×2 supposé) : les échelles normales changent d'exposant
    scales = bs.view(torch.float8_e4m3fn).to(torch.bfloat16)
    facteur = facteur_nvfp4(scales)
    s_m = traiter_echelles_nvfp4(permuter_echelles(scales.T, K, N, GROUP_SIZE), facteur).view(torch.uint8)
    faux = D.echelles_ue4m3_depuis_marlin(s_m, D.decal_echelles(facteur * 2), K, N)
    assert not torch.equal(faux[bs > 8], bs[bs > 8])
    # facteur > 1 : la sous-normale qui SURVIT au repack est reconvertie exactement
    bs2 = torch.full((64, 16), 0x20, dtype=torch.uint8); bs2[0, 0] = 0x03          # E=0, M=3 : 2^-6 · 3/8
    sc2 = bs2.view(torch.float8_e4m3fn).to(torch.bfloat16)
    f2 = facteur_nvfp4(sc2); assert f2 > 1
    s2 = traiter_echelles_nvfp4(permuter_echelles(sc2.T, 256, 64, GROUP_SIZE), f2).view(torch.uint8)
    assert torch.equal(D.echelles_ue4m3_depuis_marlin(s2, D.decal_echelles(f2), 256, 64), bs2)


def test_le_cu_porte_la_meme_arithmetique():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "acvram" / "kernels" / "acvram_kernels.cu").read_text()
    assert "template <int BT, int S, int KS, bool MARLIN = false>" in src
    for motif in ("const int pA = kb + 2 * h;", "(4 * c + j) * 16 + w * 4", "const int j = (i + g) & 3;",
                  "8 * (o & 7) + ((q & ~3) | ((q & 1) << 1) | ((q >> 1) & 1))", "gm2_ue4m3_depuis_marlin(by, decal)",
                  "(8u | m) >> (1 - E)"):
        assert motif in src, motif
