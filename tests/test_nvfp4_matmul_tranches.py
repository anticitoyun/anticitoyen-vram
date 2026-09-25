"""Repli GEMM d'`nvfp4_matmul` (n > seuil GEMV, mode bf16 par défaut) : la
déquantification se fait par tranches de lignes quand la matrice entière
dépasserait `_DEQUANT_TRANCHE_MAX` — pièce 153 (poste4, 25/09), tête d'un
modèle à vocabulaire étendu 248 320 × 5 120 : 2,49 + 4,74 Gio d'un coup au
premier appel PPL (tranches de 256 lignes), OOM sur un modèle déjà serré en
marge (mêmes symptômes que Gemma-4-31B côté int8, `test_int8_matmul_tranches.py`,
jamais porté ici).

Contrairement au repli int8 (chemin CPU dans ses tests, déterministe,
`torch.equal`), ce chemin passe par cuBLAS : le tenseur n'est déplacé sur GPU
que si une carte est visible (`CUDA`, comme les autres tests CUDA du dépôt) —
sans carte, `nvfp4_matmul` prend le repli CPU et ne traverse jamais ce code,
ces tests seraient alors un faux positif silencieux (skip explicite au lieu).
Sur GPU, découper un GEMM 40×1000 en huit GEMM 40×≤128 change l'ordre de
réduction K choisi par cuBLAS, donc le dernier bit, à arithmétique et valeurs
identiques par ailleurs (mesuré : écart absolu max 1,53e-5 sur des logits
~O(1-10), `scratchpad/poste4-p153-25-09/test-tranches-cuda.py`) — négligeable
devant tout seuil KL du protocole (0,5-1,4), mais PAS « au bit » au sens
strict de REGLES §1 : à trancher par chef si ce chemin reste au défaut tel
quel ou passe en écart nommé."""
import pytest
import torch

from acvram import kernels
from acvram.quant.nvfp4 import quantize_nvfp4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")


def _cas():
    torch.manual_seed(7)
    w = torch.randn(1000, 256)                       # 1000 lignes : pas multiple des tranches
    t = quantize_nvfp4(w)
    t.qweight, t.block_scale, t.global_scale = (
        t.qweight.cuda(), t.block_scale.cuda(), t.global_scale.cuda())
    x = torch.randn(40, 256, dtype=torch.float32, device="cuda")  # x fp32 : hors chemin GEMV (bf16 requis)
    return t, x


@CUDA
def test_les_tranches_rendent_la_meme_sortie_a_une_tolerance_gemm_pres(monkeypatch):
    t, x = _cas()
    entier = kernels.nvfp4_matmul(x, t)
    par_ligne = t.padded_in * (4 + torch.bfloat16.itemsize)
    monkeypatch.setattr(kernels, "_DEQUANT_TRANCHE_MAX", 128 * par_ligne)    # tranches de 128 lignes
    appels = []
    orig = kernels.nvfp4_dequant
    monkeypatch.setattr(kernels, "nvfp4_dequant", lambda tt, dt: appels.append(tt.qweight.shape[0]) or orig(tt, dt))
    tranche = kernels.nvfp4_matmul(x, t)
    assert appels == [128] * 7 + [104], appels
    assert tranche.shape == (40, 1000)
    assert torch.allclose(tranche, entier, atol=1e-4, rtol=0)  # mesure 1,53e-5, casse au-dela (ordre chef)


@CUDA
def test_une_tranche_sautee_se_voit(monkeypatch):
    t, x = _cas()
    entier = kernels.nvfp4_matmul(x, t)
    par_ligne = t.padded_in * (4 + torch.bfloat16.itemsize)
    monkeypatch.setattr(kernels, "_DEQUANT_TRANCHE_MAX", 128 * par_ligne)
    orig = kernels.nvfp4_dequant

    def sabote(tt, dt):
        w = orig(tt, dt)
        return torch.zeros_like(w) if tt.qweight.shape[0] == 104 else w      # la dernière tranche perdue
    monkeypatch.setattr(kernels, "nvfp4_dequant", sabote)
    assert not torch.allclose(kernels.nvfp4_matmul(x, t), entier, atol=1e-4, rtol=0)


@CUDA
def test_sous_le_seuil_pas_de_tranchage_inutile(monkeypatch):
    """`_DEQUANT_TRANCHE_MAX` par défaut (256 Mio) couvre déjà ce petit cas :
    le chemin entier reste pris, sans régression sur les poids ordinaires."""
    t, x = _cas()
    appels = []
    orig = kernels.nvfp4_dequant
    monkeypatch.setattr(kernels, "nvfp4_dequant", lambda tt, dt: appels.append(tt.qweight.shape[0]) or orig(tt, dt))
    kernels.nvfp4_matmul(x, t)
    assert appels == [1000]
