"""poste5-menus 29/09 : Gemma-4-26B-A4B (moe_intermediate_size 704 ≡ 64 mod 128) mourait au démarrage — « GEMM groupee
MMA : K multiple de ks » (acvram_kernels.cu) : la porte du chemin MMA (moe.py) n'exige que K % 64, le noyau K % ks avec
ks = 128 par défaut. Repli ks = 64 pour ces K ; un K multiple de 128 garde 128 (sorties inchangées)."""
import inspect
import re

from acvram.engine import moe


def test_ks_suit_la_profondeur():
    assert moe._MOE_MMA_KS == 128
    assert moe.ks_mma(704) == 64                       # Gemma-4-26B-A4B, projection descendante
    assert moe.ks_mma(2816) == 128 and moe.ks_mma(2048) == 128 and moe.ks_mma(768) == 128
    for k in range(64, 8192, 64):
        assert k % moe.ks_mma(k) == 0, k


def test_les_appels_du_noyau_passent_par_ks_mma():
    src = inspect.getsource(moe)
    appels = re.findall(r"nvfp4_gemm_grouped_mma\((.*?)\)\n", src, flags=re.S)
    assert len(appels) == 2, appels
    for a in appels:
        assert "ks_mma(k)" in a and "_MOE_MMA_KS" not in a, a
