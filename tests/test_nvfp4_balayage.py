"""ScaleSweep (poste6, 30/09, ordre chef ; arXiv 2606.07618 « ScaleSweep », Lin & Wan) : échelles de bloc nvfp4
« balayage » (MSE) et « balayage-w » (WMSE), HORS défaut (max6 reste le défaut, 4sur6 inchangé). Par bloc de 16, les
motifs E4M3 de s_base−3 (−8 en WMSE) à s_base+7 sont essayés, le moindre (W)MSE gagne, égalité → s_base.
À sec : (1) balayage == une référence en boucle explicite, bloc par bloc, au bit (e4m3, codes) — y compris blocs nuls,
échelles sous-normales, motif proche de 448 ; (2) jamais pire que max6 NI que 4sur6 en MSE par bloc (le candidat amax/4
de 4sur6 est dans la fenêtre +7) ; (3) balayage-w jamais pire que max6 en WMSE par bloc, et exige `importance` ;
(4) `sous_normales_e4m3` compte les motifs 0x01…0x07 ; (5) max6 et 4sur6 rendent exactement ce qu'ils rendaient
(stats en plus, sans « balayes »).
Cassure : remplacer le corps de `_balayer` par « rend s_base » → (1) et (2) rouges."""
import pytest
import torch

from acvram.quant.nvfp4 import (E2M1_LEVELS, E4M3_MAX, quantize_nvfp4, dequantize_nvfp4, round_to_e2m1,
                                sous_normales_e4m3, _BALAYAGE_BAS, _BALAYAGE_HAUT, _BALAYAGE_BAS_W, _E4M3_CODE_MAX)

NIV = torch.tensor(E2M1_LEVELS)


def _reference_bloc(b, g, bas, haut, imp=None):
    """Un bloc [16] fp32, échelle globale g : (e4m3, codes 3 bits) du meilleur motif, boucle explicite ; égalité → s_base."""
    base = (b.abs().max() / 6.0 / g).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    cb = int(base.view(torch.uint8))

    def essai(code):
        e4 = torch.tensor(code, dtype=torch.uint8).view(torch.float8_e4m3fn)
        eff = float(e4.to(torch.float32)) * g
        c = round_to_e2m1(b / eff) if eff > 0 else torch.zeros(16, dtype=torch.uint8)
        err = (b.abs() - NIV[c.long()] * eff) ** 2
        return e4, c, float((err * imp).sum() if imp is not None else err.sum())
    meilleur = essai(cb)
    if cb == 0:
        return meilleur[0], meilleur[1]
    for d in range(-bas, haut + 1):
        if d == 0:
            continue
        cand = essai(min(max(cb + d, 1), _E4M3_CODE_MAX))
        if cand[2] < meilleur[2]:
            meilleur = cand
    return meilleur[0], meilleur[1]


def _blocs_varies(n=64):
    torch.manual_seed(7)
    w = torch.randn(n, 16 * 8) * (torch.rand(n, 1) * 4 + 0.05)
    w[0, :16] = 0                                   # bloc nul
    w[1, :16] = 2e-3 * torch.randn(16)              # échelle sous-normale (g ≈ 0,074 : amax/6/g ≈ 0,005 < 2⁻⁶)
    w[2, :16] = 200.0 * torch.randn(16).sign()      # bloc qui fixe g (motif 0x7E)
    w[3, 16:32] = torch.tensor([3.0] + [0.01] * 15)   # un max isolé : s_base bien trop grande
    return w


def _mse_par_bloc(w, t, imp=None):
    d = dequantize_nvfp4(t, torch.float32)
    err = (w.to(torch.float32) - d) ** 2
    if imp is not None:
        err = err * imp
    return err.view(w.shape[0], -1, 16).sum(-1)


def test_balayage_egal_reference_au_bit():
    w = _blocs_varies()
    t = quantize_nvfp4(w, echelle="balayage")
    g = float(t.global_scale)
    codes = t.qweight  # empaqueté ; on compare via dequantisation + e4m3
    wb = w.view(w.shape[0], -1, 16)
    for i in range(w.shape[0]):
        for j in range(wb.shape[1]):
            e4, c = _reference_bloc(wb[i, j], g, _BALAYAGE_BAS, _BALAYAGE_HAUT)
            assert int(t.block_scale[i, j].view(torch.uint8)) == int(e4.view(torch.uint8)), (i, j)
    d = dequantize_nvfp4(t, torch.float32)
    # les codes : reconstruction identique à celle de la référence, bloc par bloc
    for i in range(w.shape[0]):
        for j in range(wb.shape[1]):
            e4, c = _reference_bloc(wb[i, j], g, _BALAYAGE_BAS, _BALAYAGE_HAUT)
            eff = float(e4.to(torch.float32)) * g
            attendu = NIV[c.long()] * eff * wb[i, j].sign()
            attendu = torch.where(wb[i, j] == 0, torch.zeros_like(attendu), attendu)
            assert torch.equal(d.view(w.shape[0], -1, 16)[i, j], attendu), (i, j)
    assert t.echelle_stats["balayes"] > 0 and t.echelle_stats["echelle"] == "balayage"


def test_balayage_jamais_pire_que_max6_ni_4sur6():
    w = _blocs_varies(256)
    m6, s46, bal = (quantize_nvfp4(w, echelle=e) for e in ("max6", "4sur6", "balayage"))
    e6, e46, eb = (_mse_par_bloc(w, t) for t in (m6, s46, bal))
    assert bool((eb <= e6 + 1e-9).all()) and bool((eb <= e46 + 1e-9).all())
    assert float(eb.sum()) < float(e46.sum()) < float(e6.sum())      # strictement mieux au total sur ces blocs


def test_balayage_w_wmse_jamais_pire_et_exige_importance():
    w = _blocs_varies(128)
    imp = torch.rand(w.shape[1]) * 5 + 0.01
    with pytest.raises(ValueError):
        quantize_nvfp4(w, echelle="balayage-w")
    m6 = quantize_nvfp4(w, echelle="max6")
    bw = quantize_nvfp4(w, echelle="balayage-w", importance=imp)
    assert bool((_mse_par_bloc(w, bw, imp) <= _mse_par_bloc(w, m6, imp) + 1e-9).all())
    g = float(bw.global_scale); wb = w.view(w.shape[0], -1, 16); impb = imp.view(-1, 16)
    for i in range(0, w.shape[0], 16):
        for j in range(wb.shape[1]):
            e4, _ = _reference_bloc(wb[i, j], g, _BALAYAGE_BAS_W, _BALAYAGE_HAUT, impb[j])
            assert int(bw.block_scale[i, j].view(torch.uint8)) == int(e4.view(torch.uint8)), (i, j)


def test_sous_normales_e4m3():
    codes = torch.tensor([0x00, 0x01, 0x07, 0x08, 0x38, 0x7E], dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert sous_normales_e4m3(codes) == 2
    w = _blocs_varies()
    assert quantize_nvfp4(w, echelle="max6").echelle_stats["sous_normales"] >= 1     # le bloc 2e-3


def test_max6_et_4sur6_inchanges():
    w = _blocs_varies(128)
    for e in ("max6", "4sur6"):
        t = quantize_nvfp4(w, echelle=e)
        assert "balayes" not in t.echelle_stats and t.echelle_stats["echelle"] == e
    # max6 == formule d'origine (témoin recopié de test_nvfp4_4sur6)
    wb = w.view(w.shape[0], -1, 16); g = wb.abs().amax() / (6.0 * E4M3_MAX)
    bs = ((wb.abs().amax(-1) / 6.0) / g).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    assert torch.equal(quantize_nvfp4(w, echelle="max6").block_scale.view(torch.uint8), bs.view(torch.uint8))
