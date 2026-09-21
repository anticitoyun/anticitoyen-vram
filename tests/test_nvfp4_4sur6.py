"""Q1 (chef 21/09, arXiv 2512.02010 « Four Over Six ») : échelle de bloc nvfp4 adaptative — par bloc de 16, deux
candidats (amax/6, amax/4) arrondis en E4M3, le moindre MSE gagne, égalité → max6. Format et noyaux inchangés.
À sec : (1) max6 == l ancienne formule au bit (aucun alias du parc ne change) ; (2) 4sur6 == une référence PyTorch
indépendante, bloc par bloc, au bit (qweight, block_scale) ; (3) un bloc construit où amax/4 bat amax/6 ; (4) jamais pire
que max6 en MSE, par bloc, sur des blocs aléatoires ; (5) cas limites : zéros, bloc constant, échelles E4M3 subnormales,
signes ; (6) sur carte : noyau nvfp4_dequant == référence au bit sur un tenseur 4sur6 (sauté sans carte, à jouer par poste2)."""
import pytest, torch
from acvram.quant import nvfp4
from acvram.quant.nvfp4 import (E2M1_LEVELS, E2M1_MAX, E4M3_MAX, quantize_nvfp4, dequantize_nvfp4, round_to_e2m1,
                                unpack_e2m1, regler_echelle, echelle_courante)

NIV = torch.tensor(E2M1_LEVELS)


def _ancienne_max6(w):
    """L ancienne formule de quantize_nvfp4 (avant Q1), recopiée : témoin de non-régression pour max6."""
    wb = w.to(torch.float32).view(w.shape[0], -1, 16)
    g = wb.abs().amax() / (E2M1_MAX * E4M3_MAX)
    bs_e4m3 = ((wb.abs().amax(dim=-1) / E2M1_MAX) / g).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    bs = bs_e4m3.to(torch.float32) * g
    codes = round_to_e2m1(wb / bs.clamp(min=torch.finfo(torch.float32).tiny).unsqueeze(-1))
    codes = torch.where(bs.unsqueeze(-1) > 0, codes, torch.zeros_like(codes)) | ((wb < 0).to(torch.uint8) << 3)
    return codes.reshape(w.shape), bs_e4m3


def _reference_4sur6_bloc(b, g):
    """Un bloc [16] fp32, échelle globale g : (code 3 bits, e4m3, mse) du meilleur candidat, boucle explicite."""
    meilleur = None
    for div in (6.0, 4.0):
        e4 = torch.tensor(float(b.abs().max()) / div / g).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        eff = float(e4.to(torch.float32)) * g
        if eff > 0:
            code = round_to_e2m1(b / eff)
        else:
            code = torch.zeros(16, dtype=torch.uint8)
        mse = float(((b.abs() - NIV[code.long()] * eff) ** 2).sum())
        if meilleur is None or mse < meilleur[2]:          # strict : égalité → max6 (premier candidat)
            meilleur = (code, e4, mse)
    return meilleur


def _mse_blocs(w, t):
    d = dequantize_nvfp4(t, torch.float32)
    return ((w.to(torch.float32) - d) ** 2).view(w.shape[0], -1, 16).sum(-1)


@pytest.fixture(autouse=True)
def _max6_par_defaut():
    regler_echelle("max6"); yield; regler_echelle("max6")


def test_max6_inchange_au_bit():
    torch.manual_seed(0)
    w = torch.randn(8, 64) * torch.tensor([0.1, 1.0, 3.0, 0.02, 1.0, 1.0, 5.0, 0.5]).unsqueeze(1)
    t = quantize_nvfp4(w)                                   # défaut du module = max6
    codes, bs = _ancienne_max6(w)
    assert torch.equal(unpack_e2m1(t.qweight), codes) and torch.equal(t.block_scale.view(torch.uint8), bs.view(torch.uint8))
    assert echelle_courante() == "max6"


def test_4sur6_egale_la_reference_bloc_par_bloc_au_bit():
    torch.manual_seed(1)
    w = torch.randn(16, 128) * torch.rand(16, 1) * 4
    w[3] *= -1                                              # signes
    t = quantize_nvfp4(w, echelle="4sur6")
    g = float(t.global_scale)
    codes = unpack_e2m1(t.qweight).view(16, -1, 16)
    e4 = t.block_scale.view(torch.uint8)
    n_4 = 0
    for i in range(16):
        for j in range(8):
            b = w[i, j * 16:(j + 1) * 16].to(torch.float32)
            code, ref_e4, _ = _reference_4sur6_bloc(b, g)
            assert torch.equal(codes[i, j] & 7, code) and int(e4[i, j]) == int(ref_e4.view(torch.uint8)), (i, j)
            assert torch.equal((codes[i, j] >> 3).bool(), b < 0)
            n_4 += int(ref_e4.view(torch.uint8)) != int(t.block_scale.view(torch.uint8)[i, j]) or 0
    # au moins un bloc a choisi amax/4 sur ces données (sinon le test ne prouve rien du chemin 4sur6)
    t6 = quantize_nvfp4(w, echelle="max6")
    assert not torch.equal(t.block_scale.view(torch.uint8), t6.block_scale.view(torch.uint8)), "aucun bloc n a pris amax/4"


def test_un_bloc_ou_amax_sur_4_bat_amax_sur_6():
    # [6, 5, 5, …] : sous amax/6, les 5 tombent entre les niveaux 4 et 6 (erreur 1) ; sous amax/4, 5 → 3,33 → 3 (erreur 0,5)
    b = torch.tensor([6.0] + [5.0] * 15)
    w = torch.stack([b, torch.full((16,), 6.0)])
    # g explicite tel que amax/4 tienne encore en E4M3 (336 < 448) : avec le g automatique (amax/6 = 448), le bloc
    # qui fixe g a ses deux candidats clampés à la même valeur — 4sur6 = max6 pour lui, par construction (documenté)
    g = 2.0 / E4M3_MAX
    t6 = quantize_nvfp4(w, global_scale=torch.tensor(g), echelle="max6")
    t4 = quantize_nvfp4(w, global_scale=torch.tensor(g), echelle="4sur6")
    m6, m4 = _mse_blocs(w, t6)[0, 0], _mse_blocs(w, t4)[0, 0]
    assert m4 < m6, (float(m4), float(m6))
    # échelle effective ≈ 1,5 (336 n est pas représentable en E4M3 : 320 → 1,4286) contre 1,0 exact
    assert float(t4.block_scale[0, 0].to(torch.float32)) * g == pytest.approx(1.5, rel=0.07) and float(t6.block_scale[0, 0].to(torch.float32)) * g == pytest.approx(1.0)
    # et avec le g automatique, ce bloc fixe g : les deux règles coïncident (clamp)
    ta6, ta4 = quantize_nvfp4(w, echelle="max6"), quantize_nvfp4(w, echelle="4sur6")
    assert torch.equal(ta6.block_scale.view(torch.uint8)[0], ta4.block_scale.view(torch.uint8)[0])


@pytest.mark.parametrize("graine", [0, 1, 2, 3])
def test_jamais_pire_que_max6_par_bloc(graine):
    torch.manual_seed(graine)
    w = torch.randn(32, 256) * (torch.rand(32, 1) * 3 + 0.01)
    w[torch.rand_like(w) < 0.02] *= 20                      # aberrants
    m6 = _mse_blocs(w, quantize_nvfp4(w, echelle="max6"))
    m4 = _mse_blocs(w, quantize_nvfp4(w, echelle="4sur6"))
    assert (m4 <= m6 + 1e-6 * m6.clamp(min=1e-30)).all(), (m4 - m6).max()
    assert (m4 < m6).any(), "aucun bloc strictement meilleur : 4sur6 ne fait rien"


def test_cas_limites_zeros_constant_subnormal_signes():
    # zéros : aucun NaN, codes 0, échelle 0 (les deux règles)
    z = torch.zeros(2, 32)
    for e in ("max6", "4sur6"):
        t = quantize_nvfp4(z, echelle=e)
        assert torch.equal(dequantize_nvfp4(t, torch.float32), z) and not torch.isnan(t.block_scale.to(torch.float32)).any()
    # bloc constant : max6 exact (tout → 6) ; 4sur6 : amax/4 exact aussi (tout → 4) → égalité → max6 conservé
    c = torch.full((1, 16), 0.7)
    t6, t4 = quantize_nvfp4(c, echelle="max6"), quantize_nvfp4(c, echelle="4sur6")
    assert torch.equal(t6.block_scale.view(torch.uint8), t4.block_scale.view(torch.uint8)) and torch.equal(t6.qweight, t4.qweight)
    assert torch.allclose(dequantize_nvfp4(t4, torch.float32), c, rtol=0.07)
    # subnormaux E4M3 : un bloc énorme fixe g, un bloc minuscule tombe sous 2^-6 relatif → échelle subnormale ; fini, jamais pire
    w = torch.zeros(1, 32); w[0, :16] = torch.tensor([6.0 * 448] + [100.0] * 15); w[0, 16:] = torch.randn(16) * 0.02
    for e in ("max6", "4sur6"):
        t = quantize_nvfp4(w, echelle=e)
        assert torch.isfinite(dequantize_nvfp4(t, torch.float32)).all()
        assert float(t.block_scale.to(torch.float32)[0, 1]) < 2 ** -6 and float(t.block_scale.to(torch.float32)[0, 1]) > 0
    assert (_mse_blocs(w, quantize_nvfp4(w, echelle="4sur6")) <= _mse_blocs(w, quantize_nvfp4(w, echelle="max6")) * (1 + 1e-6)).all()
    # signes : −w donne exactement −dequant(w), mêmes échelles
    torch.manual_seed(4)
    w = torch.randn(4, 64)
    tp, tn = quantize_nvfp4(w, echelle="4sur6"), quantize_nvfp4(-w, echelle="4sur6")
    assert torch.equal(dequantize_nvfp4(tp, torch.float32), -dequantize_nvfp4(tn, torch.float32))
    assert torch.equal(tp.block_scale.view(torch.uint8), tn.block_scale.view(torch.uint8))


def test_regler_echelle_refuse_l_inconnu():
    with pytest.raises(ValueError):
        regler_echelle("max5")
    with pytest.raises(ValueError):
        quantize_nvfp4(torch.randn(1, 16), echelle="4sur5")


CARTE = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise : noyau nvfp4_dequant (poste2)")


@CARTE
def test_noyau_dequant_egal_reference_au_bit_sur_4sur6():
    from acvram.kernels import nvfp4_dequant, get_extension
    if get_extension() is None:
        pytest.skip("extension CUDA absente")
    torch.manual_seed(5)
    w = torch.randn(64, 512) * (torch.rand(64, 1) * 3 + 0.01)
    t = quantize_nvfp4(w, echelle="4sur6").to("cuda")
    for dtype in (torch.bfloat16, torch.float32):
        assert torch.equal(nvfp4_dequant(t, dtype), dequantize_nvfp4(t, dtype)), dtype


def test_conversion_porte_l_echelle_dans_le_manifeste(tiny_checkpoint, target_rig, tmp_path):
    """`--echelle=4sur6` → ConversionOptions.echelle_nvfp4 → règle du module pendant la conversion → manifeste."""
    import json, os
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    assert echelle_courante() == "max6"
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=str(tmp_path), echelle_nvfp4="4sur6"), spec=spec)
    assert echelle_courante() == "4sur6", "la règle n a pas été posée sur le module pendant la conversion"
    m = json.load(open(os.path.join(tmp_path, "acvram_manifest.json")))
    assert m["options"]["echelle_nvfp4"] == "4sur6"
    # part des blocs amax/4 et clampés : au manifeste (total + par tenseur), écrite avant toute mesure
    tot = m["echelle_nvfp4"]
    assert tot["regle"] == "4sur6" and tot["blocs"] > 0 and 0 < tot["part_amax4"] <= 1 and 0 <= tot["part_clampes"] < 1
    par_t = [e["echelle"] for e in m["tensors"].values() if e.get("format") == "nvfp4" and "echelle" in e]
    assert par_t and sum(e["blocs"] for e in par_t) == tot["blocs"]
    # contrôle indépendant : outils/part-amax4.py relit les codes stockés — même part, blocs égaux
    import subprocess, sys
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outils", "part-amax4.py"), str(tmp_path)],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.split("RESULTAT ", 1)[1])
    assert res["ecart_manifeste"] == {"blocs_amax4": 0, "blocs_clampes": 0, "blocs": 0} and res["part_autres"] == 0.0, res
    # et la conversion par défaut la remet à max6 (un alias converti ensuite ne l hérite pas) ; l outil y lit 0 amax/4
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=str(tmp_path / "b")), spec=spec)
    assert echelle_courante() == "max6"
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outils", "part-amax4.py"), str(tmp_path / "b")],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    res = json.loads(r.stdout.split("RESULTAT ", 1)[1])
    assert res["part_amax4"] == 0.0 and res["part_max6"] + res["part_nuls"] + res["part_clampes"] == pytest.approx(1.0, abs=1e-3), res
