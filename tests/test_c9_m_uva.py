"""C9 M-UVA (`outils/gpu/mesure/c9-m-uva.py`) à sec sur un faux expert :
octets lus par pas (qweight packé + échelles), tables d adresses = data_ptr
des tampons (jamais une copie), verdicts par seuils écrits avant (vaut ≥ 17,
arrêt < 13, invalide si inexact ou témoin VRAM lent). Le GEMV lui-même est
carte (Manon)."""
import importlib.util
import os

import torch

ICI = os.path.dirname(os.path.abspath(__file__))


def _m():
    p = os.path.join(ICI, "..", "outils", "gpu", "mesure", "c9-m-uva.py")
    spec = importlib.util.spec_from_file_location("c9_m_uva", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_octets_et_tables_sur_faux_expert():
    from acvram.quant.nvfp4 import quantize_nvfp4
    m = _m()
    t = quantize_nvfp4(torch.randn(64, 256) * 0.1)
    assert m.octets_expert(t) == 64 * 128 + 64 * 16                 # E2M1 packé + E4M3 par bloc de 16
    tq, tb, gs = m.tables_de([t, t], torch.device("cpu"))
    assert tq.dtype == torch.int64 and tb.dtype == torch.int64 and gs.dtype == torch.float32
    assert tq.tolist() == [t.qweight.data_ptr()] * 2 and tb.tolist() == [t.block_scale.data_ptr()] * 2
    assert gs.tolist() == [float(t.global_scale)] * 2
    # l expert réel du 119B : w1 [2048, 4096] → 4,72 Mo, et 4 experts = 18,9 Mo par pas
    assert 2048 * 2048 + 2048 * 256 == 4_718_592


def test_verdicts_suivent_les_seuils():
    m = _m()
    assert m.verdict(18.0, 1500.0, True)["verdict"].startswith("TENU, vaut")
    assert m.verdict(15.0, 1500.0, True)["verdict"].startswith("TENU, marginal")
    assert m.verdict(12.0, 1500.0, True)["verdict"].startswith("RÉFUTÉ — arrêt")
    assert m.verdict(18.0, 1500.0, False)["verdict"].startswith("INVALIDE : sorties")
    assert m.verdict(18.0, 900.0, True)["verdict"].startswith("INVALIDE : témoin")
    v = m.verdict(18.0, 1500.0, True)
    assert v["seuil_vaut"] == 17.0 and v["seuil_arret"] == 13.0 and v["predit_go_s"] == [15.0, 19.0]
