"""Pièce 43 (`banc-etroites-noyaux.py`) à sec : octets comptés par format,
verdicts par seuils écrits avant (réfuté ≤ 1,0 To/s, alarme > 2,0, gain
qualifié « pas au bit » pour int8 canal et fp8), et la contrainte M > 16 de
cuBLASLt nommée. Les noyaux sont carte (poste2)."""
import importlib.util
import os


def _m():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-noyaux.py")
    spec = importlib.util.spec_from_file_location("banc_etroites_noyaux", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_octets_par_format():
    m = _m()
    # qkv [5120, 2048] : int8 par groupe de 128 = poids + scales fp16 + zeros u8
    assert m.octets_int8(5120, 2048) == 5120 * 2048 + (5120 * 2048 // 128) * 3
    assert abs(m.octets_int8(5120, 2048) / 1e6 - 10.73) < 0.01
    # par canal (int8 cuBLASLt, fp8) : poids + une échelle par ligne
    assert m.octets_canal(5120, 2048) == 5120 * 2048 + 5120 * 4
    assert abs(m.octets_canal(2048, 4096) / 1e6 - 8.40) < 0.01
    assert m.M == 12 and m.M_CUBLASLT == 32            # cuBLASLt refuse M <= 16 : rembourrage nommé


def test_verdicts():
    m = _m()
    def r(us_b, us_c=None, base=10.37, to_b=1.2, to_c=1.6):
        l = {"a_triton_int8_groupe": {"us": base, "to_s": 0.81},
             "b_int_mm_int8_canal": {"us": us_b, "to_s": to_b, "ecart_rel_max": 0.004}}
        if us_c is not None:
            l["c_scaled_mm_fp8"] = {"us": us_c, "to_s": to_c, "ecart_rel_max": 0.02}
        return {"formes": {"qkv": {"lignes": l}}}
    assert m.verdict(r(8.0, 6.0))["verdict"].startswith("TENU")
    assert "pas au bit" in m.verdict(r(8.0, 6.0))["verdict"]
    assert m.verdict(r(12.0, 13.0, to_b=0.7, to_c=0.6))["verdict"].startswith("RÉFUTÉ : aucun autre noyau")
    assert m.verdict(r(8.0, 6.0, to_c=2.4))["verdict"].startswith("ALARME")
    sans = {"formes": {"qkv": {"lignes": {"a_triton_int8_groupe": {"us": 10.0, "to_s": 0.81},
                                          "b_int_mm_int8_canal": {"erreur": "RuntimeError: …"}}}}}
    assert m.verdict(sans)["verdict"].startswith("RÉFUTÉ : aucun autre noyau")
