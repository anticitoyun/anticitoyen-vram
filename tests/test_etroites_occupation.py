"""Pièce 35 : forme du noyau étroit (warps, étages) — la sortie est AU BIT
quelle que soit la forme (ni l ordre des sommes en K ni celui des tranches ne
changent) ; le crochet split-K (réfuté au banc b08a3d34) est retiré ;
occupation calculée depuis la décision du compilateur (REGLES § 3)."""
import importlib.util
import os

import pytest
import torch

from acvram.kernels import gemm_etroit as GE


@pytest.fixture(autouse=True)
def _defaut(monkeypatch):
    monkeypatch.delenv("ACVRAM_ETROITES_FORME", raising=False)
    monkeypatch.setattr(GE, "_FORME", None)
    monkeypatch.setattr(GE, "_programmes", lambda device: 170)


def test_forme_par_defaut_et_variable(monkeypatch):
    assert GE.forme_noyau() == (GE._WARPS, GE._STAGES) == (4, 3) and GE.etroites_texte() == "serie"
    monkeypatch.setenv("ACVRAM_ETROITES_FORME", "8,2")
    assert GE.forme_noyau() == (8, 2) and GE.etroites_texte() == "w8s2"
    GE.regler_forme((2, 3))
    assert GE.forme_noyau() == (2, 3) and GE.etroites_texte() == "w2s3"
    GE.regler_forme(None)
    assert GE.forme_noyau() == (8, 2)


def test_crochet_splitk_retire():
    assert not hasattr(GE, "facteur_splitk") and not hasattr(GE, "FACTEUR_DEFAUT")
    from acvram import regime
    noms = {v.env for v in regime.VARIABLES}
    assert "ACVRAM_ETROITES_SPLITK" not in noms and "ACVRAM_ETROITES_FORME" in noms
    # le découpage de K revient à l arithmétique du 17/09 (2 programmes par SM visés)
    assert GE.decouper_k(16, 80, torch.device("cpu")) == (4, 4) and GE.decouper_k(32, 32, torch.device("cpu")) == (11, 3)


def test_occupation_depuis_la_decision_du_compilateur():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-occupation.py")
    spec = importlib.util.spec_from_file_location("banc_etroites_occupation", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # 40 registres × 128 fils = 5 120 → 12 blocs ; partagée 32 Kio → 7 blocs : le min décide
    o = m.occupation(registres=40, shared=32 * 1024, warps=4)
    assert o == {"blocs_par_sm": 7, "warps_actifs": 28, "par_registres": 12, "par_shared": 7}
    assert m.occupation(registres=255, shared=0, warps=8)["blocs_par_sm"] == 1        # registres saturés
    assert m.occupation(registres=0, shared=0, warps=4)["blocs_par_sm"] == 32         # inconnu : borne haute
    r = m.lire_ptxas("ptxas info : Used 47 registers, 8192 bytes smem, 12 bytes spill stores, 8 bytes spill loads")
    assert r == {"registres": 47, "shared": 8192, "spill_stores": 12, "spill_loads": 8}


def test_lecture_des_registres_triton_38_et_refus_du_repli_muet():
    """Manon 82b070fc : `k_.cache` n existe plus en Triton 3.8 et un
    `getattr(…, {})` rendait 0 registre → 32 blocs/SM faux partout."""
    from types import SimpleNamespace as S
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-occupation.py")
    spec = importlib.util.spec_from_file_location("banc_occ3", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    k = S(device_caches={0: ({"sig": S(n_regs=139, n_spills=0, metadata=S(shared=8192))},)})
    assert m.infos_noyau(k, S(index=0)) == {"registres": 139, "spills": 0, "shared": 8192}
    # 139 registres × 128 fils : 3 blocs/SM, 12 warps actifs — borné registres (lecture réelle de Manon)
    assert m.occupation(139, 8192, 4) == {"blocs_par_sm": 3, "warps_actifs": 12, "par_registres": 3, "par_shared": 28}
    assert m.occupation(139, 8192, 2)["blocs_par_sm"] == 7 and m.occupation(139, 8192, 8)["blocs_par_sm"] == 1
    with pytest.raises(RuntimeError, match="registres lus à 0"):
        m.infos_noyau(S(device_caches={0: ({"sig": S(n_regs=0, n_spills=0, metadata=S(shared=0))},)}), S(index=0))
    with pytest.raises(RuntimeError, match="aucun binaire"):
        m.infos_noyau(S(device_caches={0: ({},)}), S(index=0))
    with pytest.raises(RuntimeError, match="device_caches"):
        m.infos_noyau(S(), S(index=0))


def test_verdict_du_banc_exige_le_bit():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-occupation.py")
    spec = importlib.util.spec_from_file_location("banc_etroites_occupation2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    base = {"formes": {"o": {"lignes": {"4w3s": {"us": 11.8, "au_bit": True, "to_s": 0.71},
                                        "8w3s": {"us": 10.0, "au_bit": True, "to_s": 0.84, "ecart_max": 0.0}}}}}
    assert m.verdict(base)["verdict"].startswith("TENU")
    faux = {"formes": {"o": {"lignes": {"4w3s": {"us": 11.8, "au_bit": True},
                                        "8w3s": {"us": 10.0, "au_bit": False, "ecart_max": 0.004, "to_s": 0.84}}}}}
    assert m.verdict(faux)["verdict"].startswith("INVALIDE")
    petit = {"formes": {"o": {"lignes": {"4w3s": {"us": 11.8, "au_bit": True},
                                         "8w3s": {"us": 11.5, "au_bit": True, "ecart_max": 0.0, "to_s": 0.73}}}}}
    assert m.verdict(petit)["verdict"].startswith("RÉFUTÉ")
