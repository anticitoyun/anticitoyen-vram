"""`acvram.regime` (sage-prefill-a8-verdict-17-09 § 3) : la table des
variables de chemin suit le code, l'en-tête de mesure dit ce qui tourne, et
chaque chemin sélectionné par une variable, un backend ou un noyau est
masquable — un masque qui ne masque rien est une erreur, pas un silence."""
import glob
import os
import pathlib
import re

import pytest

import acvram
from acvram import regime
from acvram.kernels import backends


def _variables_lues():
    racine = pathlib.Path(acvram.__file__).resolve().parent
    lues = set()
    for f in glob.glob(str(racine / "**" / "*.py"), recursive=True):
        if f.endswith("cli.py") or f.endswith("regime.py"):
            continue                       # la liste de garde et cette table se citent
        src = open(f, errors="ignore").read()
        lues |= set(re.findall(r'os\.environ(?:\.get\(|\[)\s*"(ACVRAM_[A-Z0-9_]+)"', src))
        lues |= set(re.findall(r'os\.getenv\(\s*"(ACVRAM_[A-Z0-9_]+)"', src))
    # l'extension aussi : `std::getenv("ACVRAM_…")` dans le .cu choisit des
    # chemins (GROUPED_RPW, GROUPED_OLD…) — absents de toute table jusqu'au
    # 18/09 (sage-gemv-experts-rpw-18-09)
    for f in glob.glob(str(racine / "kernels" / "*.cu")):
        src = open(f, errors="ignore").read()
        lues |= set(re.findall(r'getenv\(\s*"(ACVRAM_[A-Z0-9_]+)"', src))
    return lues


def test_toute_variable_de_chemin_est_dans_la_table():
    """Casse quand on ajoute un chemin gouverné par une variable sans
    l'inscrire (ou sans le déclarer hors régime) : c'est ainsi que
    ACVRAM_PREFILL=a8 a tourné un mois sans figurer dans aucun en-tête."""
    lues = _variables_lues()
    connues = {v.env for v in regime.VARIABLES} | regime.HORS_REGIME
    manquantes = lues - connues
    assert not manquantes, f"variables lues absentes de regime.VARIABLES / HORS_REGIME : {sorted(manquantes)}"
    fantomes = {v.env for v in regime.VARIABLES} - lues
    assert not fantomes, f"dans la table mais plus lues nulle part : {sorted(fantomes)}"


def test_les_defauts_de_la_table_sont_ceux_du_code():
    """La valeur effective d'une variable lue à l'import, environnement
    vierge, est son défaut de la table — sinon l'en-tête ment."""
    for v in regime.VARIABLES:
        if v.lu_a is not None and v.env not in os.environ:
            assert regime._effective(v) == v.defaut, (v.env, regime._effective(v), v.defaut)


def test_ligne_de_regime_nomme_ce_qui_differe(monkeypatch):
    monkeypatch.setenv("ACVRAM_PREFILL", "w8a8")
    ligne = acvram.regime_ligne()
    assert ligne.startswith("[régime] ") and "ACVRAM_PREFILL=w8a8" in ligne
    monkeypatch.delenv("ACVRAM_PREFILL")
    assert "ACVRAM_PREFILL" not in acvram.regime_ligne()


def test_ligne_de_regime_porte_les_trois_versions():
    """Un pip install dans le venv de mesure change l'arithmetique sans
    qu'aucun defaut ACVRAM_* ne bouge (sage-glm-etendue-canal-saillant-18-09
    § 5) : la ligne doit porter torch, triton et fla pour qu'un JSON puisse
    distinguer un noyau Triton d'une autre version."""
    ligne = acvram.regime_ligne()
    assert re.search(r"torch=\S+", ligne), ligne
    assert re.search(r"triton=\S+", ligne), ligne
    assert re.search(r"fla=\S+", ligne), ligne


def test_prefill_regime_refuse_les_anciens_noms(monkeypatch):
    from acvram import kernels
    monkeypatch.delenv("ACVRAM_PREFILL", raising=False)
    assert kernels.prefill_regime() == "bf16"
    for ancien in ("a8", "a4", "fp8"):
        monkeypatch.setenv("ACVRAM_PREFILL", ancien)
        with pytest.raises(ValueError):
            kernels.prefill_regime()


@pytest.fixture
def masques_propres():
    from acvram import kernels
    etat = (kernels._EXT, kernels._TRIED, kernels._ERROR, set(backends._MASQUES))
    backends._RESOLVED.clear()
    try:
        yield
    finally:
        kernels._EXT, kernels._TRIED, kernels._ERROR = etat[:3]
        backends._MASQUES.clear(); backends._MASQUES.update(etat[3])
        backends._RESOLVED.clear()


def test_masquer_une_variable_reecrit_le_module_deja_importe(monkeypatch, masques_propres):
    import acvram.engine.model as M
    monkeypatch.delenv("ACVRAM_MOE_MMA", raising=False)
    monkeypatch.setattr(M, "_MOE_MMA", True)
    fait = regime.masquer(["MOE_MMA"])
    assert fait == {"MOE_MMA": "ACVRAM_MOE_MMA=0"}
    assert M._MOE_MMA is False and os.environ["ACVRAM_MOE_MMA"] == "0"
    assert regime.regime_noyaux()["hors_defaut"]["ACVRAM_MOE_MMA"] == "0"


def test_masquer_un_backend_le_retire_de_la_resolution(masques_propres):
    import torch
    avant = [b.name for b in backends.resolve("nvfp4", torch.device("cpu"))]
    assert avant[-1] == "reference-cpu"
    regime.masquer(["reference-cpu"])
    apres = [b.name for b in backends.resolve("nvfp4", torch.device("cpu"))]
    assert "reference-cpu" not in apres
    assert regime.regime_noyaux()["backends_masques"] == ["reference-cpu"]


def test_masquer_refuse_un_seuil_et_un_inconnu(masques_propres):
    with pytest.raises(ValueError):
        regime.masquer(["NVFP4_GEMV_MAX"])          # un seuil n'a pas de jumeau torch
    with pytest.raises((KeyError, RuntimeError)):
        regime.masquer(["noyau_qui_n_existe_pas"])


def test_masquer_disable_kernels_apres_import_eteint_l_extension(monkeypatch, masques_propres):
    from acvram import kernels
    monkeypatch.delenv("ACVRAM_DISABLE_KERNELS", raising=False)
    regime.masquer(["DISABLE_KERNELS"])
    assert kernels.get_extension() is None and kernels._ERROR
    r = regime.regime_noyaux()
    assert r["extension"] is False and r["hors_defaut"]["ACVRAM_DISABLE_KERNELS"] == "1"


def test_hors_regime_ne_cache_aucun_regime():
    """Sage 19/09 : HORS_REGIME ne contient que ce qui observe, compile ou
    nomme un fichier — jamais une capacité, un plafond ou un mode (HYBRID_SLOTS
    y était : le régime servi de GLM, eager dès b=5, était invisible dans
    regime_ligne). Un nom hors de ces familles casse le test."""
    familles = ("TRACE_", "VERBOSE", "TRACEBACK", "MODELS_DIR", "MODELES", "PARC", "GALERIE", "FOND_",
                "CHRONO", "SYNC_", "KERNEL_CACHE", "CUDA_HOME", "ARCH_FAMILY", "GW_WARPS", "TESTS_",
                "CARTE_", "PPL_TRANCHE", "QA_", "WARM_GRAPHS", "BANC_", "SESSION", "VERROU", "DUMP_",
                "TETE_FP32_ENTREE", "MOE_DECODE_MASQUES", "DISABLE_", "TYPE", "ARBRE", "PROFIL", "LOG",
                "CHARGE_OK", "ECO_", "SERVEUR", "PORT", "CACHE_PREFIXE", "HOTE", "MUET", "MARLIN_CACHE",
                "CHAUFFE_CTX")   # opt-out de la PREUVE du contexte, visible sur la ligne (ctx_tenu=non-verifie)
    hors = []
    for nom in regime.HORS_REGIME:
        court = nom[len("ACVRAM_"):]
        if not any(f in court for f in familles):
            hors.append(nom)
    assert not hors, f"variables de régime cachées dans HORS_REGIME : {sorted(hors)}"
    for v in ("ACVRAM_HYBRID_SLOTS", "ACVRAM_DENSE_SLOTS", "ACVRAM_MTP", "ACVRAM_PIPELINE"):
        assert v in {x.env for x in regime.VARIABLES}
