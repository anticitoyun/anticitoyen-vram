"""Mode éco d'horloge (poste7-e1-eco-tenu-19-09 § 2), à sec : `acvram.eco` lit
un verrou `-lgc` dans la sortie de nvidia-smi, refuse sans droit sudo et sous
la carte d'un pair, et `regime_ligne()` porte l'horloge dès qu'une carte est
visible — sinon un chiffre éco passerait pour un chiffre défaut."""
import os
import subprocess
import time

import pytest

import acvram
from acvram import eco, regime


# ------------------------------------------------------------------ (a) lire

def test_lire_horloge_reconnait_un_verrou_lgc():
    """Le verrou vient de l'état posé (ETAT_ECO) vérifié contre clocks.sm :
    applications_clocks_setting reste « Not Active » sous -lgc (poste2, 19/09 23 h),
    le 3e champ est désormais la raison « idle »."""
    h = eco.lire_horloge(sortie="2692, 3135, Not Active\n", etat={"mode": "2700"})
    assert h["verrou"] is True and h["sm_mhz"] == 2692 and h["max_sm_mhz"] == 3135
    assert eco.etiquette_horloge(h) == "lgc2700"
    # sans état posé, une carte à 2 692 MHz SOUS CHARGE LÉGÈRE (au lieu du boost ≥ 2 900)
    # porte un -lgc posé ailleurs : dit incertain ; au repos rien n'est concluant
    assert eco.etiquette_horloge(eco.lire_horloge(sortie="2692, 3135, Not Active\n", etat={}, sous_charge=True)) == "lgc2700?"
    assert eco.etiquette_horloge(eco.lire_horloge(sortie="2692, 3135, Active\n", etat={})) == "libre"


def test_lire_horloge_reconnait_une_carte_libre():
    h = eco.lire_horloge(sortie="225, 3135, Not Active\n", etat={})
    assert h["verrou"] is False and h["sm_mhz"] == 225
    assert eco.etiquette_horloge(h) == "libre"


@pytest.mark.parametrize("sortie", ["", "\n", "[N/A], [N/A], [N/A]", "erreur : carte absente"])
def test_lire_horloge_sans_reponse_ne_sait_pas(sortie):
    h = eco.lire_horloge(sortie=sortie)
    assert h["verrou"] is None and h["erreur"]
    assert eco.etiquette_horloge(h) == "?"


def test_lire_horloge_sans_nvidia_smi_ne_leve_pas(monkeypatch):
    def absent(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(eco.subprocess, "run", absent)
    h = eco.lire_horloge(0)
    assert h["verrou"] is None and "FileNotFoundError" in h["erreur"]
    assert eco.etiquette_horloge(h) == "?"


# ---------------------------------------------------------------- (b) regler

class _Executeur:
    """Remplaçant de subprocess.run : enregistre les commandes, rend ce qu'on
    lui dit."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.appels = []
        self.reponse = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, cmd, **k):
        self.appels.append(list(cmd))
        return self.reponse


@pytest.fixture
def carte_libre(monkeypatch, tmp_path):
    """Aucun .qui : personne ne tient la carte ; la relecture d'horloge est
    simulée (aucun nvidia-smi réel) et comptée."""
    monkeypatch.setattr(eco, "VERROU_QUI", str(tmp_path / "acvram-carte-{index}.lock.qui"))
    monkeypatch.setattr(eco, "ETAT_ECO", str(tmp_path / "acvram-eco-{index}.json"))
    relectures, original = [], eco.lire_horloge

    def lire(index=0):
        relectures.append(index)
        return original(index, sortie="2692, 3135, Not Active")     # l'état posé par regler() décide
    monkeypatch.setattr(eco, "lire_horloge", lire)
    return relectures


def test_regler_2700_verrouille_par_sudo_n(carte_libre, capsys):
    ex = _Executeur()
    assert eco.regler("2700", executer=ex) == 0
    assert ex.appels == [["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "2700,2700"]]
    assert carte_libre == [0]                      # relue après le réglage
    assert "horloge=lgc2700" in capsys.readouterr().out       # état posé 2700, relu 2 692


def test_regler_off_libere_par_rgc(carte_libre):
    ex = _Executeur()
    assert eco.regler("off", index=1, executer=ex) == 0
    assert ex.appels == [["sudo", "-n", "nvidia-smi", "-i", "1", "-rgc"]]
    assert carte_libre == [1]


def test_regler_sans_droit_sudo_refuse_code_3(carte_libre, capsys):
    ex = _Executeur(returncode=1, stderr="sudo: a password is required\n")
    assert eco.regler("2700", executer=ex) == 3
    assert len(ex.appels) == 1 and carte_libre == []   # rien d'autre appelé
    err = capsys.readouterr().err
    assert "REFUS" in err and "password" in err


def test_regler_nvidia_smi_en_echec_refuse_code_3(carte_libre):
    ex = _Executeur(returncode=255, stderr="Setting locked GPU clocks is not supported\n")
    assert eco.regler("2100", executer=ex) == 3
    assert carte_libre == []


def test_regler_refuse_un_mode_inconnu(carte_libre):
    ex = _Executeur()
    assert eco.regler("2400", executer=ex) == 2 and ex.appels == []


# ------------------------------------------------------- (c) carte tenue

@pytest.fixture
def qui(monkeypatch, tmp_path):
    monkeypatch.setattr(eco, "VERROU_QUI", str(tmp_path / "acvram-carte-{index}.lock.qui"))
    lire = eco.lire_horloge                            # aucun nvidia-smi réel à la relecture
    monkeypatch.setattr(eco, "lire_horloge", lambda index=0: lire(sortie="2692, 3135, Active"))

    def ecrire(pid, index=0):
        (tmp_path / f"acvram-carte-{index}.lock.qui").write_text(
            f"{pid} {int(time.time())} pair mesure\n")
    return ecrire


def test_regler_refuse_sous_la_carte_d_un_pair_code_4(qui, monkeypatch, capsys):
    monkeypatch.delenv("ACVRAM_CARTE_TENUE", raising=False)
    qui(os.getpid())                                   # PID vivant, pas le nôtre déclaré
    ex = _Executeur()
    assert eco.regler("2700", executer=ex) == 4
    assert ex.appels == []                             # rien n'a été écrit
    err = capsys.readouterr().err
    assert f"carte tenue par PID {os.getpid()}" in err
    assert "un réglage d'horloge se fait sous le verrou outils/carte.sh (REGLES § 1)" in err


def test_regler_passe_sous_sa_propre_prise_de_carte(qui, monkeypatch):
    monkeypatch.setenv("ACVRAM_CARTE_TENUE", str(os.getpid()))
    qui(os.getpid())
    ex = _Executeur()
    assert eco.regler("2700", executer=ex) == 0 and len(ex.appels) == 1


def test_regler_ignore_un_detenteur_disparu(qui, monkeypatch):
    monkeypatch.delenv("ACVRAM_CARTE_TENUE", raising=False)
    qui(999999999)                                     # PID hors de tout processus vivant
    ex = _Executeur()
    assert eco.regler("off", executer=ex) == 0 and len(ex.appels) == 1


# ---------------------------------------------------- (d) ligne de régime

@pytest.fixture
def carte_visible(monkeypatch):
    """Une carte « visible » sans toucher au GPU : torch.cuda.is_available()
    simulé après un premier appel réel (le compte de périphériques du
    runtime CUDA est figé à sec), et l'extension n'est jamais compilée."""
    import torch
    from acvram import kernels
    torch.cuda.is_available()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(kernels, "get_extension", lambda: None)
    monkeypatch.setattr(kernels, "_ERROR", "simulé")
    from acvram import eco
    monkeypatch.setattr(eco, "horloge_du_processus", lambda: None)   # sous carte.sh l'état éco prend la place de « horloge= » (T4 20/09)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def test_la_ligne_de_regime_porte_l_horloge_verrouillee(carte_visible, monkeypatch):
    monkeypatch.setattr(regime, "_LIRE_HORLOGE",
                        lambda index=0: {"sm_mhz": 2700, "max_sm_mhz": 3135, "verrou": True, "brut": ""})
    assert "horloge=lgc2700" in acvram.regime_ligne()
    monkeypatch.setattr(regime, "_LIRE_HORLOGE",
                        lambda index=0: {"sm_mhz": 225, "max_sm_mhz": 3135, "verrou": False, "brut": ""})
    ligne = acvram.regime_ligne()
    assert "horloge=libre" in ligne and "lgc" not in ligne


def test_la_ligne_de_regime_dit_quand_elle_ne_sait_pas(carte_visible, monkeypatch):
    def casse(index=0):
        raise RuntimeError("nvidia-smi")
    monkeypatch.setattr(regime, "_LIRE_HORLOGE", casse)
    assert "horloge=?" in acvram.regime_ligne()


def test_la_ligne_de_regime_ne_change_pas_a_sec(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    appels = []

    def lire(index=0):
        appels.append(index)
        return {"sm_mhz": 2700, "max_sm_mhz": 3135, "verrou": True, "brut": ""}
    monkeypatch.setattr(regime, "_LIRE_HORLOGE", lire)
    assert "horloge=" not in acvram.regime_ligne() and appels == []


def test_index_carte_suit_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert eco.index_carte() == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert eco.index_carte() == 0
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert eco.index_carte() == 0
