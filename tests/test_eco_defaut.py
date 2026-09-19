"""Éco d'horloge par défaut (poste7-eco-2700-defaut-19-09 § 1-3), à sec avec un faux
nvidia-smi : pose au chargement, rendu à `Engine.fermer`, rendu sur signal, refus
sudo = état nommé, config.json, lecture de l'horloge par l'état posé + clocks.sm
(le champ applications_clocks_setting ne dit rien sous -lgc : poste2, 19/09 23 h)."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

from acvram import eco

RACINE = Path(__file__).resolve().parents[1]


class _Faux:
    """Enregistre les commandes ; `rc` par sous-commande (-lgc / -rgc)."""
    def __init__(self, rc=None):
        self.appels, self.rc = [], rc or {}

    def __call__(self, cmd, **kw):
        self.appels.append(list(cmd))
        cle = next((c for c in cmd if c in ("-lgc", "-rgc")), "?")
        rc = self.rc.get(cle, 0)
        return types.SimpleNamespace(returncode=rc, stdout="" if rc == 0 else "",
                                     stderr="" if rc == 0 else "sudo: a password is required")


_LIRE_ORIG = eco.lire_horloge


def _lire_fixe(sm):
    return lambda index=0, **kw: _LIRE_ORIG(index, f"{sm}, 3135, Not Active", etat={"mode": "2700"} if sm > 1000 else {})


@pytest.fixture
def isole(tmp_path, monkeypatch):
    monkeypatch.setattr(eco, "ETAT_ECO", str(tmp_path / "eco-{index}.json"))
    monkeypatch.setattr(eco, "CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setattr(eco, "_HORLOGE", None)
    monkeypatch.delenv("ACVRAM_ECO", raising=False)
    return tmp_path


def test_config_et_mode_demande(isole, monkeypatch):
    assert eco.mode_demande({}) == "2700"                              # ni fichier ni variable : 2700
    chemin = eco.ecrire_config({"eco": "2100"})
    assert json.load(open(chemin))["eco"] == "2100" and eco.mode_demande({}) == "2100"
    eco.ecrire_config({"eco": "off"}); assert eco.mode_demande({}) == "off"
    assert eco.mode_demande({"ACVRAM_ECO": "2700"}) == "2700"           # la variable prime (bras A/B)
    eco.ecrire_config({"eco": "1234"}); assert eco.mode_demande({}) == "2700"   # inconnu → défaut, dit
    (isole / "config.json").write_text("{pas du json")
    assert eco.charger_config() == {} and eco.mode_demande({}) == "2700"


def test_lecture_par_etat_pose_et_clocks_sm(isole):
    # sous -lgc 2700 le pilote lit 2 692 et applications_clocks_setting reste Not Active
    h = eco.lire_horloge(0, "2692, 3135, Not Active", etat={"mode": "2700"})
    assert h["verrou"] is True and h["mode"] == "2700" and eco.etiquette_horloge(h) == "lgc2700"
    # état posé mais horloge ailleurs : pas tenu, dit
    h = eco.lire_horloge(0, "3000, 3135, Not Active", etat={"mode": "2700"})
    assert h["verrou"] is False and "≠" in eco.etiquette_horloge(h)
    # sans état : libre ; carte oisive à haute horloge = verrou posé ailleurs, incertain
    assert eco.etiquette_horloge(eco.lire_horloge(0, "225, 3135, Active", etat={})) == "libre"
    assert eco.etiquette_horloge(eco.lire_horloge(0, "2692, 3135, Active", etat={})) == "lgc2692?"
    # l'état posé vient du fichier quand `etat` n'est pas donné
    eco._ecrire_etat(0, "2700")
    assert eco.lire_horloge(0, "2692, 3135, Not Active")["verrou"] is True
    eco._ecrire_etat(0, None)
    assert eco.lire_horloge(0, "2692, 3135, Not Active")["verrou"] is False
    assert eco.lire_horloge(0, "n'importe quoi")["verrou"] is None


def test_pose_rendu_idempotent_et_etat_fichier(isole):
    faux = _Faux()
    h = eco.Horloge("2700", 0, executer=faux, lire=_lire_fixe(2692))
    assert h.poser() == "effectif" and h.conforme and h.etiquette() == "eco=2700(2692)"
    assert faux.appels == [["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "2700,2700"]]
    assert json.load(open(eco.ETAT_ECO.format(index=0)))["mode"] == "2700"
    assert h.rendre() is True and faux.appels[-1][-1] == "-rgc"
    assert not os.path.exists(eco.ETAT_ECO.format(index=0))
    assert h.rendre() is False and len(faux.appels) == 2               # une seule fois


def test_refus_sudo_est_un_etat_nomme_sans_rgc(isole, capsys):
    faux = _Faux(rc={"-lgc": 1})
    h = eco.Horloge("2700", 0, executer=faux, lire=_lire_fixe(225))
    assert h.poser() == "refus sudo" and not h.posee and not h.conforme
    assert h.etiquette() == "eco=2700(libre: refus sudo)"
    assert "refus" in capsys.readouterr().err
    assert h.rendre() is False and all("-rgc" not in a for a in faux.appels)   # rien à rendre
    h2 = eco.Horloge("off", 0, executer=faux, lire=_lire_fixe(225))
    assert h2.poser() == "libre" and h2.conforme and h2.etiquette() == "eco=off(libre)"


def test_engine_fermer_rend_l_horloge_et_la_ligne_de_regime_la_nomme(isole, monkeypatch):
    from acvram.engine.runner import Engine, _eco_texte
    from acvram import regime
    faux = _Faux()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    h = eco.poser_pour_ce_processus(index=0, executer=faux, lire=_lire_fixe(2692))
    assert h.posee and eco.poser_pour_ce_processus() is h              # une fois par processus
    e = eco.etat_eco()
    assert e["conforme"] and _eco_texte(e) == "eco=2700(2692)"
    assert "eco=2700(2692)" in regime.regime_ligne()
    Engine.fermer(types.SimpleNamespace())                             # le corps réel de la méthode
    assert faux.appels[-1][-1] == "-rgc", "Engine.fermer ne rend plus l'horloge"
    # non conforme : l'étiquette nomme l'état, l'instrument refuse
    faux2 = _Faux(rc={"-lgc": 1}); monkeypatch.setattr(eco, "_HORLOGE", None)
    eco.poser_pour_ce_processus(index=0, executer=faux2, lire=_lire_fixe(225))
    assert _eco_texte(eco.etat_eco()) == "eco=2700(libre: refus sudo)"
    sys.path.insert(0, str(RACINE / "outils"))
    import importlib; reg = importlib.import_module("regime")
    r = {"graphes": True, "graphes_demandes": True, "couches_exilees": 0, "experts_exiles": 0,
         "piles_ok": True, "cartes": ["cuda:0"], "eco": eco.etat_eco()}
    with pytest.raises(RuntimeError, match="eco demandé 2700"):
        reg.exiger_regime_nominal(types.SimpleNamespace(regime=lambda: r, regime_ligne=lambda: "x"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ""); monkeypatch.setattr(eco, "_HORLOGE", None)
    assert eco.poser_pour_ce_processus().etat == "sans carte"          # à sec : rien exécuté


def test_signal_rend_l_horloge(isole, tmp_path):
    """Un processus qui a posé reçoit SIGTERM : -rgc part avant la sortie (faux sudo/nvidia-smi
    sur PATH, journal des appels)."""
    binaire = tmp_path / "bin"; binaire.mkdir()
    journal = tmp_path / "journal.txt"
    for nom in ("sudo", "nvidia-smi"):
        (binaire / nom).write_text(textwrap.dedent(f"""\
            #!/bin/sh
            echo "{nom} $*" >> "{journal}"
            case "$*" in *--query-gpu*) echo "2692, 3135, Not Active";; esac
            exit 0
            """))
        (binaire / nom).chmod(0o755)
    code = textwrap.dedent(f"""\
        import os, sys, time
        sys.path.insert(0, {str(RACINE)!r})
        from acvram import eco
        eco.ETAT_ECO = {str(tmp_path / 'eco-{index}.json')!r}
        h = eco.Horloge("2700", 0); h.poser()
        print("POSE", h.etat, flush=True)
        time.sleep(30)
        """)
    env = {**os.environ, "PATH": f"{binaire}:{os.environ['PATH']}", "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().startswith("POSE effectif")
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=20)
    lignes = journal.read_text().splitlines()
    assert any("-lgc 2700,2700" in l for l in lignes) and lignes[-1].endswith("-rgc"), lignes
    assert p.returncode != 0                                            # SIGTERM propagé, pas avalé


def test_acvram_eco_ecrit_la_config_et_off_rend_sous_un_serveur(isole, monkeypatch, capsys):
    """`acvram eco 2100` : config écrite, réglage différé si un pair tient la carte ;
    `acvram eco off` : config écrite ET -rgc appliqué même sous un serveur (l'utilisateur
    rend l'horloge), l'état posé retiré."""
    import argparse
    from acvram import cli
    monkeypatch.setattr(eco, "_tenue_par_un_pair", lambda index: 4242)
    appels = []
    monkeypatch.setattr(eco.subprocess, "run", lambda cmd, **kw: (appels.append(list(cmd)),
                        types.SimpleNamespace(returncode=0, stdout="", stderr=""))[1])
    monkeypatch.setattr(eco, "lire_horloge", _lire_fixe(2692))
    eco._ecrire_etat(0, "2700")
    assert cli.cmd_eco(argparse.Namespace(mode="2100", carte=0)) == 0
    assert eco.mode_demande({}) == "2100" and appels == []             # différé : pas de -lgc sous un pair
    assert "4242" in capsys.readouterr().out
    assert cli.cmd_eco(argparse.Namespace(mode="off", carte=0)) == 0
    assert eco.mode_demande({}) == "off" and appels[-1][-1] == "-rgc"
    assert not os.path.exists(eco.ETAT_ECO.format(index=0))
    monkeypatch.setattr(eco, "_tenue_par_un_pair", lambda index: None)
    assert cli.cmd_eco(argparse.Namespace(mode="2700", carte=0)) == 0 # carte libre : appliqué
    assert appels[-1][-2:] == ["-lgc", "2700,2700"] and eco.mode_demande({}) == "2700"
