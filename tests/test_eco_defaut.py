"""Éco d'horloge par défaut (sage-eco-2700-defaut-19-09 § 1-3), à sec avec un faux
nvidia-smi : pose au chargement, rendu à `Engine.fermer`, rendu sur signal, refus
sudo = état nommé, config.json, lecture de l'horloge par l'état posé + clocks.sm
(le champ applications_clocks_setting ne dit rien sous -lgc : Manon, 19/09 23 h)."""
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
_RIEN = lambda: None                                   # pas de charge CUDA à sec


def _lire_fixe(sm, sm_repos=None):
    """Horloge lue : `sm` sous charge, `sm_repos` (défaut : 225) au repos — une carte
    verrouillée oisive lit 225, c'est la lecture sous charge qui compte."""
    repos = 225 if sm_repos is None else sm_repos

    def lire(index=0, sous_charge=False, **kw):
        v = sm if sous_charge else repos
        return _LIRE_ORIG(index, f"{v}, 3135, {'Not Active' if sous_charge else 'Active'}", sous_charge=sous_charge)
    return lire


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
    assert eco.etiquette_horloge(eco.lire_horloge(0, "2692, 3135, Not Active", etat={})) == "libre"          # au repos : rien de concluant
    assert eco.etiquette_horloge(eco.lire_horloge(0, "2692, 3135, Not Active", etat={}, sous_charge=True)) == "lgc2700?"
    assert eco.etiquette_horloge(eco.lire_horloge(0, "3030, 3135, Not Active", etat={}, sous_charge=True)) == "libre"
    # l'état posé vient du fichier quand `etat` n'est pas donné
    eco._ecrire_etat(0, "2700")
    assert eco.lire_horloge(0, "2692, 3135, Not Active")["verrou"] is True
    eco._ecrire_etat(0, None)
    assert eco.lire_horloge(0, "2692, 3135, Not Active")["verrou"] is False
    assert eco.lire_horloge(0, "n'importe quoi")["verrou"] is None


def test_pose_rendu_idempotent_et_etat_fichier(isole):
    faux = _Faux()
    h = eco.Horloge("2700", 0, executer=faux, lire=_lire_fixe(2692), charge=_RIEN)
    assert h.poser() == "effectif" and h.conforme and h.etiquette() == "eco=2700(2692)"
    assert faux.appels == [["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "2700,2700"]]
    assert json.load(open(eco.ETAT_ECO.format(index=0)))["mode"] == "2700"
    assert h.rendre() is True and faux.appels[-1][-1] == "-rgc"
    assert not os.path.exists(eco.ETAT_ECO.format(index=0))
    assert h.rendre() is False and len(faux.appels) == 2               # une seule fois


def test_refus_sudo_est_un_etat_nomme_sans_rgc(isole, capsys):
    faux = _Faux(rc={"-lgc": 1})
    h = eco.Horloge("2700", 0, executer=faux, lire=_lire_fixe(3030), charge=_RIEN)
    assert h.poser() == "refus sudo" and not h.posee and not h.conforme
    assert h.etiquette() == "eco=2700(3030: refus sudo)"        # l'horloge lue sous charge, libre au boost
    assert "refus" in capsys.readouterr().err
    assert h.rendre() is False and all("-rgc" not in a for a in faux.appels)   # rien à rendre
    h2 = eco.Horloge("off", 0, executer=faux, lire=_lire_fixe(3030), charge=_RIEN)
    assert h2.poser() == "libre" and h2.conforme and h2.etiquette() == "eco=off(3030)"


def test_engine_fermer_rend_l_horloge_et_la_ligne_de_regime_la_nomme(isole, monkeypatch):
    from acvram.engine.runner import Engine, _eco_texte
    from acvram import regime
    faux = _Faux()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    h = eco.poser_pour_ce_processus(index=0, executer=faux, lire=_lire_fixe(2692), charge=_RIEN)
    assert h.posee and eco.poser_pour_ce_processus() is h              # une fois par processus
    e = eco.etat_eco()
    assert e["conforme"] and _eco_texte(e) == "eco=2700(2692)"
    assert "eco=2700(2692)" in regime.regime_ligne()
    Engine.fermer(types.SimpleNamespace())                             # le corps réel de la méthode
    assert faux.appels[-1][-1] == "-rgc", "Engine.fermer ne rend plus l'horloge"
    # non conforme : l'étiquette nomme l'état, l'instrument refuse
    faux2 = _Faux(rc={"-lgc": 1}); monkeypatch.setattr(eco, "_HORLOGE", None)
    eco.poser_pour_ce_processus(index=0, executer=faux2, lire=_lire_fixe(3030), charge=_RIEN)
    assert _eco_texte(eco.etat_eco()) == "eco=2700(3030: refus sudo)"
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
    monkeypatch.setattr(eco, "lire_horloge", _lire_fixe(2692)); monkeypatch.setattr(eco, "_charge_cuda", _RIEN)
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


def test_verrouillee_oisive_au_chargement_et_verrou_exterieur_sous_off(isole, monkeypatch):
    """Les deux bras faux de verdict-verif-eco-defaut-19-09 (Manon 16ea8b0) :
    (1) au chargement la carte verrouillée est OISIVE (225 MHz) — l'effectif se lit
    sous charge (2 692) : conforme, pas « non pris » ; (2) `-lgc 2700` posé à la main
    puis `ACVRAM_ECO=off` : sous charge la carte reste à 2 692 au lieu du boost —
    état nommé « verrou 2700 posé hors processus », jamais `eco=off(libre)`."""
    from acvram.engine.runner import _eco_texte
    faux = _Faux()
    h = eco.Horloge("2700", 0, executer=faux, lire=_lire_fixe(2692, sm_repos=225), charge=_RIEN)
    assert h.poser() == "effectif" and h.etiquette() == "eco=2700(2692)"
    assert h.relire(sous_charge=False) == "225"                         # ce que la carte oisive dit
    assert h.relire() == "2692" and h.conforme                          # ce qui compte
    h.rendre(); n = len(faux.appels)
    h2 = eco.Horloge("off", 0, executer=faux, lire=_lire_fixe(2655, sm_repos=870), charge=_RIEN)
    assert h2.poser() == "verrou 2700 posé hors processus" and not h2.conforme
    assert h2.etiquette() == "eco=off(2655: verrou 2700 posé hors processus)"
    assert h2.rendre() is False and len(faux.appels) == n                # on ne rend pas ce qu'on n'a pas posé
    h3 = eco.Horloge("off", 0, executer=faux, lire=_lire_fixe(3030, sm_repos=225), charge=_RIEN)
    assert h3.poser() == "libre" and h3.conforme and h3.etiquette() == "eco=off(3030)"
    # etat_eco(relire=True) : ce que Engine.regime() fait après le chargement
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0"); monkeypatch.setattr(eco, "_HORLOGE", None)
    lu = {"n": 0}
    def lire(index=0, sous_charge=False, **kw):
        lu["n"] += 1
        return _LIRE_ORIG(index, "2692, 3135, Not Active" if sous_charge else "225, 3135, Active", sous_charge=sous_charge)
    eco.poser_pour_ce_processus(index=0, executer=_Faux(), lire=lire, charge=_RIEN)
    e = eco.etat_eco(relire=True)
    assert e["conforme"] and _eco_texte(e) == "eco=2700(2692)" and lu["n"] >= 2


def test_lecture_sous_charge_attend_un_regime(isole):
    """Carte froide (Manon, verdict-verif-eco-defaut addendum 21 h 08) : deux lectures
    égales sur un palier de montée (1 102) ne sont pas un régime. Stable = après ≥ 0,8 s
    de charge, deux lectures à ± 30 et ≥ 1 000 ; ou, cible demandée, deux lectures dans
    sa bande (arrivée sur la consigne). Séquences de Jérôme : [1102, 1102, 2692, 2692…]
    → 2692 ; [870×4, 487×4, 2872, 2872…] → libre."""
    def lecteur(seq):
        it = iter(list(seq) + [seq[-1]] * 40)                        # la dernière valeur tient
        return lambda index=0, sous_charge=False, **kw: _LIRE_ORIG(index, f"{next(it)}, 3135, Not Active", sous_charge=sous_charge)
    # cible demandée (serveur verrouillé, carte froide) : les 1 102 n'arrêtent pas, 2 692 ×2 oui
    h = eco.lire_sous_charge(0, lecteur([1102, 1102, 2692, 2692]), charge=_RIEN, attente=0.05, cible=2700)
    assert h["lectures"] == [1102, 1102, 2692, 2692] and h["stable"] and h["sm_mhz"] == 2692
    # libre, montée lente : rien n'est cru avant 0,8 s, puis 2 872 ×2
    h = eco.lire_sous_charge(0, lecteur([870] * 4 + [487] * 4 + [2872, 2872]), charge=_RIEN, attente=0.05)
    assert h["stable"] and h["sm_mhz"] == 2872 and eco.etiquette_horloge(h) == "libre"
    # libre, palier de montée à 1 102 tenu deux lectures AVANT 0,8 s : pas cru
    h = eco.lire_sous_charge(0, lecteur([225, 1102, 1102, 1980, 2977, 2985]), charge=_RIEN, attente=0.05)
    assert h["stable"] and h["sm_mhz"] == 2985 and eco.etiquette_horloge(h) == "libre"
    # verrouillée hors processus, sans cible : 2 692 tenu après le minimum
    h = eco.lire_sous_charge(0, lecteur([225, 1102, 1102, 2692]), charge=_RIEN, attente=0.05)
    assert h["stable"] and eco.etiquette_horloge(h) == "lgc2700?"
    # une Horloge posée par ce processus, lue pendant la montée : « effectif », pas « non pris »
    hz = eco.Horloge("2700", 0, executer=_Faux(), lire=lecteur([225, 225, 1102, 1102, 1980, 2655, 2692]), charge=_RIEN)
    assert hz.poser() == "effectif" and hz.effectif == "2692"
    # jamais stable dans le plafond : dit, effectif = dernière lecture
    lent = iter(range(300, 9000, 100))
    h = eco.lire_sous_charge(0, lambda index=0, sous_charge=False, **kw: _LIRE_ORIG(index, f"{next(lent)}, 3135, Not Active", sous_charge=sous_charge),
                             charge=_RIEN, attente=0.05, plafond=0.3, minimum=0.1)
    assert h["stable"] is False and len(h["lectures"]) >= 4


def test_verrou_est_un_plafond_lecture_au_dessus_refus_en_dessous_bridage(isole):
    """(a) sage-mesure1-ter-c17-ecrit-c14-juge-20-09 : sous -lgc 2700 le limiteur de puissance
    tient Marlin à 1 950-2 265 MHz (400 W) — c'est le verrou effectif ; une lecture > 2 700 + 60
    sous charge = pas de verrou → refus ; (b) MHz et W publiés, jamais critères."""
    faux = _Faux()
    h = eco.Horloge("2700", 0, executer=faux, lire=_lire_fixe(2265), charge=_RIEN)   # bridé sous le verrou
    assert h.poser() == "effectif" and h.conforme and h.etiquette() == "eco=2700(2265)"
    h2 = eco.Horloge("2700", 0, executer=_Faux(), lire=_lire_fixe(2977), charge=_RIEN)  # au-dessus : pas de verrou
    assert h2.poser() == "non pris" and not h2.conforme and "non pris" in h2.etiquette()
    # lire_horloge avec état posé : en dessous tenu, au-dessus faux
    assert eco.lire_horloge(0, "1950, 3135, Not Active", etat={"mode": "2700"})["verrou"] is True
    assert eco.lire_horloge(0, "2977, 3135, Not Active", etat={"mode": "2700"})["verrou"] is False


def test_une_montee_qui_traverse_une_bande_n_est_pas_un_verrou(isole):
    """Manon 858298c : `eco etat` pendant une montée a lu [690×3, 435×3, 2062, 2062] → « lgc2100? »
    à tort ; sans consigne il faut TROIS lectures à ± 30 avant de conclure ; la montée continue
    (2 977, 2 985) rend « libre » ; un vrai verrou 2 100 (2 062 ×3) rend bien lgc2100?."""
    def lecteur(seq):
        it = iter(list(seq) + [seq[-1]] * 40)
        return lambda index=0, sous_charge=False, **kw: _LIRE_ORIG(index, f"{next(it)}, 3135, Not Active", sous_charge=sous_charge)
    h = eco.lire_sous_charge(0, lecteur([690] * 3 + [435] * 3 + [2062, 2062, 2977, 2985, 2985]), charge=_RIEN, attente=0.05)
    assert h["stable"] and eco.etiquette_horloge(h) == "libre" and h["sm_mhz"] == 2985
    h = eco.lire_sous_charge(0, lecteur([690] * 3 + [435] * 3 + [2062, 2062, 2062]), charge=_RIEN, attente=0.05)
    assert h["stable"] and eco.etiquette_horloge(h) == "lgc2100?"
    # avec consigne 2700 posée par ce processus : deux lectures dans la bande suffisent (arrivée)
    h = eco.lire_sous_charge(0, lecteur([225, 1102, 2692, 2692]), charge=_RIEN, attente=0.05, cible=2700)
    assert h["stable"] and h["lectures"] == [225, 1102, 2692, 2692]
