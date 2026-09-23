"""Régime sous la charge mesurée (sage-niveau2-clos-retrait-regles-20-09 § 2) :
`SondeCharge` relève horloge, puissance et raisons de bridage pendant la
fenêtre ; l'étiquette porte la médiane d'horloge et la raison, pas l'horloge à
vide. Le b=12 Coder sous -lgc 2700 : médiane 2 550 MHz, 400 W, sw_power_cap."""
from __future__ import annotations

import itertools
import time

from acvram.eco import SondeCharge


def _lecteur(releves):
    it = itertools.cycle(releves)
    return lambda: next(it)


def test_b12_sous_2700_est_un_plafond_de_puissance():
    rel = [["2550", "398.1", "Active", "Not Active", "Not Active", "Not Active"]] * 19 \
        + [["2685", "350.0", "Not Active", "Not Active", "Not Active", "Not Active"]]
    s = SondeCharge(pas=0.01, lire=_lecteur(rel))
    with s:
        time.sleep(0.3)
    b = s.bilan()
    assert b["n"] >= 10 and b["sm_mediane"] == 2550 and b["raison"] == "sw_power_cap" and b["part"] > 0.9
    e = s.etiquette("2700")
    assert e.startswith("eco=2700(plafond 398 W : méd. 2550 MHz") and "% des" in e


def test_sans_bridage_et_sans_releve():
    s = SondeCharge(pas=0.01, lire=_lecteur([["2685", "250.0", "Not Active", "Not Active", "Not Active", "Not Active"]]))
    with s:
        time.sleep(0.1)
    assert s.bilan()["raison"] == "aucune" and s.etiquette("2700").startswith("eco=2700(sans bridage : méd. 2685 MHz")
    vide = SondeCharge(pas=0.01, lire=lambda: None)
    with vide:
        time.sleep(0.05)
    assert vide.bilan() is None and vide.etiquette(None) == "eco=off(sous charge : ?)"


def test_temoin_thermique_nomme_sa_raison():
    rel = [["2100", "300.0", "Not Active", "Not Active", "Active", "Not Active"]]
    s = SondeCharge(pas=0.01, lire=_lecteur(rel))
    with s:
        time.sleep(0.05)
    assert "sw_thermal_slowdown : méd. 2100 MHz" in s.etiquette("2700")
