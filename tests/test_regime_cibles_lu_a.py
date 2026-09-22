"""Une cible `lu_a` doit désigner le module qui DÉFINIT la variable, pas un
module qui la réexporte.

`regime.masquer` fait `setattr(module, attribut, valeur)` (regime.py:638-644)
pour envoyer un chemin vers torch. Si la cible nomme un module qui a seulement
`from .autre import X`, le setattr change une copie du nom : le code qui lit
`X` ne voit rien et le masquage devient un « sans effet » silencieux — un
contrôle qui ne contrôle plus. C'est arrivé deux fois de suite : à la scission
de `engine/attention.py` (21/09, SEUIL_FUSION et _ROPE_KV restés sur
`engine.model`) et elle aurait recommencé à celle de `engine/moe.py` (22/09,
28 cibles, migrées dans le même commit).

Le garde est STATIQUE : le module cible doit contenir une assignation de
niveau module à ce nom. Un import ne compte pas.
"""
from __future__ import annotations

import ast
import importlib
import os

import pytest

from acvram.regime import VARIABLES


def _definit(module: str, attr: str) -> bool:
    chemin = importlib.import_module(module).__file__
    arbre = ast.parse(open(chemin, encoding="utf-8").read())
    for n in arbre.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in n.targets):
            return True
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == attr:
            return True
    return False


@pytest.mark.parametrize("var", [v for v in VARIABLES if v.lu_a is not None],
                         ids=lambda v: v.nom)
def test_la_cible_definit_la_variable(var):
    mod, attr = var.lu_a
    assert _definit(mod, attr), (
        f"{var.env} : `lu_a` nomme {mod}, qui ne DÉFINIT pas {attr} — "
        f"regime.masquer y poserait une valeur que personne ne lit")


def test_masquer_atteint_le_module_qui_lit(monkeypatch):
    """Le bout dynamique : masquer ROPE_KV change bien le témoin lu par
    engine/attention.py, pas le réexport de engine/model.py."""
    from acvram import regime
    from acvram.engine import attention, model
    monkeypatch.setattr(attention, "_ROPE_KV", True, raising=False)
    monkeypatch.setattr(model, "_ROPE_KV", True, raising=False)
    monkeypatch.setenv("ACVRAM_ROPE_KV", "1")
    regime.masquer(["ROPE_KV"])
    assert attention._ROPE_KV is False, "le module qui LIT le témoin n'a pas été masqué"
