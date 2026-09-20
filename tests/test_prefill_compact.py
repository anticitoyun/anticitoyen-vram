"""C15-prefill (revue/chantier-c15-prefill-20-09) : la glue du préfill eager sous
`ACVRAM_PREFILL_COMPACT=1`, fusion par fusion, AU BIT contre le témoin (0, défaut).
Chaque test : le chemin compact rend les MÊMES octets que le chemin d'avant, et un
témoin cassant montre que la comparaison peut rendre « faux » (REGLES § 7)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from acvram import kernels, regime


def _var(nom):
    return next(v for v in regime.VARIABLES if v.nom == nom)


# --- régime ----------------------------------------------------------------

def test_le_defaut_est_le_temoin():
    """Défaut 0 tant que le scellé n'est pas mesuré sur carte (poste7, C15-prefill)."""
    assert _var("PREFILL_COMPACT").defaut == "0" and _var("PREFILL_COMPACT").torch == "0"
    assert _var("PREFILL_COMPACT_ITEMS").defaut == ""


def test_le_module_lit_le_defaut_sans_variable():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_PREFILL_COMPACT")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram import kernels; print(kernels._PREFILL_COMPACT)"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["0"], out.stdout + out.stderr[-500:]


def test_la_ligne_de_regime_nomme_la_glue_du_prefill(monkeypatch):
    """Défaut compris : un chiffre de préfill sans cette étiquette ne dit pas son chemin."""
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
    assert regime.prefill_glue_texte() == "prefill_glue=temoin"
    assert "prefill_glue=temoin" in regime.regime_ligne()
    assert not kernels.prefill_compact() and not kernels.prefill_compact("epilogue")
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    assert regime.prefill_glue_texte() == "prefill_glue=compact"
    assert kernels.prefill_compact() and kernels.prefill_compact("permut")
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "epilogue,a8")
    assert regime.prefill_glue_texte() == "prefill_glue=compact(items=epilogue,a8)"
    assert kernels.prefill_compact("epilogue") and not kernels.prefill_compact("residu")
    with pytest.raises(AssertionError):
        kernels.prefill_compact("inconnue")


def test_une_valeur_hors_domaine_est_refusee():
    env = dict(os.environ, ACVRAM_PREFILL_COMPACT="2", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", "import acvram.kernels"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "ACVRAM_PREFILL_COMPACT" in out.stderr
    env = dict(os.environ, ACVRAM_PREFILL_COMPACT_ITEMS="rmsnorm", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", "import acvram.kernels"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "PREFILL_COMPACT_ITEMS" in out.stderr
