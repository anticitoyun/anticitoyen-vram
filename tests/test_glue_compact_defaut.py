"""C15-3d bis, 20/09 : `ACVRAM_GLUE_COMPACT` vaut 1 par défaut et l'attention fusionnée
tourne à 8 warps (verdict-c15-niveau3-coder-19-09 addenda 05 h 15 / 05 h 26 : Coder b=12
+11,5 % t/s, J 0,966 ×, experts égaux 0,80 × le témoin ≤ 1,2, capture 5/5 ; 4 warps ne
rend rien en W et perd 2 %). Les témoins restent 0 et 4."""
from __future__ import annotations

import os
import subprocess
import sys

from acvram import regime


def _var(nom):
    return next(v for v in regime.VARIABLES if v.nom == nom)


def test_les_defauts_du_regime_sont_1_et_8():
    assert _var("GLUE_COMPACT").defaut == "1" and _var("GLUE_COMPACT").torch == "0"
    assert _var("ATTN_WARPS_COMPACT").defaut == "8" and _var("ATTN_WARPS_COMPACT").torch == "4"


def test_les_modules_lisent_les_memes_defauts_sans_variable():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_GLUE_COMPACT") and k != "ACVRAM_ATTN_WARPS_COMPACT"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram import kernels; from acvram.kernels import attn_paginee as a; "
                          "print(kernels._GLUE_COMPACT, a.WARPS_COMPACT)"], env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["1", "8"], out.stdout + out.stderr[-500:]


def test_la_ligne_de_regime_nomme_la_glue_compacte():
    """Sage (0.6.24, 05 h 40) : la ligne ne disait pas le régime servi ; défaut compris."""
    ligne = regime.regime_ligne()
    assert "glue=compact(8)" in ligne, ligne
    assert regime.glue_texte() == "glue=compact(8)"


def test_le_temoin_et_la_bissection_sont_nommes(monkeypatch):
    from acvram import kernels
    from acvram.kernels import attn_paginee
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 0)
    assert regime.glue_texte() == "glue=temoin"
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1); monkeypatch.setattr(kernels, "_GLUE_COMPACT_ITEMS", "attn,kv")
    monkeypatch.setattr(attn_paginee, "WARPS_COMPACT", 4)
    assert regime.glue_texte() == "glue=compact(4,items=attn,kv)"
