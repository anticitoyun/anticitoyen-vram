"""outils/gpu/mesure/geo-sequentiel.py : décision jamais avant 5 paires, SE = max(échantillon, 1,97 %/√n),
codes 10/11/12/13 (Sage 07 h 43 : β avait été « décidé » à n = 3 avec un SE d'échantillon sous-estimé)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

JUGE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "gpu" / "mesure" / "geo-sequentiel.py"


def _paires(tmp_path, ratios):
    for k, r in enumerate(ratios, 1):
        (tmp_path / f"ppl-A-t{k}.json").write_text(json.dumps({"ppl": 15.0}))
        (tmp_path / f"ppl-B-t{k}.json").write_text(json.dumps({"ppl": 15.0 * r}))
    p = subprocess.run([sys.executable, str(JUGE), str(tmp_path), "A", "B"], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    return p.returncode, p.stdout


def test_jamais_decide_avant_cinq_paires(tmp_path):
    rc, out = _paires(tmp_path, [1.02, 1.03, 1.03, 1.03])       # +2,7 % net, 4 paires : β le 20/09
    assert rc == 12 and "CONTINUER (4/9)" in out


def test_cinq_paires_a_plus_2_6_decide_faux_avec_le_plancher(tmp_path):
    rc, out = _paires(tmp_path, [1.026] * 5)
    assert rc == 11 and "plancher 0.88" in out                  # 1,97 %/√5 = 0,88 %, l'échantillon dirait 0


def test_le_plancher_empeche_une_decision_sur_un_echantillon_trop_serre(tmp_path):
    rc, out = _paires(tmp_path, [1.004] * 5)                    # +0,4 % : sous le seuil, mais 2 SE plancher = 1,76 %
    assert rc == 12 and "CONTINUER" in out


def test_indecidable_a_neuf(tmp_path):
    rc, out = _paires(tmp_path, [1.0, 1.03] * 4 + [1.0])
    assert rc == 13 and "INDÉCIDABLE à 9" in out


def test_tenu_a_neuf_paires_egales(tmp_path):
    rc, out = _paires(tmp_path, [1.0] * 9)                      # 2 SE plancher = 1,31 % > 0,5 % : même à 9, non tranché
    assert rc == 13
