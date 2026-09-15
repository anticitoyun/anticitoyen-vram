"""poste7 (`poste7-corpus-16-09.md` § 8) : le manifeste doit porter le nom + le
sha256 du corpus de calibration réellement utilisé — ou une absence
EXPLICITE quand `awq=False` — jamais une absence ambiguë. C'est ce qui
manquait pour trancher le doute sur GLM -k48 (`--calib-file` absent :
DEFAULT_CALIB_TEXT intégré, ou le corpus d'éval wiki-gptq passé par erreur
en `--calib-file` ? le manifeste seul ne le disait pas)."""
import hashlib

from acvram.cli import _calib_source
from acvram.quant.collect import DEFAULT_CALIB_TEXT


def test_sans_awq_rend_une_absence_explicite():
    src = _calib_source(use_awq=False, calib_file="/tmp/inexistant.txt")
    assert src == {"fichier": None, "sha256": None, "note": "aucune (awq=False)"}


def test_fichier_de_calibration_reel_est_nomme_et_hache(tmp_path):
    fichier = tmp_path / "corpus.txt"
    fichier.write_text("contenu de calibration")
    src = _calib_source(use_awq=True, calib_file=str(fichier))
    assert src["fichier"] == str(fichier.resolve())
    assert src["sha256"] == hashlib.sha256(
        b"contenu de calibration").hexdigest()
    assert src["note"] is None


def test_sans_fichier_fourni_le_corpus_integre_est_identifie():
    src = _calib_source(use_awq=True, calib_file=None)
    assert src["fichier"] is None
    assert src["sha256"] == hashlib.sha256(
        "".join(DEFAULT_CALIB_TEXT).encode("utf-8")).hexdigest()
    assert "DEFAULT_CALIB_TEXT" in src["note"]


def test_fichier_absent_du_disque_retombe_sur_le_corpus_integre():
    src = _calib_source(use_awq=True, calib_file="/tmp/n-existe-pas-du-tout.txt")
    assert src["fichier"] is None
    assert "DEFAULT_CALIB_TEXT" in src["note"]
