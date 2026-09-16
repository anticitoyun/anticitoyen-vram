"""poste7 (`poste7-corpus-16-09.md` § 8) : le manifeste doit porter le nom + le
sha256 du corpus de calibration réellement utilisé — ou une absence
EXPLICITE quand `awq=False` — jamais une absence ambiguë. C'est ce qui
manquait pour trancher le doute sur GLM -k48 (`--calib-file` absent :
corpus intégré, ou le corpus d'éval wiki-gptq passé par erreur
en `--calib-file` ? le manifeste seul ne le disait pas)."""
import hashlib
import os

from acvram.cli import _calib_source


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
    """Le corpus intégré est un FICHIER du paquet (Gutenberg #1342, poste7
    poste7-calibration-verdict-17-09) : nom et sha256 du fichier réel."""
    from acvram.quant.collect import DEFAULT_CALIB_FILE
    src = _calib_source(use_awq=True, calib_file=None)
    assert src["fichier"] == DEFAULT_CALIB_FILE and os.path.isfile(src["fichier"])
    assert os.path.getsize(src["fichier"]) > 700_000
    with open(src["fichier"], "rb") as fh:
        assert src["sha256"] == hashlib.sha256(fh.read()).hexdigest()
    assert "Gutenberg" in src["note"]


def test_fichier_absent_du_disque_est_une_erreur_pas_un_repli():
    """Un --calib-file introuvable ne doit plus calibrer en silence sur le
    corpus intégré : load_calib_ids refuse (la conversion retombe alors sur
    awq=False, explicitement)."""
    import pytest
    from acvram.quant.collect import load_calib_ids
    with pytest.raises(FileNotFoundError):
        load_calib_ids(object(), "/tmp/n-existe-pas-du-tout.txt", 4, 64, 1000)


def test_le_manifeste_porte_ce_que_la_calibration_a_rendu():
    """poste7 : calib_seqs/calib_tokens = ce que load_calib_ids a RÉELLEMENT
    rendu — jamais un défaut de classe qui ressemble à une mesure (16/128 sont
    restés au manifeste sans qu'aucune passe ne les ait produits). Casse si un
    défaut non nul revient sur ConversionOptions, ou si le corpus intégré ne
    rend pas les 32 séquences demandées par défaut."""
    from acvram.quant.collect import load_calib_ids
    from acvram.quant.convert import ConversionOptions

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 500 for c in text]
    opts = ConversionOptions(out_dir="/tmp/x")
    assert (opts.calib_seqs, opts.calib_tokens) == (0, 0), "les défauts de classe doivent être 0/0"
    calib = load_calib_ids(_Tok(), None, 32, 512, 1000)
    assert len(calib) == 32 and all(len(c) == 512 for c in calib)
    reel = (len(calib), sum(len(c) for c in calib))
    assert reel == (32, 32 * 512)
    opts = ConversionOptions(out_dir="/tmp/x", calib_seqs=reel[0], calib_tokens=reel[1])
    assert (opts.calib_seqs, opts.calib_tokens) == reel
