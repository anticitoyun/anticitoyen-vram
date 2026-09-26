"""Pièce 132 (chef) : `acvram eval` écrivait le chemin ABSOLU du corpus (« /home/<utilisateur>/... »)
dans son JSON de sortie — a fait sauter le cliquet d'identité (`test_depot_sans_identite`) à la fusion
de poste2, corrigé à la main par le chef. `corpus_chemin` ne porte plus que le nom du fichier ;
`corpus_sha256` identifie déjà le contenu sans ambiguïté. Casse si un `/home/` réapparaît."""
import hashlib
import os

from acvram.evaluate import _identite_corpus


def test_corpus_reel_pas_de_chemin_absolu(tmp_path):
    d = tmp_path / "home_utilisateur" / "corpus"
    d.mkdir(parents=True)
    f = d / "wiki-gptq.txt"
    contenu = "quelques jetons de texte pour le corpus\n" * 10
    f.write_text(contenu, encoding="utf-8")
    chemin, octets, sha = _identite_corpus(str(f), "")
    assert chemin == "wiki-gptq.txt", chemin
    assert "/" not in chemin and "\\" not in chemin
    assert "home_utilisateur" not in chemin
    assert octets == len(contenu.encode("utf-8"))
    assert sha == hashlib.sha256(contenu.encode("utf-8")).hexdigest()[:24]


def test_corpus_par_defaut_sans_chemin():
    chemin, octets, sha = _identite_corpus(None, "texte intégré")
    assert chemin == "(corpus par defaut, integre)"
    assert "/" not in chemin
    assert octets == len("texte intégré".encode("utf-8"))
    assert sha == hashlib.sha256("texte intégré".encode("utf-8")).hexdigest()[:24]


def test_aucun_home_dans_la_sortie_json(tmp_path):
    """Contrôle direct de la classe de faute (`/home/` dans la sortie), pas seulement du nom exact."""
    d = tmp_path / "home" / "quelqu_un"
    d.mkdir(parents=True)
    f = d / "corpus.txt"
    f.write_text("x", encoding="utf-8")
    chemin, _, _ = _identite_corpus(str(f), "")
    assert "/home/" not in chemin
    assert os.sep not in chemin
