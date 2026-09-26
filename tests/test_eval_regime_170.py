"""Pièce 170 (poste6, 25/09) : `acvram eval` imprime la ligne [régime] du chargement (celle des instruments KL/ABBA, plus
la disposition Marlin du modèle et son compteur de replis) — sinon le bras B d'une PPL n'est prouvé par rien dans sa sortie."""
import types

import pytest

from acvram import evaluate, kernels


def _modele(bilan):
    return types.SimpleNamespace(proj_marlin_bilan=bilan, nbytes=0)


def test_le_fragment_marlin_nomme_disposition_et_replis(monkeypatch):
    monkeypatch.delenv("ACVRAM_PROJ_MARLIN", raising=False)
    b = {"doubles": 0, "seuls": 305, "octets_doubles": 0, "exclus": 96, "inexacts": 2, "capacite_kv": 8192}
    assert kernels.marlin_bilan_texte(b) == "+marlin(doubles=0,seuls=305,0.00Go,kv=8192,exclus=96,replis=2)"
    assert kernels.marlin_bilan_texte(_modele(b)) == kernels.marlin_bilan_texte(b)
    assert kernels.marlin_bilan_texte({"repli": "port Marlin : compilation échouée"}) == \
        "+marlin(repli:port Marlin : compilation échouée)"
    assert kernels.marlin_bilan_texte({"portee": "denses:moe-exclu"}) == "+marlin(moe-exclu)"
    assert kernels.marlin_bilan_texte(None) == ""
    monkeypatch.setenv("ACVRAM_PROJ_MARLIN", "0")
    assert kernels.marlin_bilan_texte(None) == "+marlin(off:ACVRAM_PROJ_MARLIN=0)"


def test_la_ligne_de_l_eval_est_celle_des_instruments(monkeypatch):
    monkeypatch.delenv("ACVRAM_PROJ_MARLIN", raising=False)
    from acvram import regime_ligne
    b = {"doubles": 0, "seuls": 7, "octets_doubles": 0, "exclus": 1, "capacite_kv": 512}
    ligne = evaluate.ligne_regime(_modele(b))
    assert ligne.startswith(regime_ligne()) and ligne.startswith("[régime]")
    assert "+marlin(doubles=0,seuls=7,0.00Go,kv=512,exclus=1,replis=0)" in ligne
    assert " marlin_port_so=" in ligne


def test_le_service_et_l_eval_partagent_le_fragment():
    import inspect
    from acvram.engine import runner
    src = inspect.getsource(runner)
    assert "kernels.marlin_bilan_texte(self.model)" in src, "la ligne du service n'utilise plus le fragment partagé"


class _Vue(Exception):
    pass


def test_perplexity_imprime_la_ligne_juste_apres_le_chargement(monkeypatch, tmp_path):
    """Casse si `perplexity` n'appelle plus `ligne_regime` après `load_model` (avant toute fenêtre)."""
    from acvram.engine import loader
    from acvram.server import chat
    monkeypatch.setattr(loader, "load_model", lambda *a, **k: types.SimpleNamespace(model=_modele(None)))
    monkeypatch.setattr(chat, "load_tokenizer", lambda d: types.SimpleNamespace(encode=lambda t: list(range(64))))
    monkeypatch.setattr(evaluate, "_load_corpus", lambda p: "un corpus factice " * 8)

    def vue(model):
        raise _Vue(model)
    monkeypatch.setattr(evaluate, "ligne_regime", vue)
    with pytest.raises(_Vue):
        evaluate.perplexity(str(tmp_path / "modele-factice"), corpus_path=None)


def test_le_resultat_porte_le_regime():
    r = evaluate.EvalResult(model="m", regime="[régime] x")
    assert r.to_dict()["regime"] == "[régime] x"
