"""Pièce 283 (poste5 277a-bis puis 277fix, poste5-277 9fdea0a23 ; élargie sur ordre chef) :
`ngram` (défaut de `serve` jusqu'ici) émettait des jetons hors distribution — cause trouvée
par poste5 : le pipeline n'était pas vidé au passage du décodage simple au spéculatif, jetons
répétés. Le bogue touche TOUT modèle servi avec ngram, dense compris (pas seulement les
hybrides, où 277a-bis avait d'abord vu 13 à 24 logits sous le premier choix, 4/5 invites).
Défaut désormais `none` pour TOUS les alias ; `ngram` reste servable sur demande explicite, avec un
avertissement VRAI après la 277fix : bogue corrigé, écarts restants = quasi-égalités (marge < 0,02 sur le
Coder), le ngram peut faire plus de pas que none (jusqu'à 55 pour 32 jetons), il n'est pas le défaut.
À sec : `load_model`, `Engine`, `create_app`, `uvicorn.run` neutralisés (aucune carte)."""
import contextlib
import io
from types import SimpleNamespace

from acvram import cli


class _ModeleSansTete:
    mtp = None
    mtp_raison = "aucune tête dans le manifeste"
    nbytes = 4096


def _charge(hybride: bool):
    return SimpleNamespace(model=_ModeleSansTete(),
                           spec=SimpleNamespace(layer_types=["linear_attention", "full_attention"] if hybride else []))


class _EngineFausse:
    def __init__(self, loaded, tokenizer, **kw):
        self.speculator = kw.get("speculator")
        self.allocator = SimpleNamespace(num_blocks=1)
        self._garde_spec = SimpleNamespace(
            lot_max=2, etat_dict=lambda nom: {"mode": nom, "garde_active": True, "gain_moyen": None, "lot_max": 2})
        self.graphs = None
        self.max_model_len = 4096

    def demarrer_service(self, strict=False, warm_max_len=2048):
        return (100, 0)

    def regime(self):
        spec = (dict(self._garde_spec.etat_dict(self.speculator.name), repli=getattr(self.speculator, "repli", None))
                if self.speculator is not None else {"mode": "off", "garde_active": False, "gain_moyen": None, "lot_max": 0})
        return {"speculation": spec}


def _lancer(monkeypatch, hybride: bool, *args_serve: str):
    monkeypatch.setattr("acvram.engine.loader.load_model", lambda *a, **k: _charge(hybride))
    monkeypatch.setattr("acvram.server.chat.load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr("acvram.engine.runner.Engine", _EngineFausse)
    monkeypatch.setattr("acvram.server.app.create_app", lambda *a, **k: object())
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    parsed = cli.build_parser().parse_args(["serve", "modele/factice", *args_serve])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.cmd_serve(parsed)
    assert rc == 0, err.getvalue()
    return parsed, out.getvalue()


def test_defaut_none_sur_hybride_sans_rien_demander(monkeypatch):
    parsed, sortie = _lancer(monkeypatch, True)
    assert parsed.speculative == "none"
    assert "speculation   : none" in sortie
    assert "AVERTISSEMENT" not in sortie


def test_defaut_none_sur_non_hybride_aussi(monkeypatch):
    """283 élargie : le défaut est none PARTOUT désormais, le bogue touchait aussi le dense."""
    parsed, sortie = _lancer(monkeypatch, False)
    assert parsed.speculative == "none"
    assert "speculation   : none" in sortie
    assert "AVERTISSEMENT" not in sortie


def test_ngram_explicite_sur_hybride_sert_quand_meme_avec_avertissement(monkeypatch):
    parsed, sortie = _lancer(monkeypatch, True, "--speculative", "ngram")
    assert parsed.speculative == "ngram"
    assert "AVERTISSEMENT" in sortie and "bogue 277 corrigé" in sortie and "quasi-égalités" in sortie
    assert "jusqu'à 55 pour 32 jetons" in sortie and "pas le défaut" in sortie
    assert "qualification en cours" not in sortie          # texte d'avant la 277fix, faux désormais


def test_ngram_explicite_sur_non_hybride_avertit_aussi(monkeypatch):
    """283 élargie : l'avertissement ne dépend plus de l'hybride, le bogue touchait le dense aussi."""
    parsed, sortie = _lancer(monkeypatch, False, "--speculative", "ngram")
    assert parsed.speculative == "ngram"
    assert "AVERTISSEMENT" in sortie and "bogue 277 corrigé" in sortie
    # chef (relecture 283, puis 277cm) : ce message n'existe qu'AVEC la 277fix dans l'arbre (même commit) —
    # le correctif y est, le texte le dit ; il ne dit jamais que le ngram est redevenu le défaut.
    assert "pas le défaut" in sortie


def test_none_explicite_pas_d_avertissement(monkeypatch):
    parsed, sortie = _lancer(monkeypatch, True, "--speculative", "none")
    assert parsed.speculative == "none" and "AVERTISSEMENT" not in sortie
