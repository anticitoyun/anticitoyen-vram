"""Pièce 117 : la bannière de démarrage (`cmd_serve`, SANS `--regime`) doit imprimer le repli de spéculation
NOMMÉ (`speculator.repli`, posé par `repli_speculatif`, pièce 105) — `engine.regime_ligne()` le portait déjà
(donc `--regime` ne prouve rien ici), mais la bannière courte restait muette. Casse si le print disparaît de
`cmd_serve` avant `create_app`. À sec, `uvicorn.run` et `create_app` neutralisés (aucun réseau, aucun serveur)."""
import io
import contextlib
from types import SimpleNamespace

from acvram import cli
from acvram.engine.runner import _speculation_texte


class _ModeleSansTete:
    mtp = None
    mtp_raison = "aucune tête dans le manifeste"
    nbytes = 4096


class _ChargeSansTete:
    model = _ModeleSansTete()


class _EngineFausse:
    """Assez d'`Engine` pour traverser `cmd_serve` jusqu'à `create_app` — pas de carte, pas de graphes."""

    def __init__(self, loaded, tokenizer, **kw):
        self.speculator = kw.get("speculator")
        self.allocator = SimpleNamespace(num_blocks=1)
        self._garde_spec = SimpleNamespace(
            lot_max=2,
            etat_dict=lambda nom: {"mode": nom, "garde_active": True, "gain_moyen": None, "lot_max": 2},
        )
        self.graphs = None
        self.max_model_len = 4096

    def demarrer_service(self, strict=False, warm_max_len=2048):
        return (100, 0)

    def regime(self):
        spec = (dict(self._garde_spec.etat_dict(self.speculator.name),
                     repli=getattr(self.speculator, "repli", None))
                if self.speculator is not None
                else {"mode": "off", "garde_active": False, "gain_moyen": None, "lot_max": 0})
        return {"speculation": spec}


def _lancer(monkeypatch, speculative: str) -> str:
    monkeypatch.setattr("acvram.engine.loader.load_model", lambda *a, **k: _ChargeSansTete())
    monkeypatch.setattr("acvram.server.chat.load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr("acvram.engine.runner.Engine", _EngineFausse)
    monkeypatch.setattr("acvram.server.app.create_app", lambda *a, **k: object())
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    args = cli.build_parser().parse_args(
        ["serve", "modele/factice", "--speculative", speculative])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cli.cmd_serve(args)
    assert rc == 0
    return out.getvalue()


def test_la_banniere_courte_nomme_le_repli_auto_vers_ngram(monkeypatch):
    texte = _lancer(monkeypatch, "auto")
    attendu = _speculation_texte({
        "mode": "ngram", "garde_active": True, "gain_moyen": None, "lot_max": 2,
        "repli": "mtp absent : aucune tête dans le manifeste",
    }).strip()
    assert "mtp absent" in texte, texte
    assert attendu in texte, (attendu, texte)


def test_la_banniere_courte_reste_muette_sans_repli(monkeypatch):
    # ngram choisi directement (pas `auto`) : aucun repli à nommer.
    texte = _lancer(monkeypatch, "ngram")
    assert "mtp absent" not in texte
    assert "speculation=ngram" not in texte          # pas de fragment de régime dans la bannière courte
    assert "speculation   : ngram" in texte, texte
