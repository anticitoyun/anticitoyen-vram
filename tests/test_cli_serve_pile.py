"""`acvram serve` qui meurt au chargement laisse sa PILE dans son journal (stderr), pas seulement `str(e)` : P3 (4)
30B (Manon 21/09) n avait que « size of tensor a (256) must match b (257) », sans site. Les autres commandes gardent
la ligne courte (ACVRAM_TRACEBACK=1 pour la pile)."""
import io, contextlib, pytest
from acvram import cli


def _rc_et_stderr(monkeypatch, argv, exc):
    def plante(args): raise exc
    def plante_eco(args): raise exc                # deux fonctions : `func is cmd_serve` doit distinguer
    monkeypatch.setattr(cli, "cmd_serve", plante)
    monkeypatch.setattr(cli, "cmd_eco", plante_eco)
    monkeypatch.delenv("ACVRAM_TRACEBACK", raising=False)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, err.getvalue()


def test_serve_mort_imprime_la_pile(monkeypatch):
    rc, err = _rc_et_stderr(monkeypatch, ["serve", "/nulle/part"], RuntimeError("The size of tensor a (256) must match the size of tensor b (257)"))
    assert rc == 1 and "RuntimeError: The size of tensor a (256)" in err and "Traceback" in err and "plante" in err


def test_autre_commande_reste_courte(monkeypatch):
    rc, err = _rc_et_stderr(monkeypatch, ["eco", "etat"], RuntimeError("x"))
    assert rc == 1 and "RuntimeError: x" in err and "Traceback" not in err
