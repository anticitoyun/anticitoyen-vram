"""Une perplexite ne se publie jamais sans son cadrage.

Le 9/09/2026 : `--min-context` a pour defaut 0, et rien ne le signale. La
seule garde existante n'avertit que si le corpus est plus court que la
fenetre — ce qui n'arrive jamais sur wiki.test.raw (1,29 Mo). Oublier
`--min-context 256` donnait donc un chiffre faux d'un facteur proche de 2
(9,525 contre 7,233 a la table du protocole) sans le moindre signe.

La barriere de qualite de six correctifs reposait sur la memoire de
l'operateur, sans filet.
"""
import inspect

from acvram import cli


def test_le_cadrage_est_imprime_avant_la_mesure():
    src = inspect.getsource(cli.cmd_eval)
    assert "min_context=" in src and "window=" in src, \
        "le cadrage n est pas imprime : un chiffre publie ne dira pas lequel a servi"
    i, j = src.index("cadrage"), src.index("results.append")
    assert i < j, "le cadrage doit sortir AVANT la mesure, pas apres"


def test_min_context_nul_est_signale_comme_non_comparable():
    src = inspect.getsource(cli.cmd_eval)
    assert "if not args.min_context" in src, "aucune garde sur min_context=0"
    bloc = src[src.index("if not args.min_context"):]
    assert "N'EST PAS comparable" in bloc, "le message ne dit pas ce qui est en jeu"
    assert "256" in bloc, "le message ne donne pas la valeur du protocole"


def test_le_defaut_reste_zero_et_l_aide_le_dit():
    """Changer le defaut invaliderait les mesures anterieures. On avertit."""
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    src = inspect.getsource(cli)
    i = src.index('"--min-context"')
    bloc = src[i:i + 500]
    assert "default=0" in bloc, "le defaut a change : les mesures d avant ne sont plus comparables"
    assert "NON COMPARABLE" in bloc, "l aide ne dit pas que 0 n est pas comparable"
    assert "256" in bloc, "l aide ne nomme pas la valeur du protocole"
