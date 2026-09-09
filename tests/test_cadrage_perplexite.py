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


# --- plusieurs modeles dans un processus -----------------------------------

def test_plusieurs_modeles_dans_un_processus_sont_signales():
    """Le 9/09 : une barriere de qualite a mesure un nvfp4 puis charge un
    bf16 par-dessus — 32 MLP de plus en RAM hote, puis OutOfMemoryError avec
    111,88 MiB libres sur 31,36 Gio. Le second modele etait mesure en regime
    degrade, et une perplexite prise sur un plan degrade n est comparable a
    rien."""
    src = inspect.getsource(cli.cmd_eval)
    assert "len(args.models) > 1" in src, "aucune garde sur le multi-modeles"
    i = src.index("len(args.models) > 1")
    bloc = src[i:i + 400]
    assert "un appel par modele" in bloc, \
        "le message ne dit pas quoi faire a la place"


def test_la_memoire_est_liberee_entre_deux_modeles():
    """Liberer ne garantit pas un plan identique — le cache de l allocateur
    et la fragmentation survivent — mais ne pas liberer garantit l inverse."""
    src = inspect.getsource(cli.cmd_eval)
    assert "empty_cache" in src, "la VRAM n est pas rendue entre deux modeles"
    assert "gc.collect" in src, "les references Python ne sont pas laches"
    # la liberation doit venir APRES la mesure, pas avant
    assert src.index("results.append") < src.index("empty_cache")


def test_l_avertissement_ne_se_declenche_pas_sur_un_seul_modele():
    """Une garde qui crie toujours cesse d etre lue."""
    src = inspect.getsource(cli.cmd_eval)
    assert "if len(args.models) > 1:" in src, \
        "la garde doit etre conditionnelle, pas systematique"
