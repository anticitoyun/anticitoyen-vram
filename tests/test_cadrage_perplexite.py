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


def test_min_context_nul_dit_ce_qu_il_invalide_ET_ce_qu_il_permet():
    """Le message disait « le protocole impose --min-context 256 ». Or il y a
    DEUX protocoles, et min_context=0 est le cadrage JUSTE pour l'un des deux.

    Le 10/09, un avertissement au singulier a envoye chercher un defaut la ou
    il n'y en avait pas : la barriere de qualite de noyau impose 256, mais
    l'etalon exterieur transformers/GPTQ note des segments disjoints de 2048
    sans contexte reporte, et min_context=0 est alors le seul cadrage qui lui
    corresponde. Un avertissement doit nommer ce qu'il INVALIDE, sinon il
    invalide aussi ce qu'il autorise."""
    src = inspect.getsource(cli.cmd_eval)
    assert "if not args.min_context" in src, "aucune garde sur min_context=0"
    bloc = src[src.index("if not args.min_context"):]
    assert "non comparable" in bloc, "le message ne dit pas ce qui est en jeu"
    assert "256" in bloc, "le message ne donne pas la valeur de la barriere"
    assert "COMPARABLE" in bloc, \
        "le message n'indique pas a quoi min_context=0 EST comparable"
    assert "5,4141" in bloc or "5.4141" in bloc, \
        "le message ne donne pas la reference de l'etalon exterieur"
    assert "2048" in bloc, "le message ne donne pas le cadrage de l'etalon"


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


# --- la configuration effective sort avec le resultat -----------------------

def test_le_corpus_et_son_sha_sortent_avec_le_resultat():
    """Un chiffre qui peut sortir SEUL sera compare a tort."""
    src = inspect.getsource(cli.cmd_eval)
    assert "sha256:" in src, "le sha du corpus n est pas imprime"
    assert "mem_get_info" in src, "la VRAM libre au chargement n est pas dite"


def test_la_liste_des_variables_lues_ne_derive_pas():
    """Le 9/09, MAXTOK=65536 a ete pose dans l environnement d une mesure que
    rien ne lisait : elle s est arretee a 16 fenetres au lieu de 128, sans
    aucun signe. Une variable posee qui ne va nulle part est une consigne
    silencieusement ignoree.

    Cette epreuve verifie que la liste suit le code — sinon la garde
    signalerait comme inconnue une variable devenue valide."""
    import glob
    import pathlib
    import re

    racine = pathlib.Path(cli.__file__).resolve().parent
    reelles = set()
    for f in glob.glob(str(racine / "**" / "*.py"), recursive=True):
        reelles |= set(re.findall(r"ACVRAM_[A-Z0-9_]+",
                                  open(f, errors="ignore").read()))
    # la liste elle-meme contient les noms : on retire ce que le fichier declare
    manquantes = reelles - set(cli.VARIABLES_LUES)
    assert not manquantes, f"variables lues mais absentes de la liste : {sorted(manquantes)}"


def test_la_garde_tire_sur_une_variable_inconnue(capsys, monkeypatch):
    """Sans ce controle, un silence ne se distingue pas d un detecteur muet."""
    monkeypatch.setenv("ACVRAM_CETTE_VARIABLE_N_EXISTE_PAS", "1")
    cli._avertir_variables_inconnues()
    sortie = capsys.readouterr().out
    assert "ACVRAM_CETTE_VARIABLE_N_EXISTE_PAS" in sortie


def test_la_garde_se_tait_sur_une_variable_valide(capsys, monkeypatch):
    """Une garde qui crie sur du valide cesse d etre lue."""
    monkeypatch.setenv("ACVRAM_PLAN_FIGE", "1")
    cli._avertir_variables_inconnues()
    assert "ACVRAM_PLAN_FIGE" not in capsys.readouterr().out
