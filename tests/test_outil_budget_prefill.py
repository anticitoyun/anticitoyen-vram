"""Rodage de l'instrument qui décidera du budget de prefill, sur processeur.

« L'instrument avant la mesure » : le script ne sera lancé sur carte qu'une
fois, quand poste3 l'aura rendue, et une erreur de montage y coûterait la
manche. On l'exécute donc ici de bout en bout sur un converti minuscule —
les CHIFFRES qu'il rend alors n'ont aucune valeur, seul le montage est
éprouvé.

Ce que ce rodage attrape et qu'une relecture ne peut pas attraper : le
contrôle « la variable est-elle seulement lue » — une consigne posée qui ne
va nulle part est un balayage muet, arrivé le 9/09 avec MAXTOK=65536.
"""
import importlib.util
import os
import pathlib

import pytest

_OUTIL = pathlib.Path(__file__).resolve().parents[1] / "outils" / "gpu" / "mesure" / "budget-prefill.py"


@pytest.fixture(autouse=True)
def _pas_de_fuite_d_environnement(monkeypatch):
    """Le script POSE `ACVRAM_BUDGET_JETONS` lui-même, avant d'importer le
    moteur — c'est voulu là-bas, la variable est lue à l'import.

    Mais l'appeler depuis un essai la laisse posée dans le processus pytest,
    donc dans TOUS les essais qui suivent. Relevé par sonde : la suite se
    terminait avec `ACVRAM_BUDGET_JETONS=0` posée alors qu'elle ne l'était
    pas au départ. À zéro c'était sans effet ; le défaut, lui, ne l'est pas —
    un essai qui laisse un état derrière lui rend la suite dépendante de son
    ordre, et le suivant ne teste plus ce qu'il croit tester.
    """
    monkeypatch.setenv("ACVRAM_BUDGET_JETONS",
                       os.environ.get("ACVRAM_BUDGET_JETONS", "0"))


def _charger():
    spec = importlib.util.spec_from_file_location("budget_prefill", _OUTIL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(converted, budget, regime):
    return ["budget-prefill.py", "--processeur", converted,
            "--budget", str(budget), "--regime", regime,
            "--invite", "200", "--max-model-len", "512",
            "--concurrence", "2", "--max-tokens", "16"]


@pytest.mark.parametrize("regime", ["seule", "charge"])
@pytest.mark.parametrize("budget", [0, 32])
def test_le_montage_tient_dans_les_quatre_cases(converted, regime, budget, capsys):
    assert _charger().main(_args(converted, budget, regime)) == 0
    ligne = capsys.readouterr().out.strip().splitlines()[-1].split("\t")
    assert len(ligne) == 14, "la ligne ne suit plus son en-tete"
    assert int(ligne[0]) == budget and ligne[1] == regime
    passes = int(ligne[5])
    assert passes == (1 if budget == 0 else 7), \
        f"200 jetons par tranches de {budget} : {passes} passes"


def test_un_budget_non_lu_arrete_la_manche(converted, monkeypatch, capsys):
    """Le contrôle qui compte : si le moteur cessait de lire la variable, la
    manche rendrait des chiffres plausibles pour un réglage inexistant."""
    mod = _charger()
    import acvram.engine.runner as runner
    monkeypatch.setattr(runner.Engine, "_budget_jetons", lambda self: 0)
    with pytest.raises(SystemExit, match="budget non lu"):
        mod.main(_args(converted, 32, "seule"))


def test_le_regime_charge_fait_bien_decoder_les_voisines(converted, capsys):
    """Sans quoi le seul régime qui peut conclure ne mesurerait rien."""
    assert _charger().main(_args(converted, 32, "charge")) == 0
    ligne = capsys.readouterr().out.strip().splitlines()[-1].split("\t")
    assert int(ligne[6]) > 0, "aucune voisine n'a produit pendant le prefill"


def test_des_places_de_graphes_non_prises_arretent_la_manche(converted, monkeypatch):
    """`MAX_GRAPHS` est lu À L'IMPORT : posé après, il ne fait rien, en
    silence. Le bras se comparerait alors à lui-même. Le contrôle interroge
    le MODULE, pas l'environnement — c'est la seule façon de le savoir
    avant la mesure plutôt qu'après, en constatant que les bras diffèrent."""
    from acvram.engine import graphs
    mod = _charger()
    monkeypatch.setenv("ACVRAM_MAX_GRAPHS", str(graphs.MAX_GRAPHS + 48))
    with pytest.raises(SystemExit, match="places de graphes non prises"):
        mod.main(_args(converted, 0, "seule"))


def test_le_regime_lot_prefille_bien_tout_le_groupe(converted, capsys):
    """Le régime qui peut préciser le verdict : N invites ensemble, coût fixe
    partagé. Le piège serait qu'une seule soit réellement préfillée — les
    autres servies par le cache de préfixe si les invites étaient identiques,
    ou la boucle s'arrêtant sur la première. La colonne compte donc les
    jetons RÉELLEMENT passés en avant, pas ceux demandés."""
    mod = _charger()
    args = ["budget-prefill.py", "--processeur", converted, "--budget", "0",
            "--regime", "lot", "--lot", "3", "--invite", "200",
            "--max-model-len", "512", "--max-tokens", "8"]
    assert mod.main(args) == 0
    ligne = capsys.readouterr().out.strip().splitlines()[-1].split("\t")
    assert ligne[1] == "lot"
    assert int(ligne[3]) == 600, "trois invites de 200 doivent etre prefillees"
