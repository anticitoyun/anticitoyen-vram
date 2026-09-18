"""Garde-fou du harnais égal (poste7-profil-verdict-18-09 §4 addendum 13h55) :
2e fois qu'un contrôle de cartes rend faux (14/09 vLLM, 18/09 llama.cpp/
acvram). `_cartes_coherentes` refuse d'écrire une ligne si les deux bras
n'ont pas les mêmes cartes en en-tête -- testé ici en isolation, sans HTTP
ni carte."""
import importlib.util
import pathlib
import sys

_OUTIL = pathlib.Path(__file__).resolve().parents[1] / "scratchpad" / "banc-llamacpp-16-09.py"


def _charger():
    """Le script insère des `sys.path` absolus (son propre arbre, celui du
    dépôt principal) AVANT `from acvram...` -- exécuté tel quel dans une
    suite pytest, il ferait résoudre tout `import acvram` PLUS TARD dans la
    même session vers le dépôt principal au lieu du worktree courant
    (trouvé en combinant ce fichier à test_metrics_troncature.py : le champ
    `cartes` du §4 disparaissait, `acvram/server/app.py` relu venait
    d'ailleurs). `sys.path` est donc restauré juste après le chargement."""
    avant = list(sys.path)
    spec = importlib.util.spec_from_file_location("banc_llamacpp", _OUTIL)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = avant
    return mod


def test_cartes_depuis_regime_exclut_le_cpu():
    mod = _charger()
    assert mod._cartes_depuis_regime(["cpu", "cuda:0"]) == [0]
    assert mod._cartes_depuis_regime(["cuda:1", "cuda:0", "cpu"]) == [0, 1]
    assert mod._cartes_depuis_regime([]) == []
    assert mod._cartes_depuis_regime(None) == []


def test_sans_reference_ce_bras_est_toujours_coherent():
    mod = _charger()
    assert mod._cartes_coherentes([0], None) is True
    assert mod._cartes_coherentes([0, 1], None) is True


def test_meme_carte_est_coherent():
    mod = _charger()
    assert mod._cartes_coherentes([0], "0") is True


def test_cartes_differentes_est_le_cas_reel_du_18_09():
    """Le test qui casse : acvram [0,1] contre llama.cpp [0] -- doit être
    incohérent, pas passer en silence."""
    mod = _charger()
    assert mod._cartes_coherentes([0, 1], "0") is False


def test_ordre_ne_compte_pas():
    mod = _charger()
    assert mod._cartes_coherentes([1, 0], "0,1") is True
