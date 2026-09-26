"""Pièce 168 : la sonde de régime des instruments HTTP (banc-llamacpp-16-09.py, via
energie.py:_regime_noyaux) doit importer L'ARBRE DE SON PROPRE WORKTREE, jamais un chemin
en dur vers un autre (main ou un pair) — sinon la garde a86fa1dd refuse l'import à raison
(mesure du 24/09, pièce 162 : "regime": "indisponible (ImportError…)" sur toute la cellule).

Régression : reproduit un DEUXIÈME arbre acvram minimal (son propre acvram/regime_ligne()
distinct), copie la logique de sys.path du script réel (racine = deux dirname() au-dessus
du script), et vérifie que la sonde lit CE régime-là, pas "indisponible", pas celui du
dépôt réel."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _construire_faux_arbre(tmp_path: Path) -> Path:
    arbre = tmp_path / "faux-worktree"
    (arbre / "acvram").mkdir(parents=True)
    (arbre / "acvram" / "__init__.py").write_text(
        "def regime_ligne():\n    return 'REGIME-DU-FAUX-ARBRE'\n"
    )
    (arbre / ".git").write_text("gitdir: ailleurs\n")
    (arbre / "outils" / "gpu" / "mesure").mkdir(parents=True)
    shutil.copyfile(RACINE / "outils" / "gpu" / "mesure" / "energie.py",
                     arbre / "outils" / "gpu" / "mesure" / "energie.py")
    (arbre / "scratchpad").mkdir()
    shutil.copyfile(RACINE / "scratchpad" / "banc-llamacpp-16-09.py",
                     arbre / "scratchpad" / "banc-llamacpp-16-09.py")
    return arbre


def _sonder(cwd: Path) -> str:
    # Reproduit exactement le bloc de sys.path du script réel (pas d'exécution du
    # script entier, qui exige httpx/argv) : seule la sonde de régime nous intéresse.
    code = (
        "import os, sys\n"
        "_RACINE = os.path.dirname(os.path.dirname(os.path.abspath("
        "os.path.join(os.getcwd(), 'scratchpad', 'banc-llamacpp-16-09.py'))))\n"
        "sys.path.insert(0, os.path.join(_RACINE, 'outils', 'gpu', 'mesure'))\n"
        "sys.path.insert(0, _RACINE)\n"
        "from energie import _regime_noyaux\n"
        "print(_regime_noyaux())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_ARBRE_LIBRE"}
    r = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip()


def test_sonde_lit_le_regime_de_son_propre_arbre(tmp_path):
    arbre = _construire_faux_arbre(tmp_path)
    regime = _sonder(arbre)
    assert regime == "REGIME-DU-FAUX-ARBRE", (
        f"la sonde a lu {regime!r} au lieu du régime du faux arbre — elle importe "
        "acvram depuis un chemin en dur (ou le sys.path ambiant), pas depuis son "
        "propre worktree ; c'est exactement le défaut de la pièce 162 (24/09)")


def test_sonde_ne_leve_jamais_sans_arbre_acvram(tmp_path):
    # Sans .git ni acvram/ dans le faux arbre, la garde a86fa1dd n'y trouve rien à comparer
    # et l'import retombe sur un acvram trouvé ailleurs sur sys.path (installation éditable) —
    # ou échoue proprement. Dans les deux cas _regime_noyaux() ne doit jamais lever : elle
    # capture toute exception (energie.py) et rend une chaîne, jamais un crash du bras.
    arbre = tmp_path / "incomplet"
    (arbre / "scratchpad").mkdir(parents=True)
    (arbre / "outils" / "gpu" / "mesure").mkdir(parents=True)
    shutil.copyfile(RACINE / "outils" / "gpu" / "mesure" / "energie.py",
                     arbre / "outils" / "gpu" / "mesure" / "energie.py")
    shutil.copyfile(RACINE / "scratchpad" / "banc-llamacpp-16-09.py",
                     arbre / "scratchpad" / "banc-llamacpp-16-09.py")
    regime = _sonder(arbre)
    assert regime, "la sonde doit toujours rendre une chaîne non vide, jamais planter"
