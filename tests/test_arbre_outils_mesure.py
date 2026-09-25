"""Pièce 211 : les outils de `outils/gpu/mesure/` qui importent `acvram` ou `energie` doivent
dériver leur racine de `__file__`, jamais du cwd ni d'un chemin en dur vers un autre arbre — sinon
la garde d'import `a86fa1dd` refuse, nommément, quand le script tourne depuis un worktree
(« regime: indisponible (ImportError…) », constat poste5 25/09, même défaut que la 168 sur
`banc-llamacpp-16-09.py`).

Contrôle : pour chaque script concerné, on rejoue SA PROPRE logique de `sys.path` (le code qui
précède son premier `from acvram`/`import acvram`/`from energie import`/`import energie`), avec
`__file__` pointé sur le fichier réel, et on vérifie qu'au moins un chemin inséré est une racine
acvram valide (contient `acvram/__init__.py`) — donc dérivé de l'emplacement RÉEL du script, pas
du cwd ni d'un chemin figé. Capable de rendre faux : retirer une correction de la 211 (cwd, chemin
en dur, ou aucun `sys.path` du tout) fait échouer ce test sur ce fichier précis."""
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER = os.path.join(RACINE, "outils", "gpu", "mesure")
MOTIF_IMPORT = re.compile(r"^\s*(from acvram\b|import acvram\b|from energie import|import energie\b)")


def _fichiers_concernes():
    for nom in sorted(os.listdir(DOSSIER)):
        if not nom.endswith(".py"):
            continue
        chemin = os.path.join(DOSSIER, nom)
        texte = open(chemin, encoding="utf-8").read()
        if MOTIF_IMPORT.search(texte):
            yield nom, chemin


def _prelude(chemin):
    """Le code du fichier AVANT son premier import acvram/energie (celui qui doit poser sys.path)."""
    lignes = []
    for ligne in open(chemin, encoding="utf-8"):
        if MOTIF_IMPORT.match(ligne):
            break
        lignes.append(ligne)
    return "".join(lignes)


def _chemins_inseres(chemin):
    """Exécute le prélude avec __file__ = chemin réel, sys.path/argv réels sauvegardés-restaurés,
    et rend l'ensemble des chemins que le prélude a ajoutés à sys.path (normalisés)."""
    code = _prelude(chemin)
    ancien_path, ancien_argv = sys.path[:], sys.argv[:]
    sys.path[:] = []
    sys.argv[:] = ["prog"] + ["1"] * 8              # de quoi satisfaire un sys.argv[1..4] sans planter
    try:
        exec(compile(code, chemin, "exec"), {"__file__": chemin, "__name__": "__test_arbre__"})
    except BaseException:                            # noqa: BLE001 — SystemExit compris (torch.cuda.device_count())
        pass
    inseres = {os.path.normpath(p) for p in sys.path if isinstance(p, str) and p}
    sys.path[:] = ancien_path
    sys.argv[:] = ancien_argv
    return inseres


def _racine_acvram_valide(chemins):
    return any(os.path.isfile(os.path.join(p, "acvram", "__init__.py")) for p in chemins if os.path.isdir(p))


def test_tous_les_outils_de_mesure_visent_une_racine_acvram_derivee_de_leur_fichier():
    manquants = []
    for nom, chemin in _fichiers_concernes():
        if not _racine_acvram_valide(_chemins_inseres(chemin)):
            manquants.append(nom)
    assert not manquants, (
        "ces outils de outils/gpu/mesure/ n'insèrent, avant leur premier import acvram/energie, "
        "aucun chemin qui résout sur une racine acvram valide : la garde a86fa1dd les refusera "
        f"nommément depuis un worktree — {manquants}")


def test_le_controle_peut_rendre_faux(tmp_path):
    """Bras cassant : un fichier qui insère seulement son propre dossier (le défaut d'origine de
    ttft-service-p145.py/banc-4moteurs.py) doit échouer le contrôle ci-dessus."""
    faux = tmp_path / "faux_outil.py"
    faux.write_text(
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "from acvram import kernels\n"
    )
    assert not _racine_acvram_valide(_chemins_inseres(str(faux))), (
        "le contrôle ne détecte pas un outil qui n'insère que son propre dossier — il ne peut "
        "plus rendre faux")
