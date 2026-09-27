"""Pièce anticitoyen-vram-9dc : le filtre du champ de recherche d'acvram-gui (sous-chaîne, sans casse ni
accents) est une fonction pure du script (`cle_recherche_modele`/`modele_correspond_filtre`), extraite du
SOURCE par AST et exécutée hors du module complet — importer `packaging/acvram-gui` charge `gi`/GTK3/
WebKit2 (pas installés partout, voir tests/test_langues_gui.py) alors que ce filtre n'en a besoin nulle
part. À sec, sans GTK."""
import ast
import pathlib
import unicodedata

RACINE = pathlib.Path(__file__).resolve().parent.parent
GUI = RACINE / "packaging" / "acvram-gui"
FONCTIONS = ("cle_recherche_modele", "modele_correspond_filtre")


def _fonctions_pures() -> dict:
    """Compile SEULEMENT les deux fonctions nommées, sans exécuter le reste du script (pas de `import gi`)."""
    arbre = ast.parse(GUI.read_text(encoding="utf-8"))
    trouvees = {n.name: n for n in arbre.body if isinstance(n, ast.FunctionDef) and n.name in FONCTIONS}
    assert set(trouvees) == set(FONCTIONS), sorted(trouvees)
    module = ast.Module(body=list(trouvées := trouvees.values()), type_ignores=[])
    ns = {"unicodedata": unicodedata}
    exec(compile(module, str(GUI), "exec"), ns)
    return ns


def test_sous_chaine_insensible_a_la_casse_et_aux_accents():
    ns = _fonctions_pures()
    correspond = ns["modele_correspond_filtre"]
    assert correspond("GLM-4.7-Flash-nvfp4", "glm")
    assert correspond("Qwen3.6-35B-awq", "QWEN3.6")
    assert correspond("gemma-4-31b-it-nvfp4-vision", "gémma")            # accent en trop côté requête
    assert correspond("Nemotron-3.5-Lightning", "eclair") is False       # pas de traduction, juste sous-chaîne
    assert correspond("Nemotron-3.5-Lightning", "lightning")
    assert correspond("Kimi-Linéaire-preview", "lineaire")               # accent en moins côté requête


def test_requete_vide_correspond_a_tout():
    ns = _fonctions_pures()
    assert ns["modele_correspond_filtre"]("n'importe-quel-alias", "")


def test_pas_de_correspondance_hors_sous_chaine():
    ns = _fonctions_pures()
    assert ns["modele_correspond_filtre"]("Coder-30B-A3B-awq", "devstral") is False


def test_cle_recherche_egale_des_deux_cotes_de_la_comparaison():
    """La clé JS (_cleRecherche, page_accueil) applique la même normalisation : NFKD, sans combinants,
    casse neutralisée — vérifié ici côté Python, seul côté testable sans navigateur."""
    ns = _fonctions_pures()
    cle = ns["cle_recherche_modele"]
    assert cle("Éléments-Numériques") == cle("elements-numeriques".upper())
