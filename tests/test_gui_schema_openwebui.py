"""Schéma Open WebUI de l'onglet « Faire agir le modèle » (01/10, utilisatrice) : la 4e boîte, « modèle (carte) »,
sortait du viewBox (x=568 + 150 = 718 > 700) et les étiquettes de flèche chevauchaient les boîtes. La fonction est
compilée seule depuis la SOURCE par AST, comme test_gui_filtre_modele (importer le script charge GTK)."""
import ast
import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parent.parent
GUI = RACINE / "packaging" / "acvram-gui"


def _svg() -> str:
    arbre = ast.parse(GUI.read_text(encoding="utf-8"))
    fn = [n for n in arbre.body if isinstance(n, ast.FunctionDef) and n.name == "svg_schema_openwebui"]
    module = ast.Module(body=fn, type_ignores=[])
    ns = {"T": lambda s, **kw: s, "PORT_DEFAUT": 8765}
    exec(compile(module, str(GUI), "exec"), ns)
    return ns["svg_schema_openwebui"]()


def test_boites_dans_le_viewbox():
    svg = _svg()
    larg = float(re.search(r'viewBox="0 0 ([\d.]+) ', svg).group(1))
    rects = [(float(x), float(w)) for x, w in re.findall(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', svg)]
    assert len(rects) == 4
    for x, w in rects:
        assert x >= 0 and x + w <= larg, f"boîte {x}+{w} hors du viewBox {larg}"


def test_etiquettes_de_fleche_au_dessus_des_boites():
    svg = _svg()
    haut = min(float(y) for y in re.findall(r'<rect x="[\d.]+" y="([\d.]+)"', svg))
    lignes = re.findall(r'<line[^>]*/><text x="[\d.]+" y="([\d.]+)"[^>]*font-size="(\d+)"', svg)
    assert len(lignes) == 3
    for y, taille in lignes:
        # ligne de base + jambage : le texte doit finir avant le haut des boîtes
        assert float(y) + 0.3 * float(taille) < haut, f"étiquette y={y} chevauche les boîtes (haut {haut})"


def test_sous_titre_le_plus_long_tient_dans_sa_boite():
    svg = _svg()
    w = min(float(w) for w in re.findall(r'<rect x="[\d.]+" y="[\d.]+" width="([\d.]+)"', svg))
    # ≈ 0,6 em par caractère à 11 px : borne large, pour une police proportionnelle
    assert len("Réglages → Connexions → OpenAI") * 11 * 0.6 <= w
