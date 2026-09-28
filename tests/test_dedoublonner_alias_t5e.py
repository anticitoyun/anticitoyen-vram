"""t5e (poste1, 27/09, revue/poste1-t5e-verdict-27-09.md §11) : les menus
claude-modeles/kimi-modeles montraient deux alias pour un seul modèle
converti (« …-srci1q4km-… » / « …-srcq4-k-m-… », même dossier, même sha256)
— renommer-convertis.py ajoute un alias neuf à chaque renommage sans jamais
retirer l'ancien (voulu, pour la compatibilité des scripts existants), mais
rien ne les dédoublonnait ensuite. Ce test prouve : (1) un vrai doublon
(même dossier) est réduit à un seul alias, celui que produirait AUJOURD'HUI
renommer-convertis.py ; (2) deux alias à dossiers DIFFÉRENTS ne sont jamais
fusionnés ; (3) config.toml perd le bloc [models.<alias retiré>] sans que le
reste du fichier (commentaires compris) ne bouge ; (4) l'essai à blanc ne
touche à rien.

`outils/renommer-convertis.py` n'est pas importé ici : c'est un script (pas
un module), il scanne le vrai parc de modèles dès l'import. La formule
« acvram-<basename slugifié> » qu'il écrit pour un alias neuf (sa ligne
`na = …`) est dupliquée dans `dedoublonner-alias.py::canonique` — les deux
DOIVENT rester identiques ; c'est ce que ce fichier documente, pas un
import croisé risqué."""
import importlib.util
import re
from pathlib import Path

DEPOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("dedoublonner_alias", DEPOT / "outils" / "dedoublonner-alias.py")
dedoublonner_alias = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dedoublonner_alias)


TSV = """acvram-huihui-src-i1-q4km\t/mnt/x/huihui-8B-nvfp4\t32768
acvram-huihui-8b-nvfp4\t/mnt/x/huihui-8B-nvfp4\t32768
acvram-solo-13b\t/mnt/x/solo-13b\t32768
acvram-solo-13b-b\t/mnt/x/solo-13b-b\t32768
"""


def _ecrire(tmp_path, texte_tsv, texte_toml=""):
    tsv = tmp_path / "acvram-chemins.tsv"
    tsv.write_text(texte_tsv)
    config = tmp_path / "config.toml"
    config.write_text(texte_toml)
    return tsv, config


def test_meme_dossier_reduit_a_un_seul_alias(tmp_path):
    tsv, _ = _ecrire(tmp_path, TSV)
    _, entrees = dedoublonner_alias.lire_tsv(tsv)
    plan = dedoublonner_alias.plan_doublons(entrees)
    assert plan == {"acvram-huihui-8b-nvfp4": ["acvram-huihui-src-i1-q4km"]}


def test_dossiers_differents_jamais_fusionnes(tmp_path):
    tsv, _ = _ecrire(tmp_path, TSV)
    _, entrees = dedoublonner_alias.lire_tsv(tsv)
    plan = dedoublonner_alias.plan_doublons(entrees)
    retires = {a for v in plan.values() for a in v}
    assert "acvram-solo-13b" not in retires
    assert "acvram-solo-13b-b" not in retires


def test_canonique_suit_la_formule_de_renommer_convertis():
    # même construction que la ligne "na = " de renommer-convertis.py (118) :
    # acvram- + slug du basename du dossier renommé. Formule dupliquée à
    # dessein (voir docstring) : ce test la fige.
    dossier = "/mnt/x/huihui-8B-nvfp4"
    attendu = "acvram-" + re.sub(r"[^a-z0-9]+", "-", "huihui-8B-nvfp4".lower()).strip("-")
    assert dedoublonner_alias.canonique(dossier) == attendu


def test_essai_a_blanc_ne_touche_a_rien(tmp_path):
    tsv, config = _ecrire(tmp_path, TSV, "[models.acvram-huihui-8b-nvfp4]\nprovider = \"acvram\"\n")
    avant_tsv, avant_config = tsv.read_text(), config.read_text()
    total, plan = dedoublonner_alias.executer(tsv, config, appliquer=False,
                                              journal=tmp_path / "journal.tsv")
    assert total == 1
    assert tsv.read_text() == avant_tsv
    assert config.read_text() == avant_config
    assert (tmp_path / "journal.tsv").exists()


def test_appliquer_retire_du_tsv_et_du_toml_sans_toucher_au_reste(tmp_path):
    toml_avant = (
        "# fiche d'un modele quelconque, ecrite a la main\n"
        "[models.acvram-huihui-src-i1-q4km]\n"
        "provider = \"acvram\"\n"
        "model = \"huihui\"\n"
        "\n"
        "[models.acvram-solo-13b]\n"
        "provider = \"acvram\"\n"
    )
    tsv, config = _ecrire(tmp_path, TSV, toml_avant)
    total, plan = dedoublonner_alias.executer(tsv, config, appliquer=True,
                                              journal=tmp_path / "journal.tsv")
    assert total == 1
    lignes = tsv.read_text().splitlines()
    alias_restants = {l.split("\t")[0] for l in lignes}
    assert "acvram-huihui-src-i1-q4km" not in alias_restants
    assert "acvram-huihui-8b-nvfp4" in alias_restants          # gardé
    assert "acvram-solo-13b" in alias_restants           # dossier distinct, intact
    assert "acvram-solo-13b-b" in alias_restants         # dossier distinct, intact

    toml_apres = config.read_text()
    assert "[models.acvram-huihui-src-i1-q4km]" not in toml_apres
    assert "[models.acvram-solo-13b]" in toml_apres      # non concerné, intact
    assert "# fiche d'un modele quelconque, ecrite a la main" in toml_apres  # commentaire intact
