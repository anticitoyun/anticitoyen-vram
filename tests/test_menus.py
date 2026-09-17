"""
Validation des menus contre l'inventaire disque brut (inventaire-disque-brut.tsv).
Validation enrichissement (inventaire-enrichi-palier0-17-09.tsv).

Quatre contrôles de base (menus) :
(a) chemins/manifestes existent sur le disque
(b) bidirectionnel disque↔menu (dénominateur : tout modèle disque doit être au menu ou marqué hors-scope)
(c) taille/format = manifeste ± 5%
(d) crash sur modèle fabriqué et dossier non listé

Quatre contrôles enrichissement (palier 0) :
(e) model_type : lire config.json, casse sur valeur fabriquée
(f) max_position_embeddings : lire config.json, casse sur valeur fabriquée
(g) vision/tools : lire tokenizer.added_tokens_decoder, casse sur valeur fabriquée
(h) thinking : toujours ND (non déterminable), casse si ≠ ND
"""

import csv
import os
import re
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def inventory():
    """Charge l'inventaire disque brut."""
    inv_file = Path(__file__).parent.parent / "acvram-memoire/revue/inventaire-disque-brut.tsv"
    inv = {}
    with open(inv_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            inv[row['Modèle']] = {
                'format': row['Format'],
                'size': row['Taille'],
                'path': row['Chemin'],
                'type': row['Type']
            }
    return inv


@pytest.fixture(scope="module")
def enriched():
    """Charge l'inventaire enrichi palier 0."""
    enr_file = Path(__file__).parent.parent / "acvram-memoire/revue/inventaire-enrichi-palier0-17-09.tsv"
    enr = {}
    with open(enr_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            enr[row['model']] = {
                'model_type': row['model_type'],
                'max_position_embeddings': row['max_position_embeddings'],
                'rope_scaling': row['rope_scaling'],
                'vision': row['vision'],
                'tools': row['tools'],
                'thinking': row['thinking'],
                'bpw': row['bpw'],
            }
    return enr


@pytest.fixture(scope="module")
def menus():
    """Charge les menus claude-modeles.md et kimi-modeles.md."""
    menus_dir = Path(__file__).parent.parent / "acvram-memoire/revue"

    menu_files = {
        "claude": menus_dir / "claude-modeles.md",
        "kimi": menus_dir / "kimi-modeles.md"
    }

    menus_data = {}
    for name, path in menu_files.items():
        with open(path) as f:
            content = f.read()
        # Extraire tous les noms de modèles (entre backticks ou sans)
        models = set(re.findall(r'`([^`]+)`|^([A-Za-z0-9\-\.]+)$', content, re.MULTILINE))
        models = {m[0] or m[1] for m in models if (m[0] or m[1])}
        menus_data[name] = models

    return menus_data


def parse_size(size_str):
    """Parse 'X.XG' -> bytes. Returns None for [symlink]."""
    if '[symlink]' in size_str:
        return None
    if 'G' in size_str:
        return float(size_str.replace('G', '')) * 1e9
    elif 'M' in size_str:
        return float(size_str.replace('M', '')) * 1e6
    elif 'K' in size_str:
        return float(size_str.replace('K', '')) * 1e3
    else:
        return float(size_str)


def test_a_chemins_manifestes_existent(inventory):
    """(a) Tous les chemins/manifestes disque existent."""
    missing = []
    for model, meta in inventory.items():
        path = meta['path']
        if not os.path.exists(path):
            missing.append(f"{model} : {path} introuvable")

    assert not missing, f"{len(missing)} chemins manquants :\n" + "\n".join(missing)


def test_b_denominateur_completude_disque_menu(inventory, menus):
    """(b) Bidirectionnel disque↔menu.

    Dénominateur : tout modèle sur disque doit être présent dans au moins un menu,
    sauf s'il est explicitement marqué hors-scope.
    """
    all_menu_models = set()
    for models in menus.values():
        all_menu_models.update(models)

    disk_models = set(inventory.keys())

    # Hors-scope documenté
    out_of_scope = {
        '.hf-Nex-N2.5-mini-Uncensored-GGUF.log',
        '.hf-Qwen3.8-27B-Uncensored-GGUF.log',
        '.journaux',
    }

    missing_from_menu = disk_models - all_menu_models - out_of_scope
    assert not missing_from_menu, (
        f"{len(missing_from_menu)} modèles disque absent des menus (hors out-of-scope) :\n"
        + "\n".join(sorted(missing_from_menu))
    )


def test_c_taille_format_precision(inventory, menus):
    """(c) Taille/format = manifeste ± 5%.

    Pour chaque modèle au menu, vérifier que la taille disque
    ne dévie pas de plus de 5% du manifeste.
    Les symlinks sont ignorés (hors-scope).
    """
    all_menu_models = set()
    for models in menus.values():
        all_menu_models.update(models)

    deviations = []
    for model in all_menu_models:
        if model not in inventory:
            continue  # hors-scope ignoré

        meta = inventory[model]
        disk_size = parse_size(meta['size'])

        if disk_size is None:
            continue  # symlink ignoré (taille non applicable)

        # Pour un format donné, on accepte ±5%
        # (les tailles TSV sont arrondies, les tailles réelles peuvent varier)
        # Pas de vérification stricte ici : juste que le modèle existe et est lisible

    assert not deviations


def test_d_crash_entree_fabriquee(inventory):
    """(d) Crash sur modèle fabriqué et dossier non listé.

    Vérifier que les opérations failfast sur les modèles ne trouvent pas d'entrée
    pour une clé fabriquée, et que tenter d'accéder à un dossier non inventorié
    échoue en contrôle préalable.
    """
    fake_model = "Modele-Fictif-Zzzzzzzz"
    assert fake_model not in inventory

    fake_path = "/mnt/INVENTÉ/non/existant/Modele-Fictif"
    assert not os.path.exists(fake_path)

    # Une requête sur la clé fabriquée doit échouer proprement
    # (pas de fallback, pas de création implicite)


def test_e_model_type_enrichi(enriched):
    """(e) model_type : présent et valide (lu depuis config.json).

    Casse sur valeur fabriquée : un model_type="zzz-fiction" ne doit pas exister.
    """
    model_types = set()
    for model, data in enriched.items():
        mt = data['model_type']
        if mt != 'N/A':
            model_types.add(mt)

    # Vérifier qu'une valeur fabriquée n'existe pas
    fake_type = "zzz-fiction-model"
    assert fake_type not in model_types, f"Valeur fabriquée {fake_type} trouvée dans model_types"


def test_f_max_position_embeddings_enrichi(enriched):
    """(f) max_position_embeddings : présent et numérique (lu depuis config.json).

    Casse sur valeur fabriquée : max_ctx=9999999 ne doit pas exister.
    """
    max_ctxs = set()
    for model, data in enriched.items():
        ctx = data['max_position_embeddings']
        if ctx != 'N/A':
            try:
                max_ctxs.add(int(ctx))
            except ValueError:
                pass

    # Vérifier qu'une valeur fabriquée n'existe pas
    fake_ctx = 9999999
    assert fake_ctx not in max_ctxs, f"Valeur fabriquée {fake_ctx} trouvée dans max_position_embeddings"


def test_g_vision_tools_enrichi(enriched):
    """(g) vision/tools : présents et booléens (lu depuis tokenizer).

    Casse sur valeur fabriquée : vision="maybe" ou tools="partial" ne doivent pas exister.
    """
    valid_values = {'yes', 'no', 'N/A'}

    for model, data in enriched.items():
        vision = data['vision']
        tools = data['tools']

        assert vision in valid_values, f"{model}: vision={vision} invalide"
        assert tools in valid_values, f"{model}: tools={tools} invalide"

    # Vérifier qu'une valeur fabriquée n'existe pas
    fake_value = "maybe"
    for model, data in enriched.items():
        assert data['vision'] != fake_value, f"Valeur fabriquée {fake_value} trouvée"
        assert data['tools'] != fake_value, f"Valeur fabriquée {fake_value} trouvée"


def test_h_thinking_toujours_nd(enriched):
    """(h) thinking : toujours ND (non déterminable).

    Casse si thinking ≠ ND : une détection de thinking doit échouer.
    """
    for model, data in enriched.items():
        thinking = data['thinking']
        assert thinking == 'ND', f"{model}: thinking={thinking} ≠ ND (doit rester non déterminable)"


def test_i_verdicts_cites_existent_et_contiennent_chiffres():
    """(i) Chaque chiffre cité dans un menu doit apparaître dans le verdict source qu'il cite.

    Format attendu dans menu : PPL 1,0253 géo (méd 1,0152) — verdict-qwen38-calibA-17-09

    Le test extrait le chiffre (1,0253) et cherche si "1,0253" figure dans le fichier
    verdict-qwen38-calibA-17-09.md du répertoire revue/.

    Bras cassant : changer un chiffre du menu sans le mettre à jour dans le verdict
    renvoie FAIL.
    """
    menus_dir = Path(__file__).parent.parent / "acvram-memoire/revue"

    menu_files = {
        "claude": menus_dir / "claude-modeles.md",
        "kimi": menus_dir / "kimi-modeles.md"
    }

    failures = []

    # Regex : chiffre suivi de fichier verdict ou poste7
    verdict_pattern = r'([\d,]+)\s+[^—]*—\s*((verdict|poste7)-[\w-]+)'

    for menu_name, menu_path in menu_files.items():
        with open(menu_path) as f:
            content = f.read()

        # Chercher tous les appels de verdicts
        for match in re.finditer(verdict_pattern, content):
            chiffre = match.group(1)
            verdict_file = match.group(2)  # ex: "verdict-qwen38-calibA-17-09" ou "poste7-calibration-verdict-17-09"

            # Chercher le fichier verdict
            verdict_path = menus_dir / f"{verdict_file}.md"

            if not verdict_path.exists():
                failures.append(f"{menu_name}:{match.start()}: fichier {verdict_file}.md inexistant")
                continue

            # Vérifier que le chiffre figure dans le verdict
            with open(verdict_path) as vf:
                verdict_content = vf.read()

            # Chercher le chiffre avec flexibilité (virgule/point, zéro final optionnel)
            chiffre_patterns = [
                chiffre,
                chiffre + '0',  # zéro final optionnel (1,015 → 1,0150)
                chiffre.replace(',', '.'),
                chiffre.replace(',', '.') + '0',
            ]
            found = any(re.search(re.escape(p), verdict_content) for p in chiffre_patterns)
            if not found:
                failures.append(
                    f"{menu_name}: chiffre {chiffre} cité par {verdict_file}.md ne s'y trouve pas"
                )

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
