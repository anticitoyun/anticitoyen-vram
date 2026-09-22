"""Les menus de modèles (revue/claude-modeles.md par moteur, kimi-modeles.md
par usage) contre le DISQUE, pas contre l'inventaire écrit à la main.

Sage, plan de complétion § 4, quatre contrôles : (a) chaque entrée de menu
existe et son manifeste se lit ; (b) les deux sens — tout dossier des racines
indexées est au menu, toute entrée du menu est sur disque ; (c) format et
taille = manifeste et `du` ± 5 % ; (d) le test casse sur une entrée fabriquée
et sur un dossier ajouté non listé — prouvé ici même dans un `tmp_path`, avec
un jumeau cohérent qui doit passer.

Palier 0 (Katy, `inventaire-enrichi-palier0-17-09.tsv`), quatre contrôles
supplémentaires, chacun casse sur une valeur fabriquée : (e) model_type lu
depuis config.json ; (f) max_position_embeddings lu depuis config.json ; (g)
vision/tools lus depuis tokenizer.added_tokens_decoder ; (h) thinking, toujours
ND (non déterminable sans exécution) — casse si autre chose que ND apparaît.

Le 17/09 la première version (63ace69) était verte avec une entrée fabriquée,
un dossier non listé et une taille ×12 : `deviations` jamais rempli, (d)
assertant qu'un nom inventé n'est pas dans un dict, dénominateur = le TSV
(verdict-menus-tests-relecture-laurine-17-09). Ici le dénominateur est
`os.scandir` des racines ; le TSV n'apporte que l'emplacement des racines et
est lui-même contrôlé contre le disque. Les racines sont lues dans le TSV, pas
codées ici ; ailleurs que sur le poste, `skip`.
"""
import csv
import functools
import json
import os
import sys
import re
import subprocess
from pathlib import Path

import pytest

REVUE = Path(__file__).resolve().parent.parent / "acvram-memoire" / "revue"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outils"))
from racine_modeles import racine_modeles  # noqa: E402
MENU_MOTEURS = REVUE / "claude-modeles.md"
MENU_USAGES = REVUE / "kimi-modeles.md"
INVENTAIRE = REVUE / "inventaire-disque-brut.tsv"

# Une entrée de menu, c'est une PUCE : « - `nom` (taille) — commentaire ».
# Un nom cité en prose n'est pas une entrée.
_PUCE = re.compile(r"^- `([^`\s]+)`(?:\s+\(([0-9.]+)([KMGT]?)\))?(.*)$")
_TITRE = re.compile(r"^(#{2,3}) (.+)$")
_UNITES = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
# Titre de section du menu moteurs → racine indexée (nom du dossier).
SECTIONS = {"acvram": "models_acvram", "vLLM": "models_vllm",
            "GGUF": "models_gguf", "EXL3": "models_exl3"}
_DENSES = {"bf16", "fp16", "fp32"}
TOLERANCE = 0.05


def lire_menu(texte, doublons=False):
    """{nom: (taille_octets|None, racine|None, ligne)} — racine d'après le
    dernier titre `## … Moteur : X` ; None dans un menu par usage, où un
    modèle peut servir deux usages (`doublons`), à taille égale."""
    entrees, racine = {}, None
    for ligne in texte.splitlines():
        t = _TITRE.match(ligne)
        if t:
            if t.group(1) == "##":
                racine = next((r for cle, r in SECTIONS.items() if cle in t.group(2)), None)
            continue
        p = _PUCE.match(ligne)
        if not p:
            continue
        nom, val, unite, reste = p.groups()
        taille = float(val) * _UNITES[unite] if val else None
        if nom in entrees and not (doublons and entrees[nom][0] == taille):
            raise ValueError(f"entrée en double au menu : {nom}")
        entrees[nom] = (taille, racine, reste)
    return entrees


def racines_du_tsv(chemin_tsv):
    """{nom_de_dossier: Path} — l'inventaire ne sert qu'à dire OÙ sont les
    racines ; ce qu'elles contiennent se lit sur le disque."""
    with open(chemin_tsv, newline="") as f:
        lignes = list(csv.DictReader(f, delimiter="\t"))
    # Le parc acvram change de disque (980 PRO USB → SATA CMR → AI_GENERATOR) : sa
    # racine du moment est racine_modeles(), le TSV ne garde que le nom du dossier
    # (T4 20/09 : 16 + 95 « écarts » n'étaient que l'ancien préfixe).
    for r in lignes:
        p = Path(r["Chemin"])
        if p.parent.name == "models_acvram":
            r["Chemin"] = str(Path(racine_modeles()) / p.name)
    racines = {}
    for r in lignes:
        p = Path(r["Chemin"]).parent
        racines.setdefault(p.name, set()).add(p)
    doubles = {k: v for k, v in racines.items() if len(v) > 1}
    assert not doubles, f"une racine sous deux chemins dans le TSV : {doubles}"
    return {k: next(iter(v)) for k, v in racines.items()}, lignes


def lister_disque(racines):
    """{nom: Path} de TOUT ce que contiennent les racines — dossiers, fichiers,
    liens, cachés : c'est le dénominateur."""
    disque = {}
    for racine in racines.values():
        for e in os.scandir(racine):
            if e.name in disque:
                raise ValueError(f"{e.name} présent sous deux racines : {disque[e.name]} et {e.path}")
            disque[e.name] = Path(e.path)
    return disque


# REGLES § 4, trois états d'un alias acvram (Sage 12 h 30, sage-t4-tri-69-20-09) : « présent » (dossier sous
# racine_modeles()), « absent » (nulle part : effacé ou disque non monté), « relocalisé hors racine » (retrouvé
# sous une autre racine connue — transfert USB → nvme3 en cours). Absent et relocalisé sont des ÉTATS du menu,
# pas des rouges (48 alias le 20/09) ; rouge seulement si un alias présent ne résout pas, ou résout hors racine.
# Même recherche que ~/.local/bin/modeles-a-jour (RACINES, profondeur 4, deux derniers composants puis le dernier).
RACINES_CONNUES = ["/mnt/AI_GENERATOR/models_acvram", "/mnt/2TO_2023_980PRO/Modeles",
                   "/mnt/4TO_SATACMR_2022/Modeles", "/mnt/2TO_SSD_2025_IA"]
PROFONDEUR = 4


@functools.lru_cache(maxsize=None)
def _index_racines():
    un = {}
    for racine in RACINES_CONNUES:
        if not os.path.isdir(racine):
            continue
        base = racine.rstrip("/").count("/")
        for dossier, sous, fichiers in os.walk(racine):
            if dossier.count("/") - base >= PROFONDEUR:
                sous[:] = []
            for nom in sous + fichiers:
                un.setdefault(nom, []).append(os.path.join(dossier, nom))
    return un


def etat_alias(nom, racine):
    """'present' | 'absent' | 'relocalise' (hors racine) | 'hors-racine' (présent mais résout ailleurs = rouge)."""
    chemin = Path(racine) / nom
    if chemin.exists() or chemin.is_symlink():
        if chemin.is_symlink() and not str(chemin.resolve()).startswith(str(Path(racine).resolve()) + "/"):
            return "hors-racine"
        return "present"
    ailleurs = [c for c in _index_racines().get(nom, []) if not c.startswith(str(racine) + "/")]
    return "relocalise" if ailleurs else "absent"


def alias_absents(menu, racines):
    """Les alias acvram du menu dont l'état est absent ou relocalisé : exclus des contrôles a/b/l/e."""
    r = racines.get("models_acvram")
    return frozenset(n for n, (_, racine, _) in menu.items()
                     if racine == "models_acvram" and r is not None and etat_alias(n, r) in ("absent", "relocalise"))


def formats_manifeste(dossier):
    with open(dossier / "acvram_manifest.json") as f:
        tenseurs = json.load(f)["tensors"]
    return {v.get("format", "?") for v in tenseurs.values()}


def tailles_du(chemins):
    """`du -sk` en un seul processus, sans suivre les liens : {Path: octets}."""
    if not chemins:
        return {}
    sortie = subprocess.run(["du", "-sk", "--", *map(str, chemins)],
                            capture_output=True, text=True, check=True).stdout
    tailles = {}
    for ligne in sortie.splitlines():
        ko, chemin = ligne.split("\t", 1)
        tailles[Path(chemin)] = int(ko) * 1024
    return tailles


# ---- les contrôles, sous forme de fonctions qui RENDENT une liste de fautes,
# ---- pour que (d) puisse les exercer sur un disque fabriqué.

def controle_a(menu, racines, absents=frozenset()):
    """(a) chaque entrée existe SOUS LA RACINE DE SA SECTION ; un converti
    acvram a un manifeste lisible avec des tenseurs ; un alias acvram absent ou
    relocalisé (état § 4) est ignoré, un alias qui résout hors racine est une faute."""
    fautes = []
    for nom, (_, racine, _) in menu.items():
        if racine is None:
            fautes.append(f"{nom} : hors de toute section « Moteur : … »")
            continue
        if nom in absents:
            continue
        chemin = racines[racine] / nom
        if racine == "models_acvram" and etat_alias(nom, racines[racine]) == "hors-racine":
            fautes.append(f"{nom} : résout hors de racine_modeles() ({chemin.resolve()})")
            continue
        if not chemin.exists() and not chemin.is_symlink():
            fautes.append(f"{nom} : absent de {racines[racine]}")
            continue
        if racine == "models_acvram":
            try:
                if not formats_manifeste(chemin):
                    fautes.append(f"{nom} : manifeste sans tenseur")
            except (OSError, KeyError, ValueError) as e:
                fautes.append(f"{nom} : manifeste illisible ({type(e).__name__})")
    return fautes


TSV_DIR = Path(os.environ.get("ACVRAM_TSV_DIR", Path.home() / "TSV"))


@functools.lru_cache(maxsize=None)
def alias_servis():
    """{nom de dossier} présent en 2e colonne d'un TSV servi (acvram/gguf/vllm-chemins) —
    dénominateur de « alias servi », distinct du nom d'alias (1re colonne, préfixé/en
    minuscule) et de l'inventaire enrichi de Katy (REGLES : ce dernier catalogue tout le
    disque, celui-ci ne catalogue que ce qu'un lanceur sert réellement). Pour acvram-chemins.tsv
    seulement : un dossier sans `acvram_manifest.json` est une SOURCE brute (HF, bf16 externe),
    jamais un alias servi, même si une ligne de TSV pointe dessus (Maîtresse, 21/09 — mesuré sur
    gemma-4-12B-it-bf16 : `model.safetensors` HF, aucun manifeste acvram)."""
    noms = set()
    for tsv in TSV_DIR.glob("*-chemins.tsv"):
        exige_manifeste = tsv.name == "acvram-chemins.tsv"
        with open(tsv, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", fieldnames=["alias", "chemin", "ctx", "_", "_2", "regime"]):
                if not r["chemin"]:
                    continue
                chemin = Path(r["chemin"])
                if exige_manifeste and not (chemin / "acvram_manifest.json").is_file():
                    continue
                noms.add(chemin.name)
    return frozenset(noms)


def temoins_sans_fiche(disque, menu):
    """Un dossier du disque absent du menu ET absent de tout TSV servi = une source brute
    (bf16, ablation) gardée comme témoin de qualité, pas un alias servi — REGLES d'entrée
    au menu : fiche avec verdict qualité ET pas un doublon. Absent de menu ET servi ailleurs
    reste une vraie faute (mesuré 21/09 : 2/8 candidats l'étaient, cf. laure-suite-21-09)."""
    return frozenset(n for n in disque if n not in menu and n not in alias_servis())


def controle_b(menu, disque, absents=frozenset(), temoins=frozenset()):
    """(b) les deux différences, chacune doit être vide (les alias à l'état absent/relocalisé
    et les témoins sans fiche exceptés)."""
    return ([f"sur disque, absent du menu : {n} ({disque[n]})"
             for n in sorted(set(disque) - set(menu) - temoins)]
            + [f"au menu, absent du disque : {n}" for n in sorted(set(menu) - set(disque) - absents)])


def controle_c(menu, racines, tailles):
    """(c) taille du menu = ce que `du -sh` imprime pour le dossier (arrondi
    vers le haut : 13,25 Go s'écrit « 14G ») ou à ± 5 % ; une entrée acvram
    rangée sous NVFP4 sans tenseur nvfp4 doit nommer sur sa ligne son vrai
    format — chaque format quantifié, ou le dense qu'elle contient."""
    fautes = []
    for nom, (taille, racine, reste) in menu.items():
        if racine is None or (racines[racine] / nom).is_symlink():
            continue
        chemin = racines[racine] / nom
        if chemin not in tailles:
            continue                                      # déjà compté par (a) et (b)
        if taille is not None:
            reel = tailles[chemin]
            ecart = abs(reel - taille) / max(reel, 1)
            if ecart > TOLERANCE and _du_h(reel) != _du_h(taille):
                fautes.append(f"{nom} : menu {_h(taille)}, du -sh {_du_h(reel)} ({ecart:.0%})")
        if racine == "models_acvram" and chemin.is_dir():
            try:
                fmts = formats_manifeste(chemin)
            except (OSError, KeyError, ValueError):
                continue                                  # déjà compté par (a)
            if "nvfp4" not in fmts:
                dit = lambda q: re.search(rf"(?<![A-Za-z0-9]){re.escape(q)}(?![0-9])", nom + reste)
                quant = sorted(fmts - _DENSES)
                manquants = [q for q in quant if not dit(q)]
                if not quant and not any(dit(d) for d in fmts & _DENSES):
                    manquants = [sorted(fmts & _DENSES)]
                if manquants:
                    fautes.append(f"{nom} : rangé sous NVFP4 sans tenseur nvfp4, la ligne ne dit pas {manquants}")
    return fautes


def _h(octets):
    for u in ("T", "G", "M", "K"):
        if octets >= _UNITES[u]:
            return f"{octets / _UNITES[u]:.2f}{u}"
    return f"{octets:.0f}"


def _du_h(octets):
    """Le rendu de `du -sh` : unité telle que la valeur soit < 1024, un
    décimal sous 10, entier au-dessus, toujours arrondi vers le haut."""
    import math
    for u in ("T", "G", "M", "K"):
        v = octets / _UNITES[u]
        if v >= 1:
            return (f"{math.ceil(v * 10) / 10:.1f}" if math.ceil(v * 10) < 100 else f"{math.ceil(v)}") + u
    return f"{octets}"


# ---- le poste

@pytest.fixture(scope="module")
def poste():
    racines, lignes = racines_du_tsv(INVENTAIRE)
    presentes = {k: v for k, v in racines.items() if v.is_dir()}
    if not presentes:
        pytest.skip("aucune racine de l'inventaire n'est montée : pas sur le poste")
    manquantes = set(racines) - set(presentes)
    assert not manquantes, f"racines de l'inventaire absentes du disque : {sorted(racines[m] for m in manquantes)}"
    assert set(racines) == set(SECTIONS.values()), (racines.keys(), SECTIONS.values())
    menu = lire_menu(MENU_MOTEURS.read_text())
    usages = lire_menu(MENU_USAGES.read_text(), doublons=True)
    disque = lister_disque(racines)
    tailles = tailles_du(sorted(disque.values()))
    absents = alias_absents(menu, racines) | alias_absents({r["Modèle"]: (None, "models_acvram", "") for r in lignes
                                                            if Path(r["Chemin"]).parent.name == "models_acvram"}, racines)
    etats = {n: etat_alias(n, racines["models_acvram"]) for n in sorted(absents)}
    print(f"[menus] alias acvram à l'état absent/relocalisé (§ 4, exclus des contrôles) : {len(absents)} — "
          + ", ".join(f"{n}:{e}" for n, e in etats.items()))
    temoins = temoins_sans_fiche(disque, menu)
    if temoins:
        print(f"[menus] témoins sans fiche verdict (hors menu, exclus de (b)) : {len(temoins)} — {sorted(temoins)}")
    return racines, lignes, menu, usages, disque, tailles, absents, temoins


def test_a_chaque_entree_existe_sous_sa_racine_et_son_manifeste_se_lit(poste):
    racines, _, menu, _, _, _, absents, _ = poste
    fautes = controle_a(menu, racines, absents=absents)
    assert not fautes, f"{len(fautes)} entrée(s) :\n  " + "\n  ".join(fautes)


def test_b_disque_et_menu_dans_les_deux_sens(poste):
    _, _, menu, _, disque, _, absents, temoins = poste
    fautes = controle_b(menu, disque, absents=absents, temoins=temoins)
    assert not fautes, f"{len(fautes)} écart(s) :\n  " + "\n  ".join(fautes)


def test_c_taille_et_format_suivent_du_et_le_manifeste(poste):
    racines, _, menu, _, _, tailles, _, _ = poste
    fautes = controle_c(menu, racines, tailles)
    assert not fautes, f"{len(fautes)} écart(s) :\n  " + "\n  ".join(fautes)
    sans_taille = [n for n, (t, r, _) in menu.items() if t is None and not (racines[r] / n).is_symlink()]
    assert not sans_taille, f"entrées sans taille au menu : {sans_taille}"


def test_le_menu_par_usage_est_un_sous_ensemble_du_menu_par_moteur(poste):
    _, _, menu, usages, _, _, _, _ = poste
    inconnus = sorted(set(usages) - set(menu))
    assert not inconnus, f"au menu par usage, absents du menu par moteur : {inconnus}"
    tailles = [(n, usages[n][0], menu[n][0]) for n in usages
               if usages[n][0] is not None and usages[n][0] != menu[n][0]]
    assert not tailles, f"tailles différentes entre les deux menus : {tailles}"


def test_l_inventaire_tsv_suit_le_disque(poste):
    """Le TSV est une photo à la main : il doit dire ce que le disque dit,
    sinon on le retire."""
    racines, lignes, _, _, disque, tailles, absents, temoins = poste
    tsv = {r["Modèle"]: r for r in lignes}
    fautes = controle_b(tsv, disque, absents=absents, temoins=temoins)
    for nom, r in tsv.items():
        if nom in disque and Path(r["Chemin"]) != disque[nom]:
            fautes.append(f"{nom} : TSV {r['Chemin']}, disque {disque[nom]}")
    assert not fautes, f"{len(fautes)} écart(s) TSV/disque :\n  " + "\n  ".join(fautes)


# ---- (d) : la preuve que les contrôles savent dire « faux »

def _disque_fabrique(tmp_path):
    """Deux racines, trois dossiers, dont un converti acvram avec manifeste."""
    racines = {r: tmp_path / r for r in SECTIONS.values()}
    for r in racines.values():
        r.mkdir()
    (racines["models_gguf"] / "Un-Q4_K_M").mkdir()
    (racines["models_gguf"] / "Un-Q4_K_M" / "poids.gguf").write_bytes(b"\0" * (300 * 1024))
    conv = racines["models_acvram"] / "Deux-nvfp4"
    conv.mkdir()
    (conv / "acvram_manifest.json").write_text(json.dumps(
        {"tensors": {"a": {"format": "nvfp4"}, "b": {"format": "bf16"}}}))
    (conv / "poids.safetensors").write_bytes(b"\0" * (200 * 1024))
    temoin = racines["models_acvram"] / "Trois-temoin"
    temoin.mkdir()
    (temoin / "acvram_manifest.json").write_text(json.dumps(
        {"tensors": {"a": {"format": "int8"}, "b": {"format": "bf16"}}}))
    (temoin / "poids.safetensors").write_bytes(b"\0" * (100 * 1024))
    return racines


_MENU_COHERENT = """## 1. Moteur : acvram (NVFP4)
- `Deux-nvfp4` (200K)
- `Trois-temoin` (100K) — témoin int8, expert partagé
## 3. Moteur : GGUF
- `Un-Q4_K_M` (300K)
"""


def _fautes(racines, texte):
    menu = lire_menu(texte)
    disque = lister_disque(racines)
    tailles = tailles_du(sorted(disque.values()))
    return controle_a(menu, racines), controle_b(menu, disque), controle_c(menu, racines, tailles)


def test_d_le_jumeau_coherent_passe(tmp_path):
    """Sans lui, « les contrôles trouvent des fautes » ne se distingue pas de
    « les contrôles trouvent toujours des fautes »."""
    assert _fautes(_disque_fabrique(tmp_path), _MENU_COHERENT) == ([], [], [])


def test_d_une_entree_fabriquee_au_menu_casse_a_et_b(tmp_path):
    racines = _disque_fabrique(tmp_path)
    a, b, _ = _fautes(racines, _MENU_COHERENT + "- `Modele-Fictif-Zzzz` (9.9G)\n")
    assert any("Modele-Fictif-Zzzz : absent de" in f for f in a), a
    assert b == ["au menu, absent du disque : Modele-Fictif-Zzzz"], b


def test_d_un_dossier_ajoute_non_liste_casse_b(tmp_path):
    racines = _disque_fabrique(tmp_path)
    (racines["models_exl3"] / "Dossier-Non-Liste").mkdir()
    _, b, _ = _fautes(racines, _MENU_COHERENT)
    assert b == [f"sur disque, absent du menu : Dossier-Non-Liste ({racines['models_exl3'] / 'Dossier-Non-Liste'})"], b


def test_d_un_nom_cite_en_prose_n_est_pas_une_entree(tmp_path):
    racines = _disque_fabrique(tmp_path)
    texte = _MENU_COHERENT.replace("- `Un-Q4_K_M` (300K)\n", "Le modèle `Un-Q4_K_M` a été retiré.\n")
    _, b, _ = _fautes(racines, texte)
    assert b == ["sur disque, absent du menu : Un-Q4_K_M (" + str(racines["models_gguf"] / "Un-Q4_K_M") + ")"], b


def test_d_une_taille_fausse_et_un_format_tu_cassent_c(tmp_path):
    racines = _disque_fabrique(tmp_path)
    texte = _MENU_COHERENT.replace("(300K)", "(360K)").replace(" — témoin int8, expert partagé", "")
    _, _, c = _fautes(racines, texte)
    assert sorted(c) == ["Trois-temoin : rangé sous NVFP4 sans tenseur nvfp4, la ligne ne dit pas ['int8']",
                         "Un-Q4_K_M : menu 360.00K, du -sh 300K (20%)"], c
    _, _, c = _fautes(racines, _MENU_COHERENT.replace("(300K)", "(312K)"))
    assert c == [], f"4 % doit passer : {c}"


def test_d_du_h_arrondit_vers_le_haut_comme_du():
    assert _du_h(13.25 * 1024 ** 3) == "14G" and _du_h(7.85 * 1024 ** 3) == "7.9G"
    assert _du_h(584 * 1024 ** 2) == "584M" and _du_h(68 * 1024) == "68K"
    assert _du_h(13.25 * 1024 ** 3) == subprocess.run(
        ["numfmt", "--to=iec", "--round=up", str(int(13.25 * 1024 ** 3))],
        capture_output=True, text=True).stdout.strip()


def test_d_un_manifeste_illisible_casse_a(tmp_path):
    racines = _disque_fabrique(tmp_path)
    (racines["models_acvram"] / "Deux-nvfp4" / "acvram_manifest.json").write_text("{")
    a, _, _ = _fautes(racines, _MENU_COHERENT)
    assert a == ["Deux-nvfp4 : manifeste illisible (JSONDecodeError)"], a


# ---- palier 0 (Katy 24feac3, relu REGLES § 7) : le TSV enrichi doit SUIVRE LES
# ---- FICHIERS — chaque ligne est recalculée depuis le disque par la même règle
# ---- que le script qui l'a produite (outils/enrichir-inventaire-17-09.py)

def _enrichir():
    import importlib.util
    chemin = Path(__file__).parent.parent / "outils" / "enrichir-inventaire-17-09.py"
    spec = importlib.util.spec_from_file_location("enrichir_inventaire", chemin)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def enrichi_brut():
    with open(REVUE / "inventaire-enrichi-palier0-17-09.tsv", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_e_le_tsv_enrichi_suit_les_fichiers(poste, enrichi_brut):
    """(e) Chaque colonne de chaque ligne = ce que les fichiers du dossier
    disent (config.json, gabarit, manifeste, nom) ; N/A quand la source
    manque. La première version (24feac3) disait vision=yes sur 42 modèles
    sans vision_config (Qwen3-Coder, GLM, Llama…) et tools=no sur 31 dont le
    gabarit contient « tools » : aucun test ne lisait la source."""
    _, lignes, _, _, _, _, absents, _ = poste
    par_nom = {r["Modèle"]: r for r in lignes}
    enr = _enrichir()
    fautes = []
    for r in enrichi_brut:
        src = par_nom.get(r["model"])
        if src is None:
            fautes.append(f"{r['model']} : absent de l'inventaire brut")
            continue
        if r["model"] in absents:
            continue
        attendu = enr.enrichir(src["Chemin"], src["Format"], r["model"])
        for col in enr.COLONNES:
            if r.get(col) != attendu[col]:
                fautes.append(f"{r['model']}.{col} : TSV {r.get(col)!r}, fichiers {attendu[col]!r}")
    assert not fautes, f"{len(fautes)} écart(s) TSV/fichiers :\n  " + "\n  ".join(fautes[:40])


def test_g_le_tsv_enrichi_couvre_exactement_l_inventaire(poste, enrichi_brut):
    _, lignes, _, _, _, _, _, _ = poste
    a, b = {r["Modèle"] for r in lignes}, {r["model"] for r in enrichi_brut}
    assert a == b, (sorted(a - b), sorted(b - a))


def test_f_les_regles_d_enrichissement_lisent_les_fichiers(tmp_path):
    """(f) Sur un dossier fabriqué : vision_config → yes, « tools » dans le
    gabarit → yes, manifeste sans nvfp4 → son vrai format ; sans config →
    N/A partout, jamais une valeur devinée."""
    enr = _enrichir()
    d = tmp_path / "Modele-Vision"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_vl", "max_position_embeddings": 4096,
        "rope_scaling": {"type": "linear", "factor": 2.0}, "vision_config": {}, "architectures": ["X"]}))
    (d / "chat_template.jinja").write_text("{% if tools %}{% endif %}")
    (d / "acvram_manifest.json").write_text(json.dumps({"tensors": {"a": {"format": "int8"}, "b": {"format": "bf16"}}}))
    r = enr.enrichir(str(d), "NVFP4_acvram", "Modele-Vision")
    assert r == {"model": "Modele-Vision", "format": "NVFP4_acvram", "model_type": "qwen3_vl",
                 "max_position_embeddings": "4096", "rope_scaling": '{"factor": 2.0, "type": "linear"}',
                 "vision": "yes", "tools": "yes", "thinking": "ND", "bpw": "int8"}, r
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    (d / "chat_template.jinja").write_text("{{ messages }}")
    r = enr.enrichir(str(d), "NVFP4_acvram", "Modele-Vision")
    assert (r["vision"], r["tools"], r["max_position_embeddings"], r["rope_scaling"]) == ("no", "no", "N/A", "None"), r
    v = tmp_path / "Truc-Q5_K_M"
    v.mkdir()
    r = enr.enrichir(str(v), "GGUF", "Truc-Q5_K_M")
    assert (r["model_type"], r["vision"], r["tools"], r["bpw"]) == ("N/A", "N/A", "N/A", "Q5_K_M"), r
    assert enr.enrichir(str(v), "EXL3", "Truc-sans-bpw")["bpw"] == "N/A"


def test_h_une_valeur_fabriquee_dans_le_tsv_casse_e(poste, enrichi_brut):
    """(h) Le contrôle (e) doit dire « faux » : une ligne du TSV modifiée sur
    chaque colonne (vision inversée, tools inversé, bpw inventé, model_type
    inventé) est vue, et seule elle."""
    _, lignes, _, _, _, _, absents, _ = poste
    par_nom = {r["Modèle"]: r for r in lignes}
    enr = _enrichir()
    r = next(x for x in enrichi_brut if x["format"] == "NVFP4_acvram" and x["model"] not in absents)   # un alias présent
    src = par_nom[r["model"]]
    for col, faux in (("vision", "yes" if r["vision"] != "yes" else "no"), ("tools", "maybe"),
                      ("bpw", "W2A2"), ("model_type", "zzz-fiction"), ("thinking", "yes")):
        copie = dict(r); copie[col] = faux
        attendu = enr.enrichir(src["Chemin"], src["Format"], r["model"])
        ecarts = [c for c in enr.COLONNES if copie[c] != attendu[c]]
        assert ecarts == [col], (col, ecarts)


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

    # Regex : chiffre suivi de fichier verdict ou sage
    verdict_pattern = r'([\d,]+)\s+[^—]*—\s*((verdict|sage)-[\w-]+)'

    for menu_name, menu_path in menu_files.items():
        with open(menu_path) as f:
            content = f.read()

        # Chercher tous les appels de verdicts
        for match in re.finditer(verdict_pattern, content):
            chiffre = match.group(1)
            verdict_file = match.group(2)  # ex: "verdict-qwen38-calibA-17-09" ou "sage-calibration-verdict-17-09"

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
