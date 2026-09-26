"""outils/usages-modeles.py dérive l'usage (tags fermés) de chaque modèle servi
depuis des signaux LOCAUX (nom d'alias, nom de modèle, VISION_TSV, ctx servi,
README local), jamais une carte HF en ligne ni une sortie de modèle. Il régénère
la colonne « usage » de notes-modeles.tsv et le journal de preuves usage-sources.tsv,
appelé par modeles-a-jour — la table n'est jamais figée à la main.

Ce test joue le dériveur RÉEL sur une fixture suivie (tests/fixtures/usage/,
10 alias couvrant les 15 tags dont un heretic, un rp, un vision, un ctx 131072 et
un « safe ») et vérifie : tags ∈ vocabulaire ; sans-censure/nsfw/explicite/porno
prouvés ; ≥ 2 tags par alias ; précédence porno ⊃ explicite ⊃ nsfw ⊃ sans-censure ;
vision ⇔ VISION_TSV. Cassant : toute règle qui régresse fait rougir un cas."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import pathlib

RACINE = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = RACINE / "outils" / "usages-modeles.py"
FIXTURE = RACINE / "tests" / "fixtures" / "usage"

_spec = importlib.util.spec_from_file_location("usages_modeles", SCRIPT)
_um = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_um)
VOCAB = set(_um.VOCAB)
CENSURE = ["sans-censure", "nsfw", "explicite", "porno"]


def _jouer(tmp_path):
    """Copie la fixture, écrit un parc.toml (chemins en tmp, jamais /home suivi),
    lance le dériveur, et rend (usage par alias, preuves par alias)."""
    dst = tmp_path / "usage"
    shutil.copytree(FIXTURE, dst)
    parc = dst / "parc.toml"
    parc.write_text(
        f'[chemins]\nkimi_dir = "{dst/"kimi"}"\ntsv_dir = "{dst/"tsv"}"\n'
        f'[moteurs.acvram]\npresent = true\n[moteurs.llamacpp]\npresent = true\n')
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(parc), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(["/usr/bin/python3", str(SCRIPT)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]

    usage = {}
    for l in (dst / "tsv" / "notes-modeles.tsv").read_text().splitlines():
        if l and not l.startswith("#"):
            p = l.split("\t")
            if len(p) >= 5 and p[4].strip():
                usage[p[0]] = [t.strip() for t in p[4].split("·") if t.strip()]
    preuves = {}
    for l in (dst / "tsv" / "usage-sources.tsv").read_text().splitlines():
        if l and not l.startswith("#"):
            p = l.split("\t")
            if len(p) >= 3:
                preuves.setdefault(p[0], []).append((p[1], p[2], p[3] if len(p) > 3 else ""))
    return usage, preuves


def test_tags_dans_le_vocabulaire_et_au_moins_deux(tmp_path):
    usage, _ = _jouer(tmp_path)
    assert usage, "aucun usage dérivé"
    for alias, tags in usage.items():
        assert set(tags) <= VOCAB, f"{alias} : tags hors vocab {set(tags) - VOCAB}"
        assert len(tags) >= 2, f"{alias} : moins de 2 tags ({tags})"


def test_les_quinze_tags_sont_couverts(tmp_path):
    usage, _ = _jouer(tmp_path)
    vus = set().union(*usage.values())
    assert vus == VOCAB, f"tags non couverts par la fixture : {VOCAB - vus}"


def test_precedence_censure(tmp_path):
    usage, _ = _jouer(tmp_path)
    for alias, tags in usage.items():
        t = set(tags)
        if "porno" in t:
            assert {"explicite", "nsfw", "sans-censure"} <= t, alias
        if "explicite" in t:
            assert {"nsfw", "sans-censure"} <= t, alias
        if "nsfw" in t:
            assert "sans-censure" in t, alias


def test_tags_de_censure_sont_prouves(tmp_path):
    usage, preuves = _jouer(tmp_path)
    for alias, tags in usage.items():
        for tag in tags:
            if tag in CENSURE:
                preuves_tag = [pr for pr in preuves.get(alias, []) if pr[0] == tag]
                assert preuves_tag, f"{alias} : tag « {tag} » sans preuve dans usage-sources.tsv"


def test_vision_equivaut_au_tsv(tmp_path):
    usage, _ = _jouer(tmp_path)
    vision_tsv = set()
    for l in (FIXTURE / "tsv" / "vision-modeles.tsv").read_text().splitlines():
        p = l.split("\t")
        if len(p) >= 2 and p[1] == "vision":
            vision_tsv.add(p[0])
    avec_vision = {a for a, tags in usage.items() if "vision" in tags}
    assert avec_vision == vision_tsv, (avec_vision, vision_tsv)


def test_cas_repere(tmp_path):
    """Points fixes de la fixture : le porno tire les trois inférieurs ; le heretic
    est sans-censure seul ; le « safe » ctx 131072 est long-contexte + vedette sans
    aucun tag de censure ; le converti n'existe pas ici mais gemma est vision."""
    usage, _ = _jouer(tmp_path)
    assert set(usage["acvram-hardcore-porn-7b"]) >= {"porno", "explicite", "nsfw", "sans-censure"}
    assert "sans-censure" in usage["acvram-heretic-uncensored-7b"]
    assert "nsfw" not in usage["acvram-heretic-uncensored-7b"]
    safe = set(usage["acvram-mistral-safe-long"])
    assert "long-contexte" in safe and "vedette" in safe
    assert not (safe & set(CENSURE)), f"le « safe » porte un tag de censure : {safe}"
    assert "vision" in usage["llamacpp-gemma-vision"]
