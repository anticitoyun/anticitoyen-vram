"""modeles-a-jour § 2b (poste7-s2-k48-feu-vert-21-09) : la colonne ctx recopie ctx_tenu= du journal
de service, toujours (pas seulement sous --appliquer) ; le tout SAUTE si .qui n'est pas vide."""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent


def _charger_module(nom="modeles_a_jour_test"):
    loader = importlib.machinery.SourceFileLoader(nom, str(RACINE / "parc" / "bin" / "modeles-a-jour"))
    spec = importlib.util.spec_from_loader(nom, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


import importlib.machinery  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    qui = tmp_path / "carte.lock.qui"
    journal = tmp_path / "acvram-serveur.log"
    tsv = tmp_path / "acvram-chemins.tsv"
    tsv.write_text(
        "acvram-essai\t/mnt/x/Essai\t32768\n"
        "acvram-vision\t/mnt/x/Vision\t262144\t\t\tduel\n"
        "acvram-autre\t/mnt/x/Autre\t8192\n"
    )
    mod = _charger_module()
    mod.QUI = str(qui)
    mod.JOURNAL_SERVICE = str(journal)
    mod.CTX_TSV = str(tsv)
    mod.TSVS = []  # jamais toucher les vrais TSV de production pendant les tests de main()
    mod.RACINES = []
    mod.VISION_TSV = str(tmp_path / "vision-modeles.tsv")  # idem : ne jamais écraser le vrai fichier
    return {"qui": qui, "journal": journal, "tsv": tsv, "mod": mod}


def test_recopie_ctx_tenu_sans_appliquer(env, monkeypatch):
    """Le flag --appliquer n'existe pas pour cette étape : elle écrit quand même."""
    env["journal"].write_text(
        "acvram : service de acvram-essai (contexte 32768)…\n"
        "kv=int8 graphes=on ctx_tenu=15360 spec=ngram\n"
        "acvram : service de acvram-vision (contexte 262144)…\n"
        "kv=int8 ctx_tenu=non-chauffe\n"
        "kv=int8 ctx_tenu=4096\n"
    )
    monkeypatch.setattr(sys, "argv", ["modeles-a-jour"])  # pas d'--appliquer
    env["mod"]._recopier_ctx_tenu()
    lignes = {l.split("\t")[0]: l.split("\t") for l in env["tsv"].read_text().splitlines()}
    assert lignes["acvram-essai"][2] == "15360"
    assert lignes["acvram-vision"][2] == "4096" and lignes["acvram-vision"][5] == "duel", "colonne duel préservée"
    assert lignes["acvram-autre"][2] == "8192", "alias non servi : colonne intacte"
    assert list(env["tsv"].parent.glob("acvram-chemins.tsv.avant-*")), "sauvegarde datée absente"


def test_ctx_tenu_non_numerique_ignore(env):
    """non-verifie / non-chauffe ne sont pas des valeurs servies : aucune écriture."""
    env["journal"].write_text(
        "acvram : service de acvram-essai (contexte 32768)…\n"
        "kv=int8 ctx_tenu=non-verifie\n"
    )
    avant = env["tsv"].read_text()
    env["mod"]._recopier_ctx_tenu()
    assert env["tsv"].read_text() == avant, "aucune valeur servie : rien à recopier"
    assert not list(env["tsv"].parent.glob("acvram-chemins.tsv.avant-*"))


def test_seul_le_dernier_bloc_de_lalias_compte(env):
    """Deux démarrages du même alias dans le journal : seul le dernier bloc fait foi."""
    env["journal"].write_text(
        "acvram : service de acvram-essai (contexte 32768)…\n"
        "kv=int8 ctx_tenu=8192\n"
        "acvram : service de acvram-essai (contexte 32768)…\n"
        "kv=int8 ctx_tenu=15360\n"
    )
    env["mod"]._recopier_ctx_tenu()
    lignes = {l.split("\t")[0]: l.split("\t") for l in env["tsv"].read_text().splitlines()}
    assert lignes["acvram-essai"][2] == "15360"


def test_qui_non_vide_saute_tout(env, capsys):
    """.qui non vide : SAUTÉ, aucune lecture de journal ni écriture, main() rend 0."""
    env["qui"].write_text("123456 1789999999 python mesure\n")
    env["journal"].write_text("acvram : service de acvram-essai (contexte 32768)…\nctx_tenu=1\n")
    avant = env["tsv"].read_text()
    rc = env["mod"].main()
    out = capsys.readouterr().out
    assert rc == 0 and "SAUTÉ" in out and "mesure" not in out.split("SAUTÉ")[0]
    assert env["tsv"].read_text() == avant, "SAUTÉ : la colonne ne doit pas bouger"
    assert not list(env["tsv"].parent.glob("acvram-chemins.tsv.avant-*"))


def test_faute_construite_qui_vide_ne_saute_pas(env, capsys):
    """Faute construite : si .qui est vide (ou absent), le passage doit avoir lieu, pas être sauté."""
    env["journal"].write_text("acvram : service de acvram-essai (contexte 32768)…\nctx_tenu=15360\n")
    rc = env["mod"].main()
    out = capsys.readouterr().out
    assert "SAUTÉ" not in out
    lignes = {l.split("\t")[0]: l.split("\t") for l in env["tsv"].read_text().splitlines()}
    assert lignes["acvram-essai"][2] == "15360"
