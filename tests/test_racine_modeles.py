"""outils/racine_modeles.py : ACVRAM_MODELES → ~/.config/acvram/modeles (même lecture que cli.py) → littéral
(20/09 : GNOME n'hérite pas de environment.d, les chaînes doivent voir le parc sans variable)."""
from __future__ import annotations

import importlib.util
import os
import pathlib

RACINE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "racine_modeles.py"


def _charge(monkeypatch, env, xdg):
    for k in ("ACVRAM_MODELES", "ACVRAM_MODELS_DIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    spec = importlib.util.spec_from_file_location("racine_modeles_test", RACINE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_la_variable_gagne(tmp_path, monkeypatch):
    m = _charge(monkeypatch, {"ACVRAM_MODELES": str(tmp_path)}, tmp_path / "xdg")
    assert m.MODELES == str(tmp_path)


def test_le_fichier_de_configuration_est_lu_comme_cli(tmp_path, monkeypatch):
    parc = tmp_path / "parc"; parc.mkdir()
    conf = tmp_path / "xdg" / "acvram"; conf.mkdir(parents=True)
    (conf / "modeles").write_text(f"# commentaire\n\n{tmp_path / 'absent'}\n{parc}\n")   # une ligne absente est sautée
    m = _charge(monkeypatch, {}, tmp_path / "xdg")
    assert m.MODELES == str(parc)


def test_sans_rien_le_litteral(tmp_path, monkeypatch):
    m = _charge(monkeypatch, {}, tmp_path / "xdg-vide")
    assert m.MODELES == m._DEFAUT and m._DEFAUT.startswith("/mnt/")


# ---- test cassant (Sage, 20/09) : aucun chemin de parc en dur hors racine_modeles.py ----

DEPOT = RACINE.parent.parent
LITTERAUX = ("/mnt/2TO_2023_980PRO/Modeles/models_acvram",
             "/mnt/4TO_SATACMR_2022/Modeles/models_acvram")
DOSSIERS_CODE = ("acvram", "outils", "tests", "tools")
EXCLUS = {RACINE, pathlib.Path(__file__).resolve()}


def _fichiers_de_code():
    for d in DOSSIERS_CODE:
        base = DEPOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.suffix in (".py", ".sh") and p not in EXCLUS and "scratchpad" not in p.parts:
                yield p


def test_aucun_chemin_de_parc_en_dur_hors_racine_modeles():
    """Le parc a déjà changé de disque deux fois (980 PRO USB, SATA CMR, AI_GENERATOR) ; un
    littéral dans un outil ou un test ne suit pas. Les lignes de commentaire (#) sont tolérées :
    elles disent l'histoire, pas le chemin lu."""
    fautes = []
    for p in _fichiers_de_code():
        for i, ligne in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if ligne.lstrip().startswith("#"):
                continue
            if any(l in ligne for l in LITTERAUX):
                fautes.append(f"{p.relative_to(DEPOT)}:{i}")
    assert not fautes, "chemin de parc en dur (utiliser outils/racine_modeles.py) : " + ", ".join(fautes)


def test_alias_absent_nomme_la_racine(tmp_path, monkeypatch):
    m = _charge(monkeypatch, {"ACVRAM_MODELES": str(tmp_path)}, tmp_path / "xdg")
    assert m.alias("x") == str(tmp_path / "x")
    assert m.alias_absent("x").startswith("alias absent sous racine_modeles() : ")
    (tmp_path / "x").mkdir()
    assert m.alias_absent("x") == ""


def test_le_script_imprime_la_racine(tmp_path, monkeypatch):
    import subprocess, sys
    env = dict(os.environ, ACVRAM_MODELES=str(tmp_path))
    out = subprocess.run([sys.executable, str(RACINE)], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == str(tmp_path)
