"""CTX_CLIENT_MIN exporté par les menus — sans col4 TSV, le serveur ajuste CTX.

Cassure : supprimer la ligne `ctx_client_min="${ctx_client_min_col4:-${CTX_CLIENT_MIN:-}}"` dans
acvram-serveur → le test échoue parce que CTX reste 32768 et kimi (65536) n'aurait pas de
contexte suffisant.
"""
import json
import os
import pathlib
import subprocess


def _serveur_sh() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "parc" / "bin" / "acvram-serveur"


def _run_a_sec(alias: str, tsv_dir: pathlib.Path, model_dir: pathlib.Path,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Lance acvram-serveur en mode à sec (ACVRAM_SERVEUR_A_SEC=1)."""
    env = dict(os.environ)
    env["HOME"] = str(tsv_dir.parent)   # TSV cherché dans $HOME/TSV/
    env["ACVRAM_SERVEUR_A_SEC"] = "1"
    # binaire factice : le vrai /usr/bin/acvram, sous un HOME neuf, crée son venv et lance pip (réseau,
    # plusieurs minutes) juste pour « --version » ; et il est absent d'une CI sans le paquet
    faux = tsv_dir.parent / "faux-acvram"
    faux.write_text("#!/bin/sh\necho 'acvram 0.0.0-essai'\n")
    faux.chmod(0o755)
    env["ACVRAM_PAQUET_BIN"] = str(faux)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(_serveur_sh()), alias],
        capture_output=True, text=True, env=env,
    )


def test_ctx_client_min_env_sans_col4(tmp_path):
    """Alias sans col4 + CTX_CLIENT_MIN=65536 → ctx ajusté à 65536."""
    model_dir = tmp_path / "mon-modele"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 131072})
    )
    tsv_dir = tmp_path / "TSV"
    tsv_dir.mkdir()
    (tsv_dir / "acvram-chemins.tsv").write_text(
        f"mon-alias\t{model_dir}\t32768\n"   # ← pas de col4
    )
    res = _run_a_sec("mon-alias", tsv_dir, model_dir,
                     env_extra={"CTX_CLIENT_MIN": "65536"})
    combined = res.stdout + res.stderr
    assert "65536" in combined and ("ajust" in combined or "65536" in res.stdout), (
        f"ctx_client_min depuis env non appliqué (code={res.returncode}) :\n{combined[:400]}"
    )


def test_col4_prime_sur_env(tmp_path):
    """Col4 TSV (surcharge admin) prime sur CTX_CLIENT_MIN de l'env."""
    model_dir = tmp_path / "mon-modele"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 131072})
    )
    tsv_dir = tmp_path / "TSV"
    tsv_dir.mkdir()
    # col4 = 98304, env = 65536 ; le serveur doit choisir 98304
    (tsv_dir / "acvram-chemins.tsv").write_text(
        f"mon-alias\t{model_dir}\t32768\t98304\n"
    )
    res = _run_a_sec("mon-alias", tsv_dir, model_dir,
                     env_extra={"CTX_CLIENT_MIN": "65536"})
    combined = res.stdout + res.stderr
    assert "98304" in combined, (
        f"col4 TSV non prioritaire sur CTX_CLIENT_MIN env :\n{combined[:400]}"
    )


def test_ctx_client_min_env_modele_trop_petit(tmp_path):
    """CTX_CLIENT_MIN > max_position_embeddings → refus avec les deux chiffres."""
    model_dir = tmp_path / "petit-modele"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 32768})
    )
    tsv_dir = tmp_path / "TSV"
    tsv_dir.mkdir()
    (tsv_dir / "acvram-chemins.tsv").write_text(
        f"petit-alias\t{model_dir}\t32768\n"
    )
    res = _run_a_sec("petit-alias", tsv_dir, model_dir,
                     env_extra={"CTX_CLIENT_MIN": "65536"})
    assert res.returncode != 0, "refus attendu mais le serveur a accepté"
    err_txt = res.stderr
    assert "65536" in err_txt and "32768" in err_txt, (
        f"chiffres absents du message de refus : {err_txt[:300]}"
    )
