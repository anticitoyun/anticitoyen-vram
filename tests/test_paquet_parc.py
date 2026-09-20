"""acvram-parc à sec (poste7-deb-parc-portable-20-09 § 4) : portabilité, installation à blanc, idempotence, rc ≠ 0.
Aucune carte, aucun serveur : HOME et PATH factices, nvidia-smi absent → classe acvram retirée, vLLM absent."""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
PARC = RACINE / "parc"
sys.path.insert(0, str(PARC / "lib"))
import acvram_parc  # noqa: E402

MOTIFS = (r"/mnt/", r"/opt/ia", r"/home/[a-z]+", r"/media/")  # chemins de machine seulement : le nom d'auteur (ids GTK fr.anticitoyen.*) n'en est pas un (poste7 20/09)


def _gguf(chemin: Path, ctx: int = 4096) -> None:
    """GGUF v3 minimal : 0 tenseur, 2 clés (general.architecture, llama.context_length)."""
    def s(x: str) -> bytes:
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = s("general.architecture") + struct.pack("<I", 8) + s("llama") + s("llama.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx)
    chemin.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 2) + kv + b"\0" * 64)


@pytest.fixture
def poste(tmp_path, monkeypatch):
    """HOME factice, PATH sans nvidia-smi/vllm mais avec un faux llama-server, deux disques et trois faux modèles."""
    home = tmp_path / "home"; home.mkdir()
    binf = tmp_path / "bin"; binf.mkdir()
    (binf / "llama-server").write_text("#!/bin/sh\necho llama-server factice 0\n"); (binf / "llama-server").chmod(0o755)
    (binf / "nvidia-smi").write_text("#!/bin/sh\nexit 1\n"); (binf / "nvidia-smi").chmod(0o755)   # aucun GPU
    for absent in ("vllm", "yals", "jan", "kimi", "claude", "hf"):                                  # moteurs absents, quoi qu'il y ait ailleurs
        (binf / absent).write_text("#!/bin/sh\nexit 127\n"); (binf / absent).chmod(0o000)
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("PATH", f"{binf}:/usr/bin:/bin")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("PARC_PREFIXES_OUTILS", str(tmp_path / "prefixe-vide"))
    d1 = tmp_path / "disque1" / "modeles"; d2 = tmp_path / "disque2" / "modeles"
    (d1 / "Petit-7B-Q4_K_M").mkdir(parents=True); _gguf(d1 / "Petit-7B-Q4_K_M" / "petit-7b-q4_k_m.gguf", 8192)
    (d2 / "Moyen-nvfp4").mkdir(parents=True)
    (d2 / "Moyen-nvfp4" / "config.json").write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 32768}))
    (d2 / "Moyen-nvfp4" / "acvram-00000.safetensors").write_bytes(b"\0" * 16); (d2 / "Moyen-nvfp4" / "acvram_manifest.json").write_text("{}")  # forme réelle d'un dossier converti
    (d2 / "Grand-AWQ").mkdir()
    (d2 / "Grand-AWQ" / "config.json").write_text(json.dumps({"model_type": "qwen2", "max_position_embeddings": 131072, "quantization_config": {"quant_method": "awq"}}))
    (d2 / "Grand-AWQ" / "model.safetensors").write_bytes(b"\0" * 16)
    # un config.toml préexistant avec un bloc de l'utilisateur, qui doit rester intact
    (home / ".kimi-code").mkdir(); (home / ".kimi-code" / "config.toml").write_text('[providers.perso]\ntype = "openai"\nbase_url = "http://exemple/v1"\napi_key = "secret-perso"\n\n[thinking]\nmode = "off"\n')
    cfg = home / ".config" / "acvram-parc" / "parc.toml"
    return {"home": home, "cfg": cfg, "d1": d1, "d2": d2, "env": dict(os.environ)}


def _installer(p, *args):
    return subprocess.run([sys.executable, str(PARC / "bin" / "parc-installer"), "--auto", "--sans-balayage", "--sans-minuteur",
                           "--config", str(p["cfg"]), "--racine", str(p["d1"]), "--racine", str(p["d2"]), *args],
                          capture_output=True, text=True, env=p["env"], timeout=120)


def _empreinte(home: Path) -> dict[str, bytes]:
    return {str(f.relative_to(home)): f.read_bytes() for f in sorted(home.rglob("*")) if f.is_file() and ".avant-" not in f.name}


def test_lecteur_defauts_sans_chemin_machine(tmp_path):
    P = acvram_parc.charger(tmp_path / "absent.toml")
    assert P.moteurs == {} and P.racines == [] and P.tsv("gguf").name == "gguf-chemins.tsv"
    assert all(not re.search(m, (PARC / "lib" / "acvram_parc.py").read_text()) for m in MOTIFS)


def test_gguf_contexte(tmp_path):
    sys.path.insert(0, str(PARC / "bin"))
    import importlib.util
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("parc_installer", str(PARC / "bin" / "parc-installer"))
    spec = importlib.util.spec_from_loader("parc_installer", loader)
    mod = importlib.util.module_from_spec(spec); loader.exec_module(mod)
    _gguf(tmp_path / "x.gguf", 16384)
    assert mod.lire_ctx_gguf(tmp_path / "x.gguf") == 16384


def test_paquet_sans_chemin_de_machine(tmp_path):
    r = subprocess.run(["bash", str(RACINE / "tools" / "construire-deb-parc.sh"), "--stage-only", str(tmp_path)], capture_output=True, text=True, cwd=RACINE)
    assert r.returncode == 0, r.stderr[-800:]
    pkg = Path(r.stdout.strip().splitlines()[-1])
    trouves = []
    for f in (pkg / "usr").rglob("*"):
        if f.is_file():
            t = f.read_text(errors="replace")
            trouves += [f"{f.relative_to(pkg)}: {m}" for m in MOTIFS if re.search(m, t)]
    assert trouves == [], trouves
    assert (pkg / "usr/bin/parc-installer").exists() and (pkg / "usr/share/acvram-parc/lib/acvram_parc.py").exists()
    assert "Version: " in (pkg / "DEBIAN/control").read_text()


def test_installation_a_blanc_un_seul_alias(poste):
    r = _installer(poste)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]
    tsv = poste["home"] / "TSV"
    gguf = [l for l in (tsv / "gguf-chemins.tsv").read_text().splitlines() if l]
    assert len(gguf) == 1 and gguf[0].startswith("llamacpp-petit-7b-q4-k-m-gguf\t") and gguf[0].endswith("\t8192"), gguf
    assert not [l for l in (tsv / "vllm-chemins.tsv").read_text().splitlines() if l], "vLLM absent : aucun alias vllm"
    assert not [l for l in (tsv / "acvram-chemins.tsv").read_text().splitlines() if l], "sans sm_120 : aucun alias acvram"
    assert "acvram RETIRÉE" in r.stdout or "acvram" not in r.stdout.split("moteur")[0] or "absent" in r.stdout
    cfg = (poste["home"] / ".kimi-code" / "config.toml").read_text()
    assert '[providers.perso]\ntype = "openai"\nbase_url = "http://exemple/v1"\napi_key = "secret-perso"' in cfg, "bloc utilisateur modifié"
    assert "[thinking]" in cfg and "[models.llamacpp-petit-7b-q4-k-m-gguf]" in cfg and "[providers.llamacpp]" in cfg
    assert "[models.acvram" not in cfg and "[providers.vllm]" not in cfg
    notes = (tsv / "notes-modeles.tsv").read_text()
    assert "llamacpp-petit-7b-q4-k-m-gguf\tinconnu\tnon mesuré\tnon mesuré\tinconnu" in notes and "?" not in notes
    p = acvram_parc.charger(poste["cfg"])
    assert set(p.moteurs) == {"llamacpp"} and p.port("llamacpp") in range(8080, 8130) and len(p.racines) == 2
    assert not any(re.search(m, poste["cfg"].read_text()) for m in MOTIFS)


def test_idempotence_deux_passages_identiques(poste):
    assert _installer(poste).returncode == 0
    avant = _empreinte(poste["home"])
    r2 = _installer(poste)
    assert r2.returncode == 0 and "rien (déjà à jour)" in r2.stdout, r2.stdout[-600:]
    assert _empreinte(poste["home"]) == avant


def test_rc_non_nul_si_un_alias_ne_resout_pas(poste):
    assert _installer(poste).returncode == 0
    import shutil
    shutil.rmtree(poste["d1"] / "Petit-7B-Q4_K_M")
    r = _installer(poste, "--verifier")
    assert r.returncode == 2 and "FAUX" in r.stdout, r.stdout[-600:]


def test_alias_disparu_marque_absent_et_conserve(poste):
    assert _installer(poste).returncode == 0
    import shutil
    shutil.rmtree(poste["d1"] / "Petit-7B-Q4_K_M")
    r = _installer(poste)
    assert r.returncode == 2
    l = (poste["home"] / "TSV" / "gguf-chemins.tsv").read_text()
    assert "llamacpp-petit-7b-q4-k-m-gguf\t" in l and "\tABSENT" in l
    assert "llamacpp-petit-7b-q4-k-m-gguf\tinconnu" in (poste["home"] / "TSV" / "notes-modeles.tsv").read_text(), "note perdue"


def _etat(racine: Path) -> dict[str, tuple[int, int]]:
    return {str(f): (f.stat().st_mtime_ns, f.stat().st_size) for f in racine.rglob("*") if f.is_file()}


def test_a_sec_zero_ecriture_hors_tmp(poste, tmp_path):
    """Instrument : un mode « essai » (parc-installer --verifier, telecharger-modele --a-sec) n'écrit nulle part.
    Faute construite : le même contrôle sur un passage qui écrit (parc-installer sans --verifier) rend rouge."""
    assert _installer(poste).returncode == 0                      # parc en place
    avant = _etat(tmp_path)
    r = _installer(poste, "--verifier")
    assert r.returncode == 0 and "écrit :" not in r.stdout, r.stdout[-400:]
    env = dict(poste["env"], ACVRAM_PARC_CONFIG=str(poste["cfg"]))
    r2 = subprocess.run(["bash", str(PARC / "bin" / "telecharger-modele"), "--a-sec", "org/depot-factice", str(tmp_path / "dest")],
                        capture_output=True, text=True, env=env, timeout=60)
    assert r2.returncode == 0 and "à sec" in r2.stdout, r2.stdout + r2.stderr[-300:]
    r3 = subprocess.run(["bash", str(PARC / "bin" / "telecharger-modele")], capture_output=True, text=True, env=env, timeout=60)
    assert r3.returncode == 2, "sans argument : usage, rc 2, aucune écriture"
    assert _etat(tmp_path) == avant, "un mode à sec a écrit : " + str(set(_etat(tmp_path).items()) ^ set(avant.items()))
    # faute construite : un passage qui écrit doit être vu par le même instrument
    (poste["d2"] / "Nouveau-AWQ").mkdir()
    (poste["d2"] / "Nouveau-AWQ" / "config.json").write_text(json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "awq"}}))
    (poste["d2"] / "Nouveau-AWQ" / "model.safetensors").write_bytes(b"\0" * 8)
    (poste["d2"] / "Nouveau-Q4.gguf").write_bytes(b"GGUF")     # nouveau gguf invalide : ignoré ; le dossier AWQ sans vLLM aussi
    avant2 = _etat(tmp_path)
    (poste["home"] / "TSV" / "gguf-chemins.tsv").write_text("")  # on force une réécriture
    assert _installer(poste).returncode == 0
    assert _etat(tmp_path) != avant2, "l'instrument ne voit pas une écriture"
