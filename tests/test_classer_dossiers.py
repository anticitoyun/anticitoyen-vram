"""Pièce 84 : `classer()` de parc-installer (le SEUL classeur, réutilisé par le balayage GUI via
`parc-installer --sans-balayage --racine …` — voir test_gui_dossiers_modeles.py) sur les trois sortes
de dossiers de modèles et le cas vide. Chaque cas positif est cassé une fois : le test tombe si la
classification devient None sans que le dossier ait vraiment changé de nature."""
from __future__ import annotations

import json
import struct
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import importlib.util
import pytest

RACINE = Path(__file__).resolve().parent.parent
PARC = RACINE / "parc"


def _charger_parc_installer():
    loader = SourceFileLoader("parc_installer_p84", str(PARC / "bin" / "parc-installer"))
    spec = importlib.util.spec_from_loader("parc_installer_p84", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pi():
    return _charger_parc_installer()


def _gguf(chemin: Path, ctx: int = 4096) -> None:
    def s(x: str) -> bytes:
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = s("general.architecture") + struct.pack("<I", 8) + s("llama") + s("llama.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx)
    chemin.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 2) + kv + b"\0" * 64)


def test_dossier_vide_aucune_classe(pi, tmp_path):
    d = tmp_path / "vide"; d.mkdir()
    assert pi.classer(d) is None


def test_gguf_classe_llamacpp(pi, tmp_path):
    d = tmp_path / "Petit-7B"; d.mkdir()
    f = d / "petit-7b-q4_k_m.gguf"
    _gguf(f, 8192)
    assert pi.classer(d) == ("llamacpp", "gguf", 8192)
    # cassé : renommer hors .gguf → plus de fichier modèle reconnu dans le dossier
    f.rename(d / "petit-7b-q4_k_m.bin")
    assert pi.classer(d) is None


def test_gguf_fichier_seul_classe_llamacpp(pi, tmp_path):
    f = tmp_path / "solo.gguf"
    _gguf(f, 16384)
    assert pi.classer(f) == ("llamacpp", "gguf", 16384)
    # cassé : magique GGUF corrompu → ctx illisible mais fichier toujours classé llamacpp par
    # l'extension seule (lire_ctx_gguf rend None, pas une exception) : le VRAI cas cassant est
    # le suffixe, testé ci-dessus pour le dossier ; ici on vérifie la robustesse du contexte.
    f.write_bytes(b"XXXX" + f.read_bytes()[4:])
    assert pi.classer(f) == ("llamacpp", "gguf", None)


def test_acvram_manifest_classe_acvram(pi, tmp_path):
    d = tmp_path / "Moyen-nvfp4"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 32768}))
    (d / "acvram-00000.safetensors").write_bytes(b"\0" * 16)
    (d / "acvram_manifest.json").write_text("{}")
    assert pi.classer(d) == ("acvram", "nvfp4", 32768)
    # cassé : retirer le manifeste ET renommer le safetensors hors du motif acvram-* → plus
    # aucun signe d'un converti acvram, et sans quantization_config ce n'est pas non plus un
    # dossier vLLM reconnu : None.
    (d / "acvram_manifest.json").unlink()
    (d / "acvram-00000.safetensors").rename(d / "model.safetensors")
    assert pi.classer(d) is None


def test_hf_awq_classe_vllm(pi, tmp_path):
    d = tmp_path / "Grand-AWQ"; d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "max_position_embeddings": 131072,
        "quantization_config": {"quant_method": "awq"}}))
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    assert pi.classer(d) == ("vllm", "awq", 131072)
    # cassé : retirer quantization_config → un HF safetensors sans quantification reconnue
    # n'est pas classé (parc-installer ne sait pas quel moteur le sert).
    (d / "config.json").write_text(json.dumps({"model_type": "qwen2", "max_position_embeddings": 131072}))
    assert pi.classer(d) is None
