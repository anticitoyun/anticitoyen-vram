"""outils/verifier-contexte.py à sec : trois verdicts sur des alias factices (OK, RÉDUIT, FAUX) + REFUS, --appliquer, --sans-plan, lots.
Aucune carte : `acvram` est un faux binaire dans le PATH qui rend un plan JSON selon le nom du dossier."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "outils" / "verifier-contexte.py"

FAUX_ACVRAM = r'''#!/usr/bin/env python3
import json, sys, os
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["MARQUE_APPELS"]).open("a").write(" ".join(args) + "\n")
dossier = Path(args[1]).name; mml = int(args[args.index("--max-model-len") + 1])
if "refus" in dossier:
    print(json.dumps({"plan": {"kv_max_tokens": 0, "warnings": [dossier + " ne tient pas sur cette machine : il manque 9 Gio"]}})); sys.exit(0)
kv = 16384 if "reduit" in dossier else 65536
print(json.dumps({"plan": {"kv_max_tokens": kv, "warnings": []}}))
'''


def _gguf(chemin: Path, ctx: int) -> None:
    def s(x: str) -> bytes:
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    kv = s("general.architecture") + struct.pack("<I", 8) + s("llama") + s("llama.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx)
    chemin.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 2) + kv + b"\0" * 32)


@pytest.fixture
def parc(tmp_path):
    binf = tmp_path / "bin"; binf.mkdir(); (binf / "acvram").write_text(FAUX_ACVRAM); (binf / "acvram").chmod(0o755)
    m = tmp_path / "modeles"; tsv = tmp_path / "TSV"; tsv.mkdir()
    for nom, ctx in (("ok-nvfp4", 131072), ("reduit-nvfp4", 131072), ("faux-nvfp4", 8192), ("refus-nvfp4", 131072)):
        (m / nom).mkdir(parents=True); (m / nom / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": ctx}}))
    (m / "petit-gguf").mkdir(); _gguf(m / "petit-gguf" / "petit.gguf", 4096)
    (tsv / "acvram-chemins.tsv").write_text("".join(f"acvram-{n}\t{m / n}\t32768\n" for n in ("ok-nvfp4", "reduit-nvfp4", "faux-nvfp4", "refus-nvfp4")))
    (tsv / "gguf-chemins.tsv").write_text(f"llamacpp-petit\t{m / 'petit-gguf'}\t8192\n")
    env = {**os.environ, "PATH": f"{binf}:{os.environ['PATH']}", "MARQUE_APPELS": str(tmp_path / "appels.txt")}
    return {"tsv": tsv, "env": env, "appels": tmp_path / "appels.txt", "m": m}


def _run(p, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--tsv-dir", str(p["tsv"]), "--acvram", "acvram", *args],
                          capture_output=True, text=True, env=p["env"], timeout=120)


def _verdicts(p) -> dict[str, list[str]]:
    return {l.split("\t")[0]: l.split("\t") for l in (p["tsv"] / "contexte-verifie.tsv").read_text().splitlines() if l and not l.startswith("#")}


def test_trois_verdicts_et_refus(parc):
    r = _run(parc)
    v = _verdicts(parc)
    assert v["acvram-ok-nvfp4"][1:5] == ["32768", "131072", "65536", "OK"]
    assert v["acvram-reduit-nvfp4"][1:5] == ["32768", "131072", "16384", "RÉDUIT"] and "proposer 16384" in v["acvram-reduit-nvfp4"][5]
    assert v["acvram-faux-nvfp4"][1:5] == ["32768", "8192", "65536", "FAUX"]
    assert v["acvram-refus-nvfp4"][4] == "REFUS" and "ne tient pas" in v["acvram-refus-nvfp4"][5]
    assert v["llamacpp-petit"][1:5] == ["8192", "4096", "-", "FAUX"], v["llamacpp-petit"]
    assert r.returncode == 1, "FAUX/REFUS présents → rc 1"
    assert len(parc["appels"].read_text().splitlines()) == 4, "un acvram plan par alias acvram, aucun pour gguf"
    assert all("CUDA_VISIBLE_DEVICES" not in l for l in parc["appels"].read_text().splitlines())


def test_plan_a_sec_sans_carte(parc):
    """Le faux acvram voit CUDA_VISIBLE_DEVICES vide : le script le pose lui-même."""
    (parc["env"]["PATH"].split(":")[0] and None)
    faux = Path(parc["env"]["PATH"].split(":")[0]) / "acvram"
    faux.write_text(FAUX_ACVRAM.replace('Path(os.environ["MARQUE_APPELS"]).open("a").write(" ".join(args) + "\\n")',
                                        'Path(os.environ["MARQUE_APPELS"]).open("a").write(repr(os.environ.get("CUDA_VISIBLE_DEVICES")) + "\\n")'))
    _run(parc)
    assert set(parc["appels"].read_text().split()) == {"''"}, parc["appels"].read_text()


def test_appliquer_reecrit_la_colonne_avec_sauvegarde(parc):
    _run(parc, "--appliquer")
    lignes = {l.split("\t")[0]: l.split("\t") for l in (parc["tsv"] / "acvram-chemins.tsv").read_text().splitlines()}
    assert lignes["acvram-reduit-nvfp4"][2] == "16384" and lignes["acvram-faux-nvfp4"][2] == "8192" and lignes["acvram-ok-nvfp4"][2] == "32768"
    assert lignes["acvram-refus-nvfp4"][2] == "32768", "REFUS : colonne intacte, c'est à l'humain"
    assert {l.split("\t")[0]: l.split("\t")[2] for l in (parc["tsv"] / "gguf-chemins.tsv").read_text().splitlines()}["llamacpp-petit"] == "4096"
    assert list(parc["tsv"].glob("acvram-chemins.tsv.avant-*")) and list(parc["tsv"].glob("gguf-chemins.tsv.avant-*"))
    # second passage : plus rien à corriger, verdicts OK/REFUS seulement
    r2 = _run(parc)
    assert {v[4] for v in _verdicts(parc).values()} == {"OK", "REFUS"}, _verdicts(parc)


def test_sans_plan_aucune_charge(parc):
    r = _run(parc, "--sans-plan")
    assert not parc["appels"].exists(), "--sans-plan a appelé acvram"
    v = _verdicts(parc)
    assert v["acvram-faux-nvfp4"][4] == "FAUX" and v["acvram-ok-nvfp4"][4] == "OK" and v["acvram-ok-nvfp4"][3] == "?"


def test_lots_et_reprise(parc):
    _run(parc, "--lot", "2")
    assert len(parc["appels"].read_text().splitlines()) == 2
    v = _verdicts(parc)
    assert sum(1 for x in v.values() if x[4] == "?" and "lot" in x[5]) == 2, "les alias hors lot sont notés « à planifier »"
    _run(parc, "--lot", "2", "--depuis", "2")
    assert len(parc["appels"].read_text().splitlines()) == 4
    assert "?" not in {x[4] for x in _verdicts(parc).values()}


def test_faute_construite_colonne_non_numerique(parc):
    (parc["tsv"] / "acvram-chemins.tsv").write_text(f"acvram-bizarre\t{parc['m'] / 'ok-nvfp4'}\tbeaucoup\n")
    _run(parc, "--sans-plan")
    assert _verdicts(parc)["acvram-bizarre"][4] == "?" and "non numérique" in _verdicts(parc)["acvram-bizarre"][5]
