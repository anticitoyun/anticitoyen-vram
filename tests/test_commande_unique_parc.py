"""Sage 21/09 (§ 1b du feu vert 0.6.34) : la commande que la GUI IMPRIME est, octet pour octet, celle que
l'exec REÇOIT — un seul constructeur par lanceur. Cassant : l'ancien « à sec » d'acvram-serveur omettait
--speculative et --no-cuda-graphs ; ici GRAPHES et SPECULATIF sont posés pour que l'ancien code diffère."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
PARC = RACINE / "parc"
FAUX_EXEC = "#!/bin/bash\nprintf 'recu :'; printf ' %q' \"$@\"; printf '\\n'\n"   # ce que l'exec reçoit, même format %q


def _exe(chemin: Path, corps: str) -> Path:
    chemin.write_text(corps); chemin.chmod(chemin.stat().st_mode | stat.S_IEXEC); return chemin


def _ligne(sortie: str, prefixe: str = "commande :") -> str:
    lignes = [l[len(prefixe):] for l in sortie.splitlines() if l.startswith(prefixe)]
    assert len(lignes) == 1, (prefixe, sortie[-800:])
    return lignes[0]


@pytest.fixture
def faux(tmp_path, monkeypatch):
    b = tmp_path / "bin"; b.mkdir()
    _exe(b / "faux-exec", FAUX_EXEC)
    # curl factice : 200 à toute sonde, un modèle servi « x » à /v1/models ; ss absent
    _exe(b / "curl", "#!/bin/bash\ncase \"$*\" in *-w*) printf 200 ;; *models*) printf '{\"data\":[{\"id\":\"x\"}]}' ;; esac\n")
    monkeypatch.setenv("PATH", f"{b}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home")); (tmp_path / "home").mkdir()
    return b


def test_acvram_serveur_imprime_ce_qu_il_execute(tmp_path, faux):
    dossier = tmp_path / "modele"; dossier.mkdir()
    env = {**os.environ, "ACVRAM_PAQUET_BIN": str(faux / "faux-exec"), "GRAPHES": "1", "SPECULATIF": "mtp", "CTX": "4096"}
    a_sec = subprocess.run(["bash", str(PARC / "bin" / "acvram-serveur"), str(dossier)], capture_output=True, text=True,
                           env={**env, "ACVRAM_SERVEUR_A_SEC": "1"}, timeout=30)
    recu = subprocess.run(["bash", str(PARC / "bin" / "acvram-serveur"), str(dossier)], capture_output=True, text=True,
                          env={**env, "ACVRAM_EXEC": str(faux / "faux-exec")}, timeout=30)
    assert a_sec.returncode == 0 and recu.returncode == 0, a_sec.stderr + recu.stderr
    imprimee, recue = _ligne(a_sec.stdout), _ligne(recu.stdout, "recu :")
    assert imprimee == recue, (imprimee, recue)
    assert "--speculative mtp" in recue and "--no-cuda-graphs" in recue and "--max-model-len 4096" in recue


@pytest.mark.parametrize("alias", ["acvram-un", "rapide-un", "vllm-un", "llamacpp-un"])
def test_claude_modele_imprime_ce_qu_il_execute_pour_les_quatre_moteurs(tmp_path, faux, alias):
    home = Path(os.environ["HOME"]); tsv = home / "TSV"; tsv.mkdir()
    dossier = tmp_path / "modele"; dossier.mkdir(); (dossier / "m.gguf").write_bytes(b"\0" * (2 << 20))
    # ctx ≥ 19096 (PROMPT_BASE=15000 + MIN_REPONSE=4096) — sinon lancer_claude refuse (D2)
    (tsv / "acvram-chemins.tsv").write_text(f"acvram-un\t{dossier}\t32768\n")
    (tsv / "vllm-chemins.tsv").write_text(f"vllm-un\t{dossier}\t32768\n")
    (tsv / "gguf-chemins.tsv").write_text(f"gguf-un\t{dossier}\t32768\t\n".replace("gguf-un", "llamacpp-un"))
    for lanceur in ("acvram-serveur", "llamacpp-appoint", "vllm-serveur", "llamacpp-serveur"):
        _exe(faux / lanceur, "#!/bin/bash\nexit 0\n")
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\ntsv_dir = "{tsv}"\nbin = "{faux}"\nmcp_claude = "{tmp_path}/mcp.json"\nsecrets = "{tmp_path}/absent.env"\n'
                   f'lib = "{PARC}/lib/kimi-menu.lib.sh"\n[outils]\nclaude = "{faux}/faux-exec"\n')
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(cfg)}
    affiche = subprocess.run(["bash", str(PARC / "bin" / "claude-modele"), "--afficher", alias, "--verbose"],
                             capture_output=True, text=True, env=env, timeout=60)
    recu = subprocess.run(["bash", str(PARC / "bin" / "claude-modele"), alias, "--verbose"],
                          capture_output=True, text=True, env={**env, "PARC_EXEC": str(faux / "faux-exec")}, timeout=60)
    assert affiche.returncode == 0 and recu.returncode == 0, affiche.stderr[-600:] + recu.stderr[-600:]
    imprimee, recue = _ligne(affiche.stdout), _ligne(recu.stdout, "recu :")
    assert imprimee == recue, (imprimee, recue)
    assert imprimee.endswith(" --verbose") and "--mcp-config" in imprimee
