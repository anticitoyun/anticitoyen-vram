"""poste7 21/09 (§ 1b du feu vert 0.6.34) : la commande que la GUI IMPRIME est, octet pour octet, celle que
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
    # ctx ≥ 29096 (PROMPT_BASE=25000 + MIN_REPONSE=4096) — sinon lancer_claude refuse (D2)
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


def _poste_acvram(tmp_path, faux, ctx_tsv, reponse_models):
    home = Path(os.environ["HOME"]); tsv = home / "TSV"; tsv.mkdir()
    dossier = tmp_path / "modele"; dossier.mkdir(exist_ok=True)
    (tsv / "acvram-chemins.tsv").write_text(f"acvram-un\t{dossier}\t{ctx_tsv}\n")
    _exe(faux / "acvram-serveur", "#!/bin/bash\nexit 0\n")
    _exe(faux / "curl", "#!/bin/bash\ncase \"$*\" in *-w*) printf 200 ;; *models*) printf '%s' '" + reponse_models + "' ;; esac\n")
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\ntsv_dir = "{tsv}"\nbin = "{faux}"\nmcp_claude = "{tmp_path}/mcp.json"\nsecrets = "{tmp_path}/absent.env"\n'
                   f'lib = "{PARC}/lib/kimi-menu.lib.sh"\n[outils]\nclaude = "{faux}/faux-exec"\n')
    return {**os.environ, "ACVRAM_PARC_CONFIG": str(cfg)}


def test_claude_modele_mcp_strict(tmp_path, faux):
    """t5e 27/09 : sans --strict-mcp-config, les MCP globaux de l'utilisateur s'ajoutaient à mcp.json (170 outils,
    invite 61 315 jetons) → « Prompt is too long » sur tout alias local."""
    env = _poste_acvram(tmp_path, faux, 32768, '{"data":[{"id":"acvram-un"}]}')
    r = subprocess.run(["bash", str(PARC / "bin" / "claude-modele"), "--afficher", "acvram-un"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr[-600:]
    assert "--strict-mcp-config --mcp-config" in _ligne(r.stdout)


def test_claude_modele_fenetre_servie(tmp_path, faux):
    """La fenêtre annoncée à claude est celle que le moteur TIENT (max_model_len de /v1/models, clampé par la
    chauffe), pas la demande du TSV — sinon claude remplit 65 536 et le moteur rend 400 au-delà de 39 936."""
    # forme réelle d'acvram (app.py : /v1/models → data[0].acvram.max_model_len), puis forme vLLM (à plat)
    for reponse in ('{"data":[{"id":"acvram-un","acvram":{"max_model_len":39936}}]}',
                    '{"data":[{"id":"acvram-un","max_model_len":39936}]}'):
        home = Path(os.environ["HOME"]); import shutil; shutil.rmtree(home / "TSV", ignore_errors=True)
        env = _poste_acvram(tmp_path, faux, 65536, reponse)
        r = subprocess.run(["bash", str(PARC / "bin" / "claude-modele"), "--afficher", "acvram-un"],
                           capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr[-600:]
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS=39936" in r.stdout, (reponse, r.stdout[-600:])
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" in r.stdout, r.stdout[-600:]


def test_claude_modele_sortie_dans_la_fenetre(tmp_path, faux):
    """t5e 27/09 : claude réserve 32 000 jetons de sortie dans la fenêtre ; à 32 768 il refusait sans rien envoyer.
    Sortie annoncée = fenêtre − PROMPT_BASE, plafonnée à 8 192. iqm 28/09 : à 32 768 le jeu est essentiel
    (PROMPT_BASE 14 000 → 8 192), et la commande porte --tools Read,Edit,Bash,Grep."""
    env = _poste_acvram(tmp_path, faux, 32768, '{"data":[{"id":"acvram-un"}]}')
    r = subprocess.run(["bash", str(PARC / "bin" / "claude-modele"), "--afficher", "acvram-un"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr[-600:]
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" in r.stdout, r.stdout[-600:]
    assert "outils : essentiel Read, Edit, Bash, Grep (fenêtre 32768 < 45000)" in r.stdout
    assert r"--tools Read\,Edit\,Bash\,Grep" in r.stdout
