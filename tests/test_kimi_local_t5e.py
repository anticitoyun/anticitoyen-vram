"""t5e 27/09 : kimi sur un alias LOCAL démarre sans MCP (invite 58 635 → 29 107 jetons, moteurs locaux 25-42 k).
KIMI_CODE_HOME dédié, régénéré : mcp.json vide, AGENTS.md lié, config = copie de l'utilisateur avec le seul
max_context_size de l'alias mis au contexte servi. La config de l'utilisateur reste intacte. À sec (faux kimi, faux curl)."""
import os
import stat
import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
PARC = RACINE / "parc"

CONFIG = """default_model = "distant"

[providers.acvram]
type = "openai"
base_url = "http://127.0.0.1:8090/v1"
api_key = "acvram-local"

[models.distant]
provider = "moonshot"
model = "kimi"
max_context_size = 262144

[models.acvram-un]
provider = "acvram"
model = "acvram-un"
max_context_size = 32768

[models.acvram-deux]
provider = "acvram"
model = "acvram-deux"
max_context_size = 32768
"""


def _exe(p: Path, corps: str) -> Path:
    p.write_text(corps); p.chmod(p.stat().st_mode | stat.S_IEXEC); return p


def test_kimi_local_sans_mcp_et_contexte_servi(tmp_path):
    home = tmp_path / "home"; kimi_dir = home / ".kimi-code"; kimi_dir.mkdir(parents=True)
    (kimi_dir / "config.toml").write_text(CONFIG)
    (kimi_dir / "mcp.json").write_text('{"mcpServers": {"tokensave": {}, "icm": {}}}')
    (kimi_dir / "AGENTS.md").write_text("consignes\n")
    tsv = home / "TSV"; tsv.mkdir(); dossier = tmp_path / "modele"; dossier.mkdir()
    (tsv / "acvram-chemins.tsv").write_text(f"acvram-un\t{dossier}\t32768\n")
    b = tmp_path / "bin"; b.mkdir()
    _exe(b / "acvram-serveur", "#!/bin/bash\nexit 0\n")
    _exe(b / "curl", "#!/bin/bash\ncase \"$*\" in *models*) printf '%s' "
         "'{\"data\":[{\"id\":\"acvram-un\",\"acvram\":{\"max_model_len\":39936}}]}' ;; esac\n")
    _exe(b / "faux-kimi", "#!/bin/bash\necho \"HOME_KIMI=$KIMI_CODE_HOME\"; printf 'recu :'; printf ' %q' \"$@\"; echo\n")
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\nkimi_dir = "{kimi_dir}"\ntsv_dir = "{tsv}"\nbin = "{b}"\nsecrets = "{tmp_path}/absent.env"\n'
                   f'lib = "{PARC}/share/kimi-menu.lib.sh"\n[outils]\nkimi = "{b}/faux-kimi"\n[moteurs.acvram]\npresent = true\nport = 8090\n')
    env = {**os.environ, "HOME": str(home), "PATH": f"{b}:/usr/bin:/bin", "ACVRAM_PARC_CONFIG": str(cfg)}
    env.pop("KIMI_CODE_HOME", None)
    r = subprocess.run(["bash", str(PARC / "bin" / "kimi-modele"), "acvram-un", "-p", "OK"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-600:]
    local = home / ".kimi-code-local"
    assert f"HOME_KIMI={local}" in r.stdout and "-m acvram-un -p OK" in r.stdout, r.stdout
    assert (local / "mcp.json").read_text().strip() == '{"mcpServers": {}}'
    assert (local / "AGENTS.md").resolve() == (kimi_dir / "AGENTS.md").resolve()
    copie = (local / "config.toml").read_text()
    # seul l'alias lancé prend le contexte servi ; le distant et l'autre alias local sont intacts
    assert copie == CONFIG.replace('model = "acvram-un"\nmax_context_size = 32768', 'model = "acvram-un"\nmax_context_size = 39936')
    assert (kimi_dir / "config.toml").read_text() == CONFIG                     # config de l'utilisateur intacte
    assert "tokensave" in (kimi_dir / "mcp.json").read_text()
