"""29/09 (edz liste-1) : kimi rc 1 « 401 clé d'API invalide » sur tous les moteurs hors acvram. kimi-routeur (8790)
n'accepte qu'une clé égale à une CLE_* de ia-secrets.env ; parc-installer a créé ce fichier le 28/09 avec des clés
aléatoires, et config.toml gardait les défauts d'avant (vllm-local…). kimi-modele reporte désormais les CLE_* courantes
dans les fournisseurs routés de la copie LOCALE (~/.kimi-code-local), jamais dans la config de l'utilisateur. À sec."""
import os
import stat
import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
PARC = RACINE / "parc"

CONFIG = """default_model = "acvram-un"

[providers.acvram]
type = "openai"
base_url = "http://127.0.0.1:8090/v1"
api_key = "acvram-local"

[providers.llamacpp]
type = "openai"
base_url = "http://127.0.0.1:8790/llamacpp/v1"
api_key = "llamacpp-ancienne"

[providers.vllm]
type = "openai"
base_url = "http://127.0.0.1:8790/vllm/v1"
api_key = "vllm-local"

[providers.rapide]
type = "openai"
base_url = "http://127.0.0.1:8790/rapide/v1"
api_key = "rapide-ancienne"

[providers.tabby]
type = "openai"
base_url = "http://127.0.0.1:8790/tabby/v1"
api_key = "tabby-ancienne"

[models.acvram-un]
provider = "acvram"
model = "acvram-un"
max_context_size = 32768
"""


def _exe(p: Path, corps: str) -> Path:
    p.write_text(corps); p.chmod(p.stat().st_mode | stat.S_IEXEC); return p


def test_la_copie_locale_prend_les_cles_courantes(tmp_path):
    # Casse si kimi-modele recopie config.toml sans reporter les CLE_* (401 au routeur après une rotation des secrets)
    home = tmp_path / "home"; kimi_dir = home / ".kimi-code"; kimi_dir.mkdir(parents=True)
    (kimi_dir / "config.toml").write_text(CONFIG)
    tsv = home / "TSV"; tsv.mkdir(); dossier = tmp_path / "modele"; dossier.mkdir()
    (tsv / "acvram-chemins.tsv").write_text(f"acvram-un\t{dossier}\t32768\n")
    secrets = tmp_path / "ia-secrets.env"
    # CLE_TABBY de ia-secrets.env est un leurre (30/09, edz définitif tabby 0/17) : TabbyAPI et le routeur ne connaissent
    # que la clé d'api_tokens.yml — c'est ELLE qui doit entrer dans la copie, jamais la valeur des secrets
    secrets.write_text("CLE_VLLM=0123456789abcdef\nCLE_LLAMACPP=fedcba9876543210\nCLE_RAPIDE=00112233aabbccdd\n"
                       "CLE_TABBY=leurre-des-secrets\n")
    tokens = tmp_path / "api_tokens.yml"
    tokens.write_text("api_key: cle-du-fichier-tabby\nadmin_key: admin-du-fichier\n")
    b = tmp_path / "bin"; b.mkdir()
    _exe(b / "acvram-serveur", "#!/bin/bash\nexit 0\n")
    _exe(b / "curl", "#!/bin/bash\ncase \"$*\" in *models*) printf '%s' "
         "'{\"data\":[{\"id\":\"acvram-un\",\"acvram\":{\"max_model_len\":32768}}]}' ;; esac\n")
    _exe(b / "faux-kimi", "#!/bin/bash\necho ok\n")
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\nkimi_dir = "{kimi_dir}"\ntsv_dir = "{tsv}"\nbin = "{b}"\nsecrets = "{secrets}"\n'
                   f'lib = "{PARC}/share/kimi-menu.lib.sh"\n[outils]\nkimi = "{b}/faux-kimi"\n[moteurs.acvram]\npresent = true\nport = 8090\n'
                   f'[moteurs.tabby]\npresent = true\ntokens = "{tokens}"\n')
    env = {**os.environ, "HOME": str(home), "PATH": f"{b}:/usr/bin:/bin", "ACVRAM_PARC_CONFIG": str(cfg)}
    for k in ("KIMI_CODE_HOME", "CLE_VLLM", "CLE_LLAMACPP", "CLE_RAPIDE", "CLE_TABBY"):
        env.pop(k, None)
    r = subprocess.run(["bash", str(PARC / "bin" / "kimi-modele"), "acvram-un", "-p", "OK"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-600:]
    copie = (home / ".kimi-code-local" / "config.toml").read_text()
    attendu = (CONFIG.replace('api_key = "vllm-local"', 'api_key = "0123456789abcdef"')
                     .replace('api_key = "llamacpp-ancienne"', 'api_key = "fedcba9876543210"')
                     .replace('api_key = "tabby-ancienne"', 'api_key = "cle-du-fichier-tabby"')
                     .replace('api_key = "rapide-ancienne"', 'api_key = "00112233aabbccdd"'))
    assert copie == attendu                                   # acvram (direct) et le reste intacts
    assert (kimi_dir / "config.toml").read_text() == CONFIG  # config de l'utilisateur intacte
