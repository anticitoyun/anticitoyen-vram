"""i61 (29/09) : parc-installer ne CRÉE plus de clés aléatoires quand des clients existent. Le 28/09 11 h 42 il a écrit
ia-secrets.env en token_hex sur un poste dont kimi et Open WebUI portaient les défauts : 401 partout
(revue/poste5-cles-verdict-29-09.md). Absent : clés des clients (config kimi, openwebui.env), défaut sinon ; conflit : rc 4."""
import re

import tomllib

from test_paquet_parc import _installer, poste  # noqa: F401 (fixture)


def _secrets(home):
    return dict(l.split("=", 1) for l in (home / ".config" / "ia-secrets.env").read_text().splitlines() if "=" in l)


def _kimi_llamacpp(home, cle):
    cfg = home / ".kimi-code" / "config.toml"
    cfg.write_text(cfg.read_text() + f'\n[providers.llamacpp]\ntype = "openai"\nbase_url = "http://127.0.0.1:8080/v1"\n'
                                      f'api_key = "{cle}"\n')


def _owui(poste, cle):
    d = poste["home"] / "owui"; d.mkdir()
    (d / "openwebui.env").write_text(f'OPENAI_API_BASE_URLS="http://127.0.0.1:8790/llamacpp/v1"\nOPENAI_API_KEYS="{cle}"\n')
    poste["cfg"].parent.mkdir(parents=True, exist_ok=True)
    poste["cfg"].write_text(f'[extras]\nopenwebui_dir = "{d}"\n')


def test_cle_du_client_kimi_reprise_jamais_aleatoire(poste):
    # Casse sur l'ancien parc-installer : CLE_LLAMACPP = token_hex(8), le client kimi en 401
    _kimi_llamacpp(poste["home"], "cle-du-client-kimi")
    r = _installer(poste)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]
    assert _secrets(poste["home"])["CLE_LLAMACPP"] == "cle-du-client-kimi"
    assert "clés reprises des clients" in r.stdout


def test_sans_cle_client_le_defaut_des_lanceurs_et_kimi_accorde(poste):
    # kimi existe sans [providers.llamacpp] : la fusion l'ajoute avec la MÊME clé que le fichier de secrets
    r = _installer(poste)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]
    cle = _secrets(poste["home"])["CLE_LLAMACPP"]
    assert not re.fullmatch(r"[0-9a-f]{16}", cle), "clé aléatoire alors qu'un client existe"
    kimi = tomllib.loads((poste["home"] / ".kimi-code" / "config.toml").read_text())
    assert kimi["providers"]["llamacpp"]["api_key"] == cle


def test_openwebui_et_kimi_d_accord(poste):
    _kimi_llamacpp(poste["home"], "meme-cle"); _owui(poste, "meme-cle")
    r = _installer(poste)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]
    assert _secrets(poste["home"])["CLE_LLAMACPP"] == "meme-cle"


def test_conflit_entre_clients_refus_rc4_sans_cle_ecrite(poste):
    _kimi_llamacpp(poste["home"], "cle-kimi"); _owui(poste, "cle-owui")
    avant = (poste["home"] / ".kimi-code" / "config.toml").read_text()
    r = _installer(poste)
    assert r.returncode == 4, r.stdout[-1500:]
    assert not (poste["home"] / ".config" / "ia-secrets.env").exists()
    assert (poste["home"] / ".kimi-code" / "config.toml").read_text() == avant
    assert "REFUS" in r.stdout and "CLE_LLAMACPP" in r.stdout and "À faire" in r.stdout
    assert "cle-kimi" not in r.stdout and "cle-owui" not in r.stdout, "une clé imprimée"


def test_secrets_existants_jamais_touches(poste):
    s = poste["home"] / ".config" / "ia-secrets.env"; s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text("CLE_LLAMACPP=celle-de-l-utilisatrice\n")
    r = _installer(poste)
    assert r.returncode == 0 and s.read_text() == "CLE_LLAMACPP=celle-de-l-utilisatrice\n"
    kimi = tomllib.loads((poste["home"] / ".kimi-code" / "config.toml").read_text())
    assert kimi["providers"]["llamacpp"]["api_key"] == "celle-de-l-utilisatrice"
