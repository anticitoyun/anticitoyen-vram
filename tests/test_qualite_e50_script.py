"""e50.3 § 7 (poste3, 01/10) : `outils/qualite-e50.sh` en `--simule` (défaut, REGLES §3 : aucune
carte, aucun processus) — résout un alias, détecte « sans raisonnement » (e50.1), refuse un
alias inconnu, refuse `--executer` sans la confirmation explicite.

Entièrement isolé du parc réel (chef, 02/10 : le parc réel avait perdu son alias
deepseek-coder-6-7b au nettoyage du 01/10, faisant échouer ce fichier sur une machine à jour) :
`ACVRAM_PARC_CONFIG` pointe un `parc.toml` jouet sous `tmp_path`, jamais `~/TSV` ni
`~/.kimi-code/config.toml` — même gabarit que `tests/test_qualite_e50_executer_faux_serveur.py`.
Passe sur une machine vierge (aucune dépendance au parc local), pourvu que GTK4/libadwaita
soient installés (requis par `parc.lire_tsv`, importé via `gi`)."""
import os
import subprocess
import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ICI, "outils", "qualite-e50.sh")

ALIAS_THINKING = "acvram-e50-test-thinking-nvfp4"
ALIAS_NON_THINKING = "acvram-e50-test-non-thinking-nvfp4"


def _gi_ok():
    r = subprocess.run(["/usr/bin/python3", "-c",
                       "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _gi_ok(), reason="GTK4/libadwaita absents")


def _un_alias(thinking_attendu):
    return ALIAS_THINKING if thinking_attendu else ALIAS_NON_THINKING


@pytest.fixture(scope="module")
def parc_toml(tmp_path_factory):
    """Un parc jouet, isolé, avec un alias thinking=True et un alias thinking=False — le
    `capabilities` de config.toml porte directement l'étiquette, sans dépendre d'un
    tokenizer_config.json réel (REGLES §3 : aucun fichier de modèle réel n'est requis)."""
    base = tmp_path_factory.mktemp("parc-e50-script")
    tsv_dir = base / "TSV"
    tsv_dir.mkdir()
    kimi_dir = base / "kimi"
    kimi_dir.mkdir()
    dossier_t = base / "modele-thinking"
    dossier_t.mkdir()
    dossier_nt = base / "modele-non-thinking"
    dossier_nt.mkdir()

    (tsv_dir / "acvram-chemins.tsv").write_text(
        f"{ALIAS_THINKING}\t{dossier_t}\t4096\n"
        f"{ALIAS_NON_THINKING}\t{dossier_nt}\t4096\n"
    )
    (tsv_dir / "gguf-chemins.tsv").write_text("")
    (tsv_dir / "vllm-chemins.tsv").write_text("")
    (tsv_dir / "vision-modeles.tsv").write_text("")
    (tsv_dir / "notes-modeles.tsv").write_text(
        f"{ALIAS_THINKING}\tnon\t?\tnon mesuré\tchat\n"
        f"{ALIAS_NON_THINKING}\tnon\t?\tnon mesuré\tchat\n"
    )
    (kimi_dir / "config.toml").write_text(
        f'[models."{ALIAS_THINKING}"]\n'
        f'provider = "acvram"\n'
        f'model = "FauxModeleThinking"\n'
        f'max_context_size = 4096\n'
        f'capabilities = ["thinking"]\n'
        f'\n'
        f'[models."{ALIAS_NON_THINKING}"]\n'
        f'provider = "acvram"\n'
        f'model = "FauxModeleNonThinking"\n'
        f'max_context_size = 4096\n'
        f'capabilities = []\n'
    )
    parc_toml = base / "parc.toml"
    parc_toml.write_text(
        f'[chemins]\n'
        f'kimi_dir = "{kimi_dir}"\n'
        f'tsv_dir = "{tsv_dir}"\n'
    )
    return parc_toml


def _env(parc_toml, **extra):
    return {**os.environ, "ACVRAM_PARC_CONFIG": str(parc_toml), **extra}


def test_simule_resout_un_alias_thinking(parc_toml):
    r = subprocess.run(["bash", SCRIPT, _un_alias(True)], env=_env(parc_toml),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "thinking  : True" in r.stdout, r.stdout
    assert "sans raisonnement si possible : oui" in r.stdout, r.stdout
    assert "rien lancé" in r.stdout


def test_simule_resout_un_alias_non_thinking(parc_toml):
    r = subprocess.run(["bash", SCRIPT, _un_alias(False)], env=_env(parc_toml),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "thinking  : False" in r.stdout, r.stdout
    assert "sans raisonnement si possible : non" in r.stdout, r.stdout


def test_refuse_un_alias_inconnu(parc_toml):
    r = subprocess.run(["bash", SCRIPT, "alias-qui-nexiste-pas-du-tout"], env=_env(parc_toml),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUS" in r.stderr


def test_refuse_executer_sans_confirmation(parc_toml):
    r = subprocess.run(["bash", SCRIPT, _un_alias(True), "--executer"], env=_env(parc_toml),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 65, r.stdout + r.stderr
    assert "je-sais-que-la-carte-est-libre" in r.stderr


def test_executer_resout_acvram_lmeval_py_en_priorite(parc_toml):
    """chef (01/10) : même ordre de résolution que le test de chargement réel —
    $ACVRAM_LMEVAL_PY prime sur tout, y compris sur un .venv-panel qui existerait.
    poste4 (01/10, implémentation du corps --executer) : ACVRAM_E50_LANCEUR=/bin/false
    EXIGÉ ici — sans ce garde-fou, --executer lancerait pour de vrai `acvram-serveur`
    (carte GPU réelle, hors de tout verrou carte.sh) juste pour vérifier une ligne
    d'impression ; /bin/false fait échouer bras_servir immédiatement, après que la ligne
    attendue a déjà été imprimée."""
    env = _env(parc_toml, ACVRAM_LMEVAL_PY="/usr/bin/python3", ACVRAM_E50_LANCEUR="/bin/false")
    r = subprocess.run(["bash", SCRIPT, _un_alias(True), "--executer",
                       "--je-sais-que-la-carte-est-libre"], env=env,
                       capture_output=True, text=True, timeout=30)
    assert "lm-eval : /usr/bin/python3" in r.stdout, r.stdout


def test_executer_refuse_si_aucun_interprete_lmeval(parc_toml):
    env = _env(parc_toml, ACVRAM_LMEVAL_PY="/tmp/acvram-e50-interprete-inexistant")
    r = subprocess.run(["bash", SCRIPT, _un_alias(True), "--executer",
                       "--je-sais-que-la-carte-est-libre"], env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 66, r.stdout + r.stderr
    assert "aucun interprète lm-eval trouvé" in r.stderr


def test_refuse_argument_inconnu(parc_toml):
    r = subprocess.run(["bash", SCRIPT, _un_alias(True), "--mode-bidon"], env=_env(parc_toml),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 64, r.stdout + r.stderr


# ---- pin : les limites par tâche ne changent pas sans qu'on les voie (REGLES) ----

def test_limites_par_tache_pin(parc_toml):
    """Les limites (30/30/30/40/40) sont des nombres en dur dans le script, jamais une variable
    surprise — ce test les lit dans le texte affiché par --simule et casse si quelqu'un les
    change : un changement est alors visible ICI, pas seulement dans le script."""
    r = subprocess.run(["bash", SCRIPT, _un_alias(True)], env=_env(parc_toml),
                       capture_output=True, text=True, timeout=30)
    ligne = next(l for l in r.stdout.splitlines() if l.startswith("tâches"))
    attendu = ("tâches    : mmlu_e50_hsm(30) mmlu_e50_law(30) mmlu_e50_ccs(30) "
              "gsm8k_e50(40) humaneval_e50(40)")
    assert ligne == attendu, f"limites changées sans que ce test le voie : {ligne!r}"
