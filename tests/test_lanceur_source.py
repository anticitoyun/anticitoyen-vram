"""acvram-serveur (parc/bin) à sec : paquet par défaut, opt-in ACVRAM_ARBRE, ligne « source= » à deux valeurs.
Rien n'est servi : ACVRAM_SERVEUR_A_SEC=1 arrête le lanceur juste avant `serve` ; faux binaires dans un tmp ; port 8090 non touché
(le lanceur sonde /v1/models par curl : ici PORT est celui d'un port fermé pour ne rien tuer)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

LANCEUR = Path(__file__).resolve().parent.parent / "parc" / "bin" / "acvram-serveur"


@pytest.fixture
def poste(tmp_path):
    home = tmp_path / "home"; (home / "TSV").mkdir(parents=True)
    modele = tmp_path / "Modele-nvfp4"; modele.mkdir(); (modele / "config.json").write_text("{}")
    (home / "TSV" / "acvram-chemins.tsv").write_text(f"acvram-essai\t{modele}\t32768\n")
    paquet = tmp_path / "paquet-acvram"; paquet.write_text("#!/bin/sh\n[ \"$1\" = --version ] && echo 'acvram 9.9.9'\n"); paquet.chmod(0o755)
    arbre = tmp_path / "arbre"; (arbre / ".venv" / "bin").mkdir(parents=True); (arbre / "acvram").mkdir()
    (arbre / ".venv" / "bin" / "acvram").write_text("#!/bin/sh\necho arbre\n"); (arbre / ".venv" / "bin" / "acvram").chmod(0o755)
    subprocess.run(["git", "init", "-q", "-b", "main", str(arbre)], check=True)
    subprocess.run(["git", "-C", str(arbre), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    # le lanceur lit ~/TSV et tue ce qui écoute sur $PORT : HOME factice, port fermé
    src = LANCEUR.read_text().replace("PORT=${PARC_PORT_ACVRAM:-8090}", "PORT=1")
    assert "PORT=1" in src, "fixture : la ligne PORT du lanceur a changé — le test viserait le vrai port"
    lanceur = tmp_path / "acvram-serveur"; lanceur.write_text(src); lanceur.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "ACVRAM_SERVEUR_A_SEC": "1", "ACVRAM_PAQUET_BIN": str(paquet),
           "ACVRAM_ATTENTE_VRAM": "0"}   # kwh : la vraie carte hors du test ; test_attente_carte_kwh la simule
    env.pop("ACVRAM_ARBRE", None)
    return {"lanceur": lanceur, "env": env, "arbre": arbre, "paquet": paquet}


def _run(p, **sup):
    return subprocess.run(["bash", str(p["lanceur"]), "acvram-essai"], capture_output=True, text=True, env={**p["env"], **sup}, timeout=60)


def test_paquet_par_defaut(poste):
    r = _run(poste)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "acvram : source=paquet(9.9.9)" in r.stdout
    assert f"commande : {poste['paquet']} serve" in r.stdout and "--served-name acvram-essai" in r.stdout


def test_arbre_opt_in_propre_puis_sale(poste):
    sha = subprocess.run(["git", "-C", str(poste["arbre"]), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    r = _run(poste, ACVRAM_ARBRE=str(poste["arbre"]))
    assert r.returncode == 0 and f"acvram : source=arbre(main@{sha},propre)" in r.stdout, r.stdout + r.stderr
    assert f"commande : {poste['arbre']}/.venv/bin/acvram serve" in r.stdout
    (poste["arbre"] / "acvram" / "x.py").write_text("# modif non commitée\n")
    r2 = _run(poste, ACVRAM_ARBRE=str(poste["arbre"]))
    assert f"source=arbre(main@{sha},sale)" in r2.stdout, r2.stdout


def test_faute_construite_paquet_absent_et_arbre_sans_venv(poste, tmp_path):
    r = _run(poste, ACVRAM_PAQUET_BIN=str(tmp_path / "absent"))
    assert r.returncode == 1 and "paquet acvram absent" in r.stderr
    r2 = _run(poste, ACVRAM_ARBRE=str(tmp_path / "pas-un-depot"))
    assert r2.returncode == 1 and "pas de .venv/bin/acvram" in r2.stderr


def test_a_sec_ne_lance_rien(poste):
    """Aucun processus « serve » ne survit au lanceur à sec."""
    _run(poste)
    ps = subprocess.run(["pgrep", "-af", "acvram-essai"], capture_output=True, text=True).stdout
    assert "serve" not in ps, ps


def test_garde_vram_dit_nvidia_smi_absent(poste, tmp_path):
    """nvidia-smi absent du PATH : la garde le DIT (une ligne), ne passe plus en silence."""
    modele = tmp_path / "Modele-nvfp4"  # créé par la fixture avec config.json = {}
    (modele / "config.json").write_text(
        '{"num_hidden_layers": 28, "num_key_value_heads": 8, "num_attention_heads": 32, "hidden_size": 4096}')
    bindir = tmp_path / "bin-sans-nvidia-smi"; bindir.mkdir()
    # le PATH réduit garde tout ce dont le lanceur a besoin, bash et env compris (shebang), sauf nvidia-smi
    import shutil
    for outil in ("bash", "env", "grep", "cut", "sed", "tr", "head", "cat", "dirname", "readlink",
                  "python3", "curl", "ss", "awk"):
        cible = shutil.which(outil)
        if cible:
            (bindir / outil).symlink_to(cible)
    r = _run(poste, PATH=str(bindir))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nvidia-smi introuvable — garde VRAM inactive" in r.stdout, r.stdout


# --- t5e (27/09) : quatre pannes du menu trouvées à sec, chaque test rouge sur le lanceur d'avant -----------------

def test_sans_cuda_visible_devices(poste):
    """Environnement de bureau : CUDA_VISIBLE_DEVICES non défini. Avant : `set -u` tuait le lanceur dans la garde
    VRAM (rc 1, « variable sans liaison »), l'épinglage n'étant posé qu'après."""
    env = dict(poste["env"]); env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run(["bash", str(poste["lanceur"]), "acvram-essai"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cartes=0" in r.stdout


def test_version_sur_deux_lignes(poste):
    """--version imprime aussi la mention de soutien : « source= » doit rester sur une ligne."""
    poste["paquet"].write_text("#!/bin/sh\n[ \"$1\" = --version ] && printf 'acvram 9.9.9\\nbuymeacoffee.com/x\\n'\n")
    r = _run(poste)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "acvram : source=paquet(9.9.9)\n" in r.stdout, r.stdout


def test_cwd_dans_un_arbre(poste):
    """Lancé depuis un arbre acvram, le paquet refuse l'import (garde d'arbre) : --version rend 1. Avant : le lanceur
    mourait en silence (set -e) et le service aurait hérité du même cwd."""
    poste["paquet"].write_text("#!/bin/sh\n[ \"$PWD\" = / ] || exit 1\n[ \"$1\" = --version ] && echo 'acvram 9.9.9'\n")
    r = subprocess.run(["bash", str(poste["lanceur"]), "acvram-essai"], capture_output=True, text=True,
                       env=poste["env"], cwd=str(poste["arbre"]), timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "source=paquet(9.9.9)" in r.stdout


def test_garde_compte_la_vram_du_serveur_remplace(poste, tmp_path):
    """Changer d'alias quand un modèle acvram occupe la carte : la VRAM du serveur en place sera rendue. Avant : la
    garde ne comptait que la VRAM libre et refusait tout changement d'alias à grand contexte."""
    (tmp_path / "Modele-nvfp4" / "config.json").write_text(
        '{"num_hidden_layers": 48, "num_key_value_heads": 4, "num_attention_heads": 32, "hidden_size": 4096}')
    ancien = subprocess.Popen(["sleep", "60"])
    try:
        faux = tmp_path / "faux"; faux.mkdir()
        (faux / "ss").write_text(f"#!/bin/sh\necho 'LISTEN 0 5 127.0.0.1:1 0.0.0.0:* users:((\"x\",pid={ancien.pid},fd=3))'\n")
        (faux / "nvidia-smi").write_text(
            "#!/bin/sh\ncase \"$*\" in *memory.free*) echo 1000;; *compute-apps*) echo "
            f"'{ancien.pid}, 30000';; esac\n")
        for f in faux.iterdir():
            f.chmod(0o755)
        # KV fp16 à 32 768 : 2 × 48 × 4 × 128 × 32768 × 2 = 3 Gio > 1 000 Mio libres, < 31 000 avec l'ancien serveur
        r = _run(poste, PATH=f"{faux}:{os.environ['PATH']}")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Alias refusé" not in r.stderr
    finally:
        ancien.kill(); ancien.wait()


def test_mort_au_demarrage_dite_sans_attendre(poste, tmp_path):
    """t5e 27/09 : un serveur qui meurt au démarrage (refus de budget KV) laissait le menu attendre 240 s. Le lanceur
    surveille le PID et rend 1 aussitôt. Rouge avant : timeout du test (30 s) au lieu d'un rc 1 nommé."""
    poste["paquet"].write_text("#!/bin/sh\n[ \"$1\" = --version ] && { echo 'acvram 9.9.9'; exit 0; }\nexit 3\n")
    env = {**poste["env"], "ACVRAM_CARTE_SH": str(tmp_path / "absent")}
    env.pop("ACVRAM_SERVEUR_A_SEC")
    try:
        r = subprocess.run(["bash", str(poste["lanceur"]), "acvram-essai"], capture_output=True, text=True, env=env, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("le lanceur attend encore un serveur mort")
    assert r.returncode == 1 and "acvram mort au démarrage" in r.stderr, r.stderr
