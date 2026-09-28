"""anticitoyen-vram-51x (28/09) : le guetteur de la campagne 290 (poste2) a
appelé liberer-vram, qui a tué le pytest d'une prise de MESURE de chef
(06:41:18, SIGTERM rc 143) — aucun port en écoute, donc « orphelin » aux
yeux de l'ancien script, alors que la carte était tenue par carte.sh.

Test cassant : rejoué sur l'ANCIEN script (~/.local/bin/liberer-vram
d'avant ce correctif, aucune lecture du `.qui`), il tue le processus
protégé — la preuve vit dans `test_ancien_script_tuerait_ce_meme_processus`.

`nvidia-smi` et `ss` factices (PATH), `.qui` factice (LIBERER_VRAM_VERROUS,
jamais les vrais /tmp/acvram-carte-*), aucune carte réelle requise. Le
processus protégé n'est pas le pid PORTÉ par le `.qui` mais un DESCENDANT
(même pgid, pid différent) — exactement la forme de la 51x : un pytest lance
des enfants qui touchent le GPU sous des pid distincts du sien."""
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

DEPOT = Path(__file__).resolve().parent.parent
SCRIPT = DEPOT / "parc" / "bin" / "liberer-vram"
ANCIEN_SCRIPT = Path.home() / ".local" / "bin" / "liberer-vram"


def _faux_bin(tmp_path, pid, vram=1298):
    d = tmp_path / "faux-bin"
    d.mkdir(exist_ok=True)
    nvidia_smi = d / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        '  *query-gpu*) echo "0, 1298, 22000" ;;\n'
        f'  *query-compute-apps*) echo "{pid}, faux-mesure, {vram}" ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IEXEC)
    ss = d / "ss"
    ss.write_text("#!/bin/bash\nexit 0\n")  # aucune écoute, jamais « en service »
    ss.chmod(ss.stat().st_mode | stat.S_IEXEC)
    return str(d) + os.pathsep + os.environ["PATH"]


def _lancer_prise_et_descendant(tmp_path):
    """P (setsid, sa propre session/pgid) lance un enfant B (même pgid, pid
    différent) qui « tient la carte » — B est le pid que le faux nvidia-smi
    rapporte. Rend (proc_P, pid_B)."""
    marqueur = tmp_path / "b.pid"
    proc = subprocess.Popen(
        ["bash", "-c", f'echo $$; sleep 30 & echo $! > "{marqueur}"; wait'],
        start_new_session=True, stdout=subprocess.PIPE, text=True)
    pid_a = int(proc.stdout.readline().strip())
    deadline = time.time() + 5
    while not marqueur.exists() and time.time() < deadline:
        time.sleep(0.05)
    pid_b = int(marqueur.read_text().strip())
    return proc, pid_a, pid_b


def _ecrire_qui(tmp_path, pid_a):
    verrou = tmp_path / "carte-0.lock"
    (tmp_path / "carte-0.lock.qui").write_text(f"{pid_a} 1234567890 chef-suite-parc mesure {pid_a}\n")
    return verrou


def _nettoyer(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def test_descendant_de_la_prise_nest_jamais_tue(tmp_path):
    proc, pid_a, pid_b = _lancer_prise_et_descendant(tmp_path)
    try:
        verrou = _ecrire_qui(tmp_path, pid_a)
        journal = tmp_path / "journal.tsv"
        env = {**os.environ, "PATH": _faux_bin(tmp_path, pid_b),
               "LIBERER_VRAM_VERROUS": str(verrou), "LIBERER_VRAM_JOURNAL": str(journal)}
        r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert "protégé (prise carte.sh)" in r.stdout, r.stdout
        assert "0 processus arrêté" in r.stdout, r.stdout

        time.sleep(0.3)
        os.kill(pid_b, 0)  # ne lève pas : encore vivant

        assert journal.exists()
        ligne = journal.read_text()
        assert str(pid_b) in ligne and "protégé" in ligne, ligne
    finally:
        _nettoyer(proc)


def test_sans_qui_le_meme_processus_est_tue(tmp_path):
    """Contrôle négatif : sans entrée .qui, le nouveau script tue toujours un
    vrai orphelin — la protection ne bloque pas tout, juste ce qui est tenu."""
    proc, pid_a, pid_b = _lancer_prise_et_descendant(tmp_path)
    try:
        verrou = tmp_path / "aucun-carte.lock"  # jamais écrit : aucune prise déclarée
        env = {**os.environ, "PATH": _faux_bin(tmp_path, pid_b),
               "LIBERER_VRAM_VERROUS": str(verrou), "LIBERER_VRAM_JOURNAL": str(tmp_path / "j2.tsv")}
        r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert "1 processus arrêté" in r.stdout, r.stdout

        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid_b, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"pid {pid_b} encore vivant — orphelin non tué")
    finally:
        _nettoyer(proc)


def test_ancien_script_tuerait_ce_meme_processus(tmp_path):
    """Cassant : preuve que ~/.local/bin/liberer-vram d'AVANT ce correctif
    (aucune lecture du `.qui`) tue le processus que le nouveau script
    protège — sinon ce fichier ne prouverait rien de la 51x. Sauté si le
    lien n'existe plus (déjà basculé vers parc/bin par ce correctif)."""
    import pytest
    if not ANCIEN_SCRIPT.exists() or ANCIEN_SCRIPT.resolve() == SCRIPT.resolve():
        pytest.skip("~/.local/bin/liberer-vram déjà basculé vers parc/bin (correctif appliqué)")
    proc, pid_a, pid_b = _lancer_prise_et_descendant(tmp_path)
    try:
        env = {**os.environ, "PATH": _faux_bin(tmp_path, pid_b)}
        r = subprocess.run(["bash", str(ANCIEN_SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
        assert "1 processus arrêté" in r.stdout, r.stdout  # tué, aucune notion de .qui
    finally:
        _nettoyer(proc)
