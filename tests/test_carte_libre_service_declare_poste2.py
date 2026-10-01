"""Signalé par poste2 (nuit du 27/09, ordre chef) : carte-libre.sh voyait un
llama-server en cours de chargement comme INTRUS alors que llamacpp-serveur
tenait la carte via `carte.sh ACVRAM_TYPE=service` — `service_declare` lisait
`outils/gpu/journal-services.tsv`, un fichier qu'aucun script du dépôt
n'écrit ; aucun service n'y était donc jamais déclaré. La vraie déclaration
vit dans le `.qui` de carte.sh (contrat 4 champs `<pid> <epoch> <nom>
service`, une ligne par service vivant). `nvidia-smi` factice (PATH), un seul
pid GPU simulé (celui du test lui-même), aucune carte réelle requise.

bd cn1 (chef, 01/10) : ces deux tests échouaient dans la suite complète pendant qu'un vrai
poste compilait des noyaux (nvcc/ninja, typique d'un chargement de modèle) — MOTIF était déjà
neutralisé ici (voir `_env`), mais le pgrep de DÉTECTION DE COMPILATION de `carte-libre.sh`
(3e critère, code de sortie 2) n'avait AUCUNE variable d'isolation, contrairement à MOTIF.
`MOTIF_COMPIL` corrige ça, neutralisé dans `_env()` au même titre.
`test_isolation_compil_casse_sans_motif_compil` est le contrôle de fuite demandé."""
import os
import signal
import stat
import subprocess
import time
import pytest

pytestmark = pytest.mark.usefixtures("recolte_carte")   # ked/7gb : aucun processus ne survit au test

DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(DEPOT, "outils", "carte-libre.sh")


def _faux_nvidia_smi(tmp_path, pid, mem=200, sm=5):
    d = tmp_path / "faux-bin"
    d.mkdir()
    f = d / "nvidia-smi"
    f.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *query-gpu=memory.used*) echo {mem} ;;\n'
        f'  *pmon*) printf "# gpu pid type sm mem enc dec command\\n0 {pid} C {sm} - - - faux\\n" ;;\n'
        f'  *query-compute-apps=pid,used_memory*) echo "{pid}, {mem}" ;;\n'
        f'  *query-compute-apps=pid*) echo "{pid}" ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(d) + os.pathsep + os.environ["PATH"]


def _env(tmp_path, pid, qui_carte=None):
    # MOTIF neutralisé : ce test juge le croisement nvidia-smi/.qui (critères 1-2),
    # pas le pgrep d'intentions (critère 3) — sans quoi une vraie campagne en
    # cours sur le poste (python outils/…) ferait échouer le test sans rapport
    # avec service_declare (non hermétique, même piège que la pièce 73c).
    env = {**os.environ, "RACINE": "1", "MOTIF": "__jamais_aucune_correspondance__",
           "MOTIF_COMPIL": "__jamais_aucune_correspondance__",
           "PATH": _faux_nvidia_smi(tmp_path, pid)}
    env.pop("ACVRAM_VERROU", None)
    if qui_carte is not None:
        env["QUI_CARTE"] = str(qui_carte)
    else:
        env["QUI_CARTE"] = str(tmp_path / "aucun.qui")  # inexistant : pas de déclaration
    return env


def _faux_serveur():
    """Un processus frère du script (PPID = pytest), jamais un ancêtre : le
    pid simulé ne doit pas être écarté comme « mien » avant même d'atteindre
    service_declare, sinon le test ne prouverait rien."""
    proc = subprocess.Popen(["sleep", "30"])
    time.sleep(0.2)
    return proc


def test_service_non_declare_reste_intrus(tmp_path):
    """Contrôle négatif : sans `.qui`, le pid GPU simulé reste un intrus —
    prouve que le test suivant réussit PARCE QU'il est déclaré, pas par hasard."""
    proc = _faux_serveur()
    try:
        r = subprocess.run([SCRIPT], cwd=DEPOT, env=_env(tmp_path, proc.pid),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 1
        assert "Intrus" in r.stderr, r.stderr
    finally:
        proc.send_signal(signal.SIGKILL); proc.wait(timeout=5)


def _lancer_leurre_compilation(tmp_path):
    """bd cn1 : nom du binaire « ninja » — motif par défaut de MOTIF_COMPIL. Tourne pendant
    le test ci-dessous ; sans l'isolation de `_env()`, il fait échouer le test (voir le
    contrôle `test_isolation_compil_casse_sans_motif_compil`)."""
    faux_ninja = tmp_path / "ninja"
    faux_ninja.write_text("#!/bin/sh\nsleep 30\n")
    faux_ninja.chmod(faux_ninja.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.Popen([str(faux_ninja)])
    time.sleep(0.2)
    return proc


def test_service_declare_dans_qui_carte_nest_plus_intrus(tmp_path):
    proc = _faux_serveur()
    leurre = _lancer_leurre_compilation(tmp_path)
    try:
        qui = tmp_path / "carte.lock.qui"
        qui.write_text(f"{proc.pid} 1234567890 llamacpp-serveur service\n")
        r = subprocess.run([SCRIPT], cwd=DEPOT, env=_env(tmp_path, proc.pid, qui_carte=qui),
                           capture_output=True, text=True, timeout=30)
        assert "Intrus" not in r.stderr, r.stderr
        assert r.returncode == 0, r.stderr
    finally:
        leurre.send_signal(signal.SIGKILL); leurre.wait(timeout=5)
        proc.send_signal(signal.SIGKILL); proc.wait(timeout=5)


def test_isolation_compil_casse_sans_motif_compil(tmp_path):
    """Contrôle demandé (chef, bd cn1) : le MÊME leurre « ninja » que ci-dessus, sans
    MOTIF_COMPIL dans l'env — prouve qu'il matche bien le motif réel de `carte-libre.sh` et
    ferait échouer `test_service_declare_dans_qui_carte_nest_plus_intrus` si l'isolation
    disparaissait de `_env()`."""
    proc = _faux_serveur()
    leurre = _lancer_leurre_compilation(tmp_path)
    try:
        qui = tmp_path / "carte.lock.qui"
        qui.write_text(f"{proc.pid} 1234567890 llamacpp-serveur service\n")
        env = _env(tmp_path, proc.pid, qui_carte=qui)
        env.pop("MOTIF_COMPIL")   # isolation retirée : le leurre doit être vu
        r = subprocess.run([SCRIPT], cwd=DEPOT, env=env,
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 2, (r.returncode, r.stderr)
        assert "processus de compilation en cours" in r.stderr, r.stderr
    finally:
        leurre.send_signal(signal.SIGKILL); leurre.wait(timeout=5)
        proc.send_signal(signal.SIGKILL); proc.wait(timeout=5)
