"""e50.3 § 2 bis (précision chef, 01/10) : chaque item HumanEval s'exécute sous bwrap, jamais
nu — ce test rejoue les 6 tentatives hostiles nommées dans la méthode (ouvrir une socket vers un
port local ET un DNS externe, écrire hors de /tmp, lire un secret, boucler, allouer trop, forker
trop) : chacune doit ÉCHOUER (rc != 0), et un code honnête doit PASSER (rc 0, stdout lu). Le test
casse si une ligne de `bwrap` disparaît — vérifié en retirant `--unshare-all` (réseau de nouveau
ouvert) et `--remount-ro "$HOME"` (écriture dans $HOME de nouveau permise), voir le verdict."""
import os
import shutil
import subprocess
import textwrap
import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ICI, "outils", "bac-a-sable-humaneval.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap (bubblewrap) absent")

CODES_HOSTILES = {
    "reseau_local": textwrap.dedent("""
        import socket
        socket.create_connection(("127.0.0.1", 8081), timeout=2)
        print("CONNECTE")
    """),
    "reseau_externe": textwrap.dedent("""
        import socket
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        print("CONNECTE")
    """),
    "ecrit_home": textwrap.dedent("""
        import os
        with open(os.path.expanduser("~/essai"), "w") as f:
            f.write("x")
        print("ECRIT")
    """),
    "ecrit_etc": textwrap.dedent("""
        with open("/etc/essai", "w") as f:
            f.write("x")
        print("ECRIT")
    """),
    "ecrit_mnt": textwrap.dedent("""
        with open("/mnt/essai", "w") as f:
            f.write("x")
        print("ECRIT")
    """),
    "lit_secret": textwrap.dedent("""
        import os
        with open(os.path.expanduser("~/.config/acvram/gitlab-token.gpg"), "rb") as f:
            print(len(f.read()))
    """),
    "boucle": "import time\ntime.sleep(60)\n",
    "alloue_trop": textwrap.dedent("""
        b = bytearray(4 * 1024**3)
        print("ALLOUE")
    """),
    "fork_trop": textwrap.dedent("""
        import os
        for _ in range(1000):
            os.fork()
    """),
}

CODE_HONNETE = textwrap.dedent("""
    def inc(x):
        return x + 1
    assert inc(1) == 2
    print("OK")
""")


def _jouer(tmp_path, nom, code):
    f = tmp_path / f"{nom}.py"
    f.write_text(code)
    env = {**os.environ, "PY_BAC_A_SABLE": "/usr/bin/python3"}
    return subprocess.run(["bash", SCRIPT, str(f)], env=env,
                          capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize("nom,code", list(CODES_HOSTILES.items()))
def test_tentative_hostile_echoue(tmp_path, nom, code):
    r = _jouer(tmp_path, nom, code)
    assert r.returncode != 0, (
        f"tentative « {nom} » a RÉUSSI dans le bac à sable (rc 0, stdout={r.stdout!r}) — "
        "le bac à sable ne protège plus")


def test_code_honnete_passe(tmp_path):
    r = _jouer(tmp_path, "honnete", CODE_HONNETE)
    assert r.returncode == 0, f"code honnête refusé : {r.stderr}"
    assert "OK" in r.stdout, r.stdout
