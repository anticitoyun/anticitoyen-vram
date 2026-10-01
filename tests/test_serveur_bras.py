"""Harnais commun `outils/gpu/mesure/serveur-bras.sh` (ordre de chef, 01/10, après la fenêtre poste1-5v7-bit perdue :
`$!` désignait le sous-shell de `cd && setsid … &`, le serveur du bras 1 a survécu et servi le bras 0). Hermétique :
faux serveur HTTP en Python sur un port libre, aucune carte ; nvidia-smi remplacé par un script qui rend un nombre fixe."""
import socket
import subprocess
import textwrap
from pathlib import Path

HARNAIS = Path(__file__).resolve().parent.parent / "outils" / "gpu" / "mesure" / "serveur-bras.sh"

FAUX = textwrap.dedent('''
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, nom = int(sys.argv[1]), sys.argv[2]
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"data": [{"id": nom}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(b)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bash(tmp_path, corps, env=None):
    # Sorties dans des fichiers, pas des tubes : un lanceur fautif (le sous-shell du 01/10) garde les tubes ouverts et
    # ferait BLOQUER le test au lieu de le rendre rouge. Tout faux serveur de ce test est tué en sortie.
    faux = tmp_path / "faux.py"
    faux.write_text(FAUX)
    script = f'set -u\n. "{HARNAIS}"\nFAUX="{faux}"\nT="{tmp_path}"\n' + textwrap.dedent(corps)
    out, err = tmp_path / "sortie", tmp_path / "erreurs"
    try:
        with open(out, "w") as o, open(err, "w") as e:
            rc = subprocess.run(["bash", "-c", script], stdout=o, stderr=e, timeout=60,
                                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path),
                                     **(env or {})}).returncode
    finally:
        subprocess.run(["pkill", "-f", str(faux)], capture_output=True)
    return rc, out.read_text(), err.read_text()


def _vivant(pid):
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


def test_servir_pret_arreter_vise_le_serveur_lui_meme(tmp_path):
    p = _port()
    rc, out, err = _bash(tmp_path, f'''
        bras_servir {p} "$T/a.log" python3 "$FAUX" {p} coder || exit 11
        bras_pret {p} coder "$BRAS_PID" 20 || exit 12
        echo "pid=$BRAS_PID"
        bras_ecoute_par {p} "$BRAS_PID" || exit 13
        bras_arreter "$BRAS_PID" {p} || exit 14
        bras_port_libre {p} || exit 15
    ''')
    assert rc == 0, (rc, err)
    pid = int(out.split("pid=")[1].split()[0])
    assert (tmp_path / "a.log.pid").read_text().strip() == str(pid)
    assert not _vivant(pid)


def test_un_bras_refuse_de_partir_si_le_port_repond_deja(tmp_path):
    """Le défaut du 01/10 : un serveur du bras précédent répond à la place du nouveau."""
    p = _port()
    rc, out, err = _bash(tmp_path, f'''
        bras_servir {p} "$T/a.log" python3 "$FAUX" {p} coder || exit 11
        bras_pret {p} coder "$BRAS_PID" 20 || exit 12
        A=$BRAS_PID
        bras_servir {p} "$T/b.log" python3 "$FAUX" {p} coder; r=$?
        bras_arreter "$A" {p}
        [ ! -e "$T/b.log.pid" ] || exit 13
        exit $r
    ''')
    assert rc == 1, (rc, err)
    assert "déjà servi" in err


def test_le_controle_d_arret_rend_faux_sur_le_pid_du_sous_shell(tmp_path):
    """La faute d'origine, rejouée : PID pris sur `( cd && setsid … & echo $! )`. L'arrêt doit le dire (rc 1)."""
    p = _port()
    rc, out, err = _bash(tmp_path, f'''
        ( cd "$T" && setsid python3 "$FAUX" {p} coder > /dev/null 2>&1 < /dev/null & echo $! > "$T/faux.pid" )
        for _ in $(seq 1 50); do bras_port_libre {p} || break; sleep 0.1; done
        BRAS_ARRET_S=2 bras_arreter "$(cat "$T/faux.pid")" {p}; r=$?
        pkill -f "$FAUX {p} coder"
        exit $r
    ''')
    assert rc == 1, (rc, err)
    assert "toujours servi" in err


def test_pret_refuse_un_autre_serveur_sur_le_port(tmp_path):
    p = _port()
    rc, out, err = _bash(tmp_path, f'''
        bras_servir {p} "$T/a.log" python3 "$FAUX" {p} coder || exit 11
        A=$BRAS_PID
        bras_pret {p} coder "$A" 20 || exit 12
        sleep 60 & B=$!
        bras_pret {p} coder "$B" 5; r=$?
        kill $B; bras_arreter "$A" {p}
        exit $r
    ''')
    assert rc == 1, (rc, err)
    assert "autre serveur" in err


def test_pret_rend_vite_si_le_serveur_meurt(tmp_path):
    rc, out, err = _bash(tmp_path, '''
        bras_servir 1 "$T/a.log" bash -c "exit 3" || exit 11
        s=$SECONDS; bras_pret 1 coder "$BRAS_PID" 60; r=$?
        [ $((SECONDS - s)) -lt 10 ] || exit 12
        exit $r
    ''')
    assert rc == 1, (rc, err)
    assert "mort" in err


def test_vram_non_rendue_bloque_le_bras_suivant(tmp_path):
    p = _port()
    smi = tmp_path / "smi"
    smi.write_text('#!/bin/sh\necho "$MIO"\n')
    smi.chmod(0o755)
    corps = f'''
        bras_servir {p} "$T/a.log" python3 "$FAUX" {p} coder || exit 11
        bras_pret {p} coder "$BRAS_PID" 20 || exit 12
        bras_arreter "$BRAS_PID" {p}
    '''
    env = {"BRAS_VRAM_MAX_MIO": "1024", "BRAS_NVIDIA_SMI": str(smi), "BRAS_VRAM_S": "2"}
    rc, _, err = _bash(tmp_path, corps, {**env, "MIO": "5000"})
    assert rc == 1 and "VRAM non rendue" in err, (rc, err)
    rc, _, err = _bash(tmp_path, corps, {**env, "MIO": "15"})
    assert rc == 0, (rc, err)
