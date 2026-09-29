"""outils/test-menus-reels.py à sec (bd edz) : faux parc (parc.toml, TSV, config kimi), faux lanceurs qui démarrent un
VRAI petit serveur HTTP sur le port du moteur, faux clients kimi/claude. Contrôles : une ligne par alias avec l'étape et
la cause de chaque panne ; serveur arrêté par son PID (plus rien n'écoute) ; reprise sans rejouer ; mode liste sans écrire.
Fautes construites : un lanceur qui échoue, un serveur qui sert un autre id, un client claude qui répond vide, un client kimi qui répond rc 0 sans « Paris » (sortie dégénérée du GLM, 28/09)."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

OUTIL = Path(__file__).resolve().parent.parent / "outils" / "test-menus-reels.py"

SERVEUR = r'''import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, ident = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, d):
        b = json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._j({"data": [{"id": ident}]})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); self._j({"choices": [{"message": {"content": "Paris"}}]})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _port_libre():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _ecoute(port):
    s = socket.socket()
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


@pytest.fixture
def parc(tmp_path):
    port = _port_libre()
    b = tmp_path / "bin"; b.mkdir(); tsv = tmp_path / "TSV"; tsv.mkdir(); kimi = tmp_path / "kimi"; kimi.mkdir()
    (tmp_path / "serveur.py").write_text(SERVEUR)
    # acvram-serveur factice : « acvram-panne » échoue ; « acvram-autre » sert un autre id ; sinon sert l'alias
    (b / "acvram-serveur").write_text(f'''#!/bin/sh
echo "$1 ctx=$2 graphes=$GRAPHES min=$CTX_CLIENT_MIN" >> {tmp_path}/lanceur.log
[ "$1" = acvram-panne ] && {{ echo "RuntimeError: cause au journal serveur" >> "$ACVRAM_SERVEUR_LOG"; echo "OOM simulé au chargement" >&2; exit 1; }}
id="$1"; [ "$1" = acvram-autre ] && id=un-autre-modele
setsid {sys.executable} {tmp_path}/serveur.py {port} "$id" >/dev/null 2>&1 < /dev/null &
for i in $(seq 1 50); do {sys.executable} -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',{port})))" && exit 0; sleep 0.1; done
exit 1
''')
    (b / "kimi-modele").write_text('#!/bin/sh\n[ "$1" = acvram-degenere ] && { printf "|~|\\n|~|\\n|~|\\n"; exit 0; }\necho Paris\n')
    (b / "claude-modele").write_text('#!/bin/sh\n[ "$1" = acvram-muet ] && exit 0\necho Paris\n')
    for f in b.iterdir():
        f.chmod(0o755)
    (kimi / "config.toml").write_text("".join(f'[models.{a}]\nprovider = "acvram"\nmodel = "{a}"\nmax_context_size = 32768\n\n'
                                              for a in ("acvram-bon", "acvram-panne", "acvram-autre", "acvram-muet", "acvram-degenere")))
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\nkimi_dir = "{kimi}"\ntsv_dir = "{tsv}"\nsecrets = "{tmp_path}/secrets.env"\nbin = "{b}"\n\n'
                   f"[moteurs.acvram]\npresent = true\nport = {port}\n")
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(cfg), "TMR_ETAT": str(tmp_path / "etat"),
           "TMR_RESULTATS": str(tsv / "menus-reels.tsv"),
           "ACVRAM_SERVEUR_LOG": str(tmp_path / "serveur.log")}
    yield {"env": env, "port": port, "tsv": tsv / "menus-reels.tsv", "tmp": tmp_path}
    # un test rouge ne laisse pas de faux serveur derrière lui (orphelins du 28/09 : arrêt sauté, port lu à None)
    r = subprocess.run(["ss", "-tlnpH"], capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if f":{port} " in l and "pid=" in l:
            pid = int(l.split("pid=")[1].split(",")[0])
            if "serveur.py" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"):
                os.kill(pid, 9)


def _lancer(parc, *args):
    return subprocess.run([sys.executable, str(OUTIL), *args], env=parc["env"], capture_output=True, text=True, timeout=180)


def _lignes(parc):
    return {l.split("\t")[0]: l.split("\t") for l in parc["tsv"].read_text().splitlines() if l and not l.startswith("#")}


def test_liste_sans_rien_lancer(parc):
    r = _lancer(parc)
    assert r.returncode == 0 and "5 alias au menu" in r.stdout and "rien lancé" in r.stdout, r.stdout + r.stderr
    assert not parc["tsv"].exists() and not _ecoute(parc["port"])


def test_une_ligne_par_alias_cause_de_chaque_panne_et_arret(parc):
    r = _lancer(parc, "--pour-de-vrai", "--delai-client", "30", "--attente", "30")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    l = _lignes(parc)
    assert set(l) == {"acvram-bon", "acvram-panne", "acvram-autre", "acvram-muet", "acvram-degenere"}
    # colonnes : alias date moteur verdict etape cause prechargement_s models completion kimi_rc claude_rc arret detail
    assert l["acvram-bon"][3] == "OK" and l["acvram-bon"][7:11] == ["ok", "ok", "0", "0"] and l["acvram-bon"][11] == "ok"
    assert l["acvram-panne"][3:5] == ["PANNE", "préchargement"] and "OOM simulé" in l["acvram-panne"][5]
    assert "cause au journal serveur" in (parc["tmp"] / "etat" / "journaux" / "acvram-panne.log").read_text()
    assert l["acvram-autre"][3:5] == ["PANNE", "/v1/models"] and "un-autre-modele" in l["acvram-autre"][5]
    assert l["acvram-muet"][3:5] == ["PANNE", "claude"] and "vide" in l["acvram-muet"][5] and l["acvram-muet"][9] == "0"  # kimi joué quand même
    assert l["acvram-degenere"][3:5] == ["PANNE", "kimi"] and "sans « Paris »" in l["acvram-degenere"][5], l["acvram-degenere"]
    assert all(v[11].startswith("ok") for v in l.values()), l                      # chaque serveur arrêté par son PID
    time.sleep(0.3)
    assert not _ecoute(parc["port"]), "un serveur écoute encore"
    r2 = _lancer(parc, "--pour-de-vrai")                                             # reprise : rien à rejouer
    assert "ce passage : 0" in r2.stdout and len(_lignes(parc)) == 5


def test_rejuger_passe_en_panne_un_ok_sans_paris(parc):
    """Lignes OK écrites avant le contrôle : --rejuger relit les journaux ; seule la réponse sans « Paris » tombe."""
    j = parc["tmp"] / "etat" / "journaux"; j.mkdir(parents=True)
    ok = "\t".join(["", "28/09 16:00", "acvram", "OK", "", "", "5", "ok", "ok", "0", "0", "ok", ""])
    parc["tsv"].write_text("# en-tête\n" + "acvram-bon" + ok + "\n" + "acvram-degenere" + ok + "\n")
    (j / "acvram-bon.log").write_text("x\n--- claude-modele rc=0\nParis\n\n--- kimi-modele rc=0\nparis.\n")
    (j / "acvram-degenere.log").write_text("x\n--- claude-modele rc=0\nParis\n\n--- kimi-modele rc=0\n``|\n``|\n")
    r = _lancer(parc, "--rejuger")
    assert r.returncode == 0 and "1 ligne(s)" in r.stdout, r.stdout + r.stderr
    l = _lignes(parc)
    assert l["acvram-bon"][3] == "OK" and l["acvram-degenere"][3:5] == ["PANNE", "kimi"], l
    assert parc["tsv"].with_name("menus-reels.tsv.avant-rejuger").exists() and not _ecoute(parc["port"])


def test_rapide_modele_servi_seul_contexte_court_pannes_rejouees_une_fois(parc):
    """--rapide (chef 29/09) : lanceur à 4096 sans graphes ni CTX_CLIENT_MIN ; preuve = id servi conforme + réponse
    non vide, clients non joués (muet et dégénéré passent, notés « rapide ») ; chaque panne rejouée UNE fois en fin."""
    r = _lancer(parc, "--pour-de-vrai", "--rapide", "--attente", "30")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    brut = [l.split("\t") for l in parc["tsv"].read_text().splitlines() if l and not l.startswith("#")]
    par = {}
    for c in brut:
        par.setdefault(c[0], []).append(c)
    for a in ("acvram-bon", "acvram-muet", "acvram-degenere"):
        assert len(par[a]) == 1 and par[a][0][3] == "OK" and par[a][0][9:11] == ["rapide", "rapide"], par[a]
    for a, etape in (("acvram-panne", "préchargement"), ("acvram-autre", "/v1/models")):
        assert [c[3:5] for c in par[a]] == [["PANNE", etape]] * 2, par[a]              # une fois, puis un rejeu
        assert "rejeu" in par[a][1][12] and "rejeu" not in par[a][0][12]
    lance = (parc["tmp"] / "lanceur.log").read_text().splitlines()
    assert "acvram-bon ctx=4096 graphes=1 min=" in lance, lance
    assert len(lance) == 7                                                              # 5 alias + 2 rejeux
    time.sleep(0.3)
    assert not _ecoute(parc["port"])


def test_liste_dans_son_ordre_meme_deja_testee_et_ctx_client(parc):
    """--liste : alias joués dans l'ordre du fichier, même déjà au TSV ; --ctx-client atteint le lanceur acvram."""
    _lancer(parc, "--pour-de-vrai", "--rapide", "--alias", "acvram-bon", "--attente", "30")
    liste = parc["tmp"] / "liste.txt"
    liste.write_text("# échantillon\nacvram-degenere\tfamille\nacvram-bon\n")
    (parc["tmp"] / "lanceur.log").unlink()
    r = _lancer(parc, "--pour-de-vrai", "--liste", str(liste), "--ctx-client", "34816", "--delai-client", "30",
                "--attente", "30")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lance = (parc["tmp"] / "lanceur.log").read_text().splitlines()
    assert lance == ["acvram-degenere ctx= graphes= min=34816", "acvram-bon ctx= graphes= min=34816"], lance
    brut = [l.split("\t") for l in parc["tsv"].read_text().splitlines() if l and not l.startswith("#")]
    assert [(c[0], c[9]) for c in brut] == [("acvram-bon", "rapide"), ("acvram-degenere", "0"), ("acvram-bon", "0")], brut


def test_service_permanent_jamais_arrete():
    """29/09 07:22 : la passe rapide a tué l'appoint 8081 (moteur « rapide ») en « arrêtant » le serveur de l'alias testé."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tmr", OUTIL)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert not m.a_arreter("rapide") and not m.a_arreter("yals")
    assert m.a_arreter("acvram") and m.a_arreter("llamacpp") and m.a_arreter("vllm")
    source = OUTIL.read_text()
    assert "elif port_moteur is not None and (a_arreter(moteur) or moteur in PORT_MOTEUR):" in source and 'ligne["arret"] = "service permanent laissé"' in source


# --- edz définitif (bd cni, 29/09) : hors champ nommé par la liste, PID par client, YALS/Tabby, doublons reportés ---

def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tmr_cni", OUTIL)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_hors_champ_claude_prechargement_du_repli_kimi_et_pid_par_client(parc):
    """Claude hors champ (modèle limité sous 29 096) : acvram-serveur refuserait le préchargement de claude ; le bras
    précharge comme kimi-modele:167 (34 816), ne joue pas claude, le nomme, et relève le PID autour de kimi."""
    liste = parc["tmp"] / "campagne.tsv"
    liste.write_text("acvram-bon\thors champ : modèle limité à 16384 < 29096\tdans le champ\n")
    r = _lancer(parc, "--pour-de-vrai", "--liste", str(liste), "--delai-client", "30", "--attente", "30")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (parc["tmp"] / "lanceur.log").read_text().splitlines() == ["acvram-bon ctx=34816 graphes= min=34816"]
    c = _lignes(parc)["acvram-bon"]
    assert c[3] == "OK" and c[10] == "hors champ" and c[9] == "0", c
    pid = c[12].split(" ; ")[0].removeprefix("préch ")
    assert pid.isdigit() and f"kimi {pid}→{pid}" in c[12] and "rechargé" not in c[12], c[12]
    assert "claude hors champ : modèle limité à 16384" in c[12], c[12]


def test_yals_tabby_refuses_hors_prise_carte(parc):
    """YALS et TabbyAPI chargent la 5090 sans verrou : la passe refuse (rc 65) hors carte.sh, sans rien écrire."""
    cfg = Path(parc["env"]["ACVRAM_PARC_CONFIG"])
    kimi = cfg.read_text().split('kimi_dir = "')[1].split('"')[0]
    with open(Path(kimi) / "config.toml", "a") as f:
        f.write('[models.yals-x]\nprovider = "yals"\nmodel = "x"\nmax_context_size = 65536\n')
    env = {k: v for k, v in parc["env"].items() if k != "ACVRAM_CARTE_TENUE"}
    r = subprocess.run([sys.executable, str(OUTIL), "--pour-de-vrai", "--moteurs", "yals"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 65 and "REFUS" in r.stderr and not parc["tsv"].exists(), r.stdout + r.stderr


def test_yals_moteur_arrete_apres_le_bras(parc, monkeypatch):
    """Le port du parc de YALS est son proxy : le bras arrête le MOTEUR apparu (PORT_MOTEUR), pas le proxy ; avant
    cni, a_arreter(« yals ») faux laissait YALS sur la carte après la passe."""
    m = _module()
    moteur = _port_libre()
    monkeypatch.setitem(m.PORT_MOTEUR, "yals", moteur)
    b = parc["tmp"] / "bin"
    (b / "kimi-modele").write_text(f'#!/bin/sh\nsetsid {sys.executable} {parc["tmp"]}/serveur.py {moteur} x '
                                   f'>/dev/null 2>&1 < /dev/null &\nsleep 1\necho Paris\n')
    cfg = Path(parc["env"]["ACVRAM_PARC_CONFIG"])
    cfg.write_text(cfg.read_text() + f"\n[moteurs.yals]\npresent = true\nport = {_port_libre()}\n")
    monkeypatch.setattr(m, "ETAT", parc["tmp"] / "etat")
    p = m.charger(str(cfg))
    import argparse
    a = argparse.Namespace(rapide=False, delai_client=30, attente=30, ctx_client=29096, hors_champ={})
    l = m.tester(p, {}, "yals-x", "yals", {"model": "x"}, a)
    assert l["verdict"] == "OK" and l["claude_rc"] == "n/a" and l["arret"] == "ok", l
    assert not _ecoute(moteur)


def test_doublons_reportes_depuis_le_representant(parc):
    """--reporter : la dernière ligne du représentant recopiée sous le nom du doublon, marquée ; représentant absent
    nommé (rc 1), aucune ligne inventée."""
    _lancer(parc, "--pour-de-vrai", "--rapide", "--alias", "acvram-bon", "--attente", "30")
    d = parc["tmp"] / "doublons.tsv"
    d.write_text("# alias\treprésentant\nacvram-bon-bis\tacvram-bon\nacvram-orphelin\tacvram-jamais-teste\n")
    r = _lancer(parc, "--reporter", str(d))
    assert r.returncode == 1 and "acvram-jamais-teste" in r.stdout, r.stdout + r.stderr
    l = _lignes(parc)
    assert l["acvram-bon-bis"][3] == l["acvram-bon"][3] == "OK" and "doublon de acvram-bon" in l["acvram-bon-bis"][12]
    assert "acvram-orphelin" not in l
