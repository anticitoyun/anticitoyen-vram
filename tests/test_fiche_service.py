"""Fiche de service (`outils/fiche-service.py`, Océane, à sec) — juge de
refus, calcul des débits et lancement/attente de serveur, tous exercés
contre un faux serveur HTTP local (aucun modèle, aucun GPU). Le faux serveur
répond selon des marqueurs dans le texte de l'invite : « REFUSER » déclenche
une réponse à motif de refus, sinon une réponse neutre — ce qui rend le juge
et le contrôle instruct/abliterated vérifiables sans jamais charger un poids.
"""
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outils"))
import importlib

fiche_service = importlib.import_module("fiche-service")


# ---------------------------------------------------------------------------
# Faux serveur OpenAI-compatible
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _corps(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": []})
        else:
            self._json(404, {})

    def do_POST(self):
        req = self._corps()
        if self.path == "/v1/chat/completions":
            contenu = req["messages"][-1]["content"]
            texte = ("Je suis desole, mais je ne peux pas vous aider avec ceci."
                     if "REFUSER" in contenu else "Voici la reponse demandee.")
            self._json(200, {
                "choices": [{"message": {"role": "assistant", "content": texte}}],
                "usage": {"prompt_tokens": len(contenu.split()), "completion_tokens": 8}})
        elif self.path == "/v1/completions":
            prompt = req["prompt"]
            n_prompt = len(prompt.split())
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                demandes = req.get("max_tokens", 1)
                # « TRONQUE » simule un EOS précoce (texte remanié, invite
                # normale) ; « Énumère » (fabrique_invite de secours) va
                # jusqu'au bout — reproduit l'asymétrie qui motive le repli
                # sur une seconde invite dans mesure_decode_b1.
                rendus = 1 if "TRONQUE" in prompt else demandes
                for i in range(rendus):
                    morceau = {"choices": [{"text": "x", "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(morceau)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.001)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json(200, {
                    "choices": [{"text": "x" * 8, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": n_prompt, "completion_tokens": 1}})
        else:
            self._json(404, {})

    def _json(self, code: int, corps: dict) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(corps).encode())


@pytest.fixture(scope="module")
def serveur():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    fil = threading.Thread(target=httpd.serve_forever, daemon=True)
    fil.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


# ---------------------------------------------------------------------------
# Juge de refus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reponse, attendu", [
    ("Je suis désolé, mais je ne peux pas vous aider avec ceci.", True),
    ("I'm sorry, but I cannot assist with that request.", True),
    ("JE SUIS DESOLE, MAIS JE NE PEUX PAS.", True),  # sans accents, majuscules
    ("Voici la réponse : la capitale de la France est Paris.", False),
    ("", False),
])
def test_juge_motifs(reponse, attendu):
    assert fiche_service.juge_motifs(reponse) is attendu


def test_echantillon_a_relire_taille_et_determinisme():
    reponses = [f"r{i}" for i in range(12)]
    a = fiche_service.echantillon_a_relire(reponses, graine=42)
    b = fiche_service.echantillon_a_relire(reponses, graine=42)
    assert a == b                       # déterministe pour une même graine
    assert len(a) == 2                   # ceil(12 * 0.10) = 2
    assert all(0 <= i < 12 for i in a)


def test_echantillon_a_relire_liste_courte():
    assert fiche_service.echantillon_a_relire(["seul"], graine=1) == [0]


# ---------------------------------------------------------------------------
# Corpus de refus — jeu fixe, jamais fabriqué par l'outil
# ---------------------------------------------------------------------------

def test_charger_corpus_refus_absent(tmp_path):
    with pytest.raises(SystemExit, match="absent"):
        fiche_service.charger_corpus_refus(tmp_path / "n-existe-pas.txt")


def test_charger_corpus_refus_mauvais_compte(tmp_path):
    p = tmp_path / "refus-12.txt"
    p.write_text("une seule invite\n")
    with pytest.raises(SystemExit, match="12 attendues"):
        fiche_service.charger_corpus_refus(p)


def test_charger_corpus_refus_correct(tmp_path):
    import hashlib
    p = tmp_path / "refus-12.txt"
    lignes = [f"invite {i}" for i in range(12)]
    p.write_text("\n".join(lignes) + "\n")
    invites, sha = fiche_service.charger_corpus_refus(p)
    assert invites == lignes
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Débits, contre le faux serveur
# ---------------------------------------------------------------------------

def test_mesure_prefill(serveur):
    with httpx.Client() as client:
        r = fiche_service.mesure_prefill(client, serveur, "m", jetons_vises=32,
                                         repetitions=2)
    assert "refus" not in r
    assert r["prefill_tok_s"] > 0
    assert len(r["prefill_jetons_reels"]) == 2


def test_mesure_decode_b1(serveur):
    with httpx.Client() as client:
        r = fiche_service.mesure_decode_b1(client, serveur, "m", n_tokens=16,
                                           prompt_len=8, chauffe=0, repetitions=2)
    assert "refus" not in r
    assert r["decode_tok_s"] > 0
    assert r["decode_jetons_produits"] == 16
    assert r["decode_invite"] == "texte remanié"


def test_mesure_decode_b1_retombe_sur_l_enumeration_si_eos_precoce(serveur, monkeypatch):
    """Sage, 17/09 (asymétrie acvram/GGUF) : une invite normale qui s'arrête
    avant `n_tokens // 2` (le faux serveur simule l'EOS via « TRONQUE ») doit
    faire essayer l'énumération avant de renvoyer un refus."""
    monkeypatch.setattr(fiche_service, "_invite_texte",
                        lambda graine, n: f"[TRONQUE {graine}]")
    with httpx.Client() as client:
        r = fiche_service.mesure_decode_b1(client, serveur, "m", n_tokens=16,
                                           prompt_len=8, chauffe=0, repetitions=2)
    assert "refus" not in r, r
    assert r["decode_invite"] == "énumération"
    assert r["decode_jetons_produits"] == 16


def test_mesure_decode_b1_refuse_si_les_deux_invites_tronquent(serveur, monkeypatch):
    monkeypatch.setattr(fiche_service, "_invite_texte",
                        lambda graine, n: f"[TRONQUE {graine}]")
    monkeypatch.setattr(fiche_service, "_invite_enumeration",
                        lambda k: f"[TRONQUE enum {k}]")
    with httpx.Client() as client:
        r = fiche_service.mesure_decode_b1(client, serveur, "m", n_tokens=16,
                                           prompt_len=8, chauffe=0, repetitions=2)
    assert "refus" in r
    assert "deux invites essayées" in r["refus"]


def test_mesure_refus(serveur):
    invites = ["REFUSER ceci"] * 5 + ["question normale"] * 7
    with httpx.Client() as client:
        r = fiche_service.mesure_refus(client, serveur, "m", invites,
                                       max_tokens=16, graine_relecture=7)
    assert r["n_refus"] == 5
    assert r["n_total"] == 12
    assert len(r["lecture_manuelle"]) == 2  # ceil(12*0.10)


# ---------------------------------------------------------------------------
# Contrôle du juge (Sage § 2) — écart ≥ 4/12
# ---------------------------------------------------------------------------

def test_valider_juge_ecart_suffisant(serveur):
    invites = ["REFUSER un peu"] * 6 + ["question normale"] * 6
    # même serveur pour les deux "modèles" ; on distingue par les invites :
    # le "abliterated" ne reçoit que les invites neutres.
    resultat = fiche_service.valider_juge(
        serveur, serveur, "instruct", "abliterated", invites, "shaXYZ")
    assert resultat["instruct"]["n_refus"] == 6
    assert resultat["abliterated"]["n_refus"] == 6
    # même corpus envoyé aux deux -> écart nul ici ; le test ci-dessous couvre
    # le cas où le juge doit refuser de publier.
    assert resultat["ok"] is (abs(6 - 6) >= fiche_service.ECART_MIN_JUGE)
    assert resultat["ok"] is False
    assert resultat["ecart"] == 0


def test_charger_validation_juge_absente(tmp_path):
    assert fiche_service.charger_validation_juge(tmp_path / "x.json", "sha") is None


def test_charger_validation_juge_corpus_change(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"ok": True, "prompts_sha256": "ancien"}))
    assert fiche_service.charger_validation_juge(p, "nouveau") is None


def test_charger_validation_juge_valide(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"ok": True, "prompts_sha256": "sha", "ecart": 6}))
    v = fiche_service.charger_validation_juge(p, "sha")
    assert v["ok"] is True and v["ecart"] == 6


# ---------------------------------------------------------------------------
# Serveur : attente et échec
# ---------------------------------------------------------------------------

def test_lancer_serveur_deja_debout(serveur, tmp_path):
    s = fiche_service.lancer_serveur(None, serveur, 0, tmp_path / "j.log", 5.0)
    assert s.charge is True
    assert s.processus is None


def test_lancer_serveur_commande_shell_avec_cd(tmp_path):
    """Les commandes réelles du plan (`cd /opt/ia/X && binaire ...`) sont des
    lignes shell, pas un exécutable+arguments — `shlex.split` les cassait
    (`["cd", "/opt/ia/X", "&&", ...]`, Popen cherchait un exécutable « cd »).
    Reproduit ici avec un petit serveur : le `cd` doit réellement s'appliquer
    (le script est résolu relativement au nouveau répertoire, pas au cwd du
    test), et l'arrêt ne doit laisser aucun processus vivant (groupe tué)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    dossier = tmp_path / "sous-dossier"
    dossier.mkdir()
    (dossier / "faux_serveur.py").write_text(
        "import http.server, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self): self.send_response(200); self.end_headers()\n"
        "    def log_message(self, *a): pass\n"
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")

    commande = f"cd {dossier} && {sys.executable} faux_serveur.py {{port}}"
    srv = fiche_service.lancer_serveur(commande, f"http://127.0.0.1:{port}", port,
                                       tmp_path / "j.log", 10.0)
    assert srv.charge is True, srv.cause_echec
    pid_shell = srv.processus.pid
    fiche_service.arreter_serveur(srv)
    # Le SIGTERM au groupe part avant que `wait()` ne rende (qui n'attend que
    # le shell) : le reste du groupe peut survivre une fraction de seconde,
    # pas indéfiniment — d'où un court sondage plutôt qu'une assertion
    # instantanée, sans quoi le test confondrait « pas encore mort » et
    # « jamais tué » (exactement ce qu'un `terminate()` sur le seul shell
    # laisserait faire, sans jamais converger).
    for _ in range(20):
        try:
            os.killpg(pid_shell, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("le groupe de processus a survécu à arreter_serveur")


def test_lancer_serveur_timeout_tue_aussi_le_groupe(tmp_path):
    """Même piège que ci-dessus, mais sur le REPLI DE `lancer_serveur` quand
    `/v1/models` ne répond jamais (trouvé le 17/09 en validant le juge de
    refus : un serveur EXL3/TabbyAPI qui écoutait sur le mauvais port a
    survécu au timeout, occupant la carte pour la mesure suivante)."""
    dossier = tmp_path / "sous-dossier"
    dossier.mkdir()
    (dossier / "ne_repond_jamais.py").write_text(
        "import time\n"
        "open('marqueur', 'w').close()\n"
        "time.sleep(60)\n")
    commande = f"cd {dossier} && {sys.executable} ne_repond_jamais.py"
    srv = fiche_service.lancer_serveur(commande, "http://127.0.0.1:1", 0,
                                       tmp_path / "j.log", 1.5)
    assert srv.charge is False
    assert "répond pas" in srv.cause_echec
    pid_shell = srv.processus.pid
    for _ in range(20):
        try:
            os.killpg(pid_shell, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("le repli sur timeout a laissé le groupe vivant")


def test_lancer_serveur_timeout(tmp_path):
    s = fiche_service.lancer_serveur(None, "http://127.0.0.1:1", 0,
                                     tmp_path / "j.log", 1.0)
    assert s.charge is False
    assert "répond pas" in s.cause_echec


def test_vram_pid_sans_exception():
    assert fiche_service.vram_pid(999999999) is None


def test_descendants_traverse_un_vrai_arbre_de_processus():
    """`shell=True` fait de `serveur.processus.pid` un `sh -c`, jamais le
    binaire qui tient la VRAM — `_descendants` doit trouver son (petit-)
    enfant réel, pas juste lui-même."""
    proc = subprocess.Popen(
        ["sh", "-c", f"{sys.executable} -c \"import time; time.sleep(5)\""])
    try:
        temps_limite = time.perf_counter() + 3
        enfant = None
        while time.perf_counter() < temps_limite and enfant is None:
            arbre = fiche_service._descendants(proc.pid)
            autres = arbre - {proc.pid}
            if autres:
                enfant = next(iter(autres))
            else:
                time.sleep(0.05)
        assert enfant is not None, "le processus python enfant n'est jamais apparu"
        assert proc.pid in fiche_service._descendants(proc.pid)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_vram_pid_somme_sur_les_descendants_pas_seulement_le_pid_donne(monkeypatch):
    """Régression trouvée par Laure en campagne (bloc 0, `vram_chargement_
    octets` toujours None depuis le passage à `shell=True`) : nvidia-smi ne
    voit que le PID du binaire réel (l'enfant), jamais celui du `sh -c`
    qu'on lui passe — sommer sur les descendants doit récupérer ce chiffre."""
    proc = subprocess.Popen(
        ["sh", "-c", f"{sys.executable} -c \"import time; time.sleep(5)\""])
    try:
        temps_limite = time.perf_counter() + 3
        arbre = {proc.pid}
        while time.perf_counter() < temps_limite and arbre == {proc.pid}:
            arbre = fiche_service._descendants(proc.pid)
        enfant = next(iter(arbre - {proc.pid}))

        def faux_run(*a, **kw):
            class R:
                stdout = f"{enfant}, 1234\n999999, 9999\n"
            return R()
        monkeypatch.setattr(fiche_service.subprocess, "run", faux_run)
        assert fiche_service.vram_pid(proc.pid) == 1234 * 1024 * 1024
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Fiche complète, TSV
# ---------------------------------------------------------------------------

def test_construire_fiche_sans_validation_juge(serveur, tmp_path, monkeypatch):
    monkeypatch.setattr(fiche_service, "VALIDATION_JUGE", tmp_path / "absente.json")
    corpus = tmp_path / "refus-12.txt"
    corpus.write_text("\n".join(f"invite {i}" for i in range(12)))
    import argparse
    args = argparse.Namespace(
        nom="test-entree", moteur="acvram", modele="m", commande=None,
        base_url=serveur, port=0, journal=str(tmp_path / "j.log"),
        timeout_chargement=5.0, arreter_serveur=False,
        prefill_jetons=32, prefill_repetitions=2, decode_jetons=16,
        decode_prompt_len=8, decode_chauffe=0, decode_repetitions=2,
        refus_corpus=str(corpus), refus_jetons=16,
        sortie=str(tmp_path / "fiche-service.tsv"))
    fiche = fiche_service.construire_fiche(args)
    assert fiche["charge"] is True
    assert fiche["refus"]["publie"] is False
    assert "non validé" in fiche["refus"]["cause"]

    chemin = Path(args.sortie)
    fiche_service.ecrire_tsv(fiche, chemin)
    fiche_service.ecrire_tsv(fiche, chemin)
    lignes = chemin.read_text().splitlines()
    assert lignes[0].startswith("nom\t")
    assert len(lignes) == 3  # en-tête + deux lignes, jamais un second en-tête
