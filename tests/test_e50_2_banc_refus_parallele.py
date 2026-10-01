"""Pièce e50.2 (poste2, ordre chef, 01/10) : le rejeu ×4 de banc-refus
(`outils/campagne-e50.2-nocturne.py:_lancer_banc_refus_parallele`) coupe la source du
banc-refus DÉPLOYÉ (hors dépôt) avant son corps CLI pour réutiliser ses constantes — chef
(REGLES § 9) demande (a) un test cassant si cette coupe devient invalide, (b) une équivalence
stricte contre le séquentiel d'origine sur un faux serveur à réponses fixes.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent.parent
PY = "/usr/bin/python3"
BANC_REFUS = Path.home() / ".local" / "bin" / "banc-refus"
CAMPAGNE = ICI / "outils" / "campagne-e50.2-nocturne.py"

# même regex que fenetre.py:1206 (et campagne-e50.2-nocturne.py:_parser_refus)
_PARSER_REFUS = re.compile(r"\t(\d+)/(\d+) refus\t(\S.*)$", re.MULTILINE)


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
                                    "from gi.repository import Adw"], capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not BANC_REFUS.exists() or not _gi_ok(),
                                 reason="banc-refus absent ou GTK4/libadwaita absent (/usr/bin/python3)")


def test_parser_outils_insensible_aux_couleurs_ansi():
    """Trouvé en mesure réelle (01/10, campagne.log 10:35-10:37) : banc-outils colore sa
    sortie SANS tester isatty (~/.local/bin/banc-outils:163,183-190) — avant le correctif,
    `_parser_outils` ne matchait jamais sur une sortie de terminal capturée
    (`\\x1b[1m4/8\\x1b[0m` au lieu de `4/8`), et le tok/s de TOUS les alias acvram restait
    silencieusement à son ancienne valeur. `_lancer_banc` doit nettoyer les codes ANSI
    avant de rendre la sortie à `_parser_outils`."""
    code = (
        f"import sys; sys.path.insert(0, {str(ICI / 'outils')!r}); "
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('c', {str(CAMPAGNE)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "brut = '  \\u2192 \\x1b[1m4/8\\x1b[0m appels corrects \\u00b7 132 tok/s\\n'; "
        "assert m._parser_outils(brut) is None, 'la sortie brute matche déjà — revoir ce test'; "
        "nettoye = m._ANSI.sub('', brut); "
        "r = m._parser_outils(nettoye); "
        "print(r.groups() if r else None)"
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "('4', '8', '132')", r.stdout


def test_arreter_attend_la_mort_et_tue_si_sigterm_ignore(tmp_path):
    """Trouvé par chef (01/10 14h15) : `_arreter` envoyait SIGTERM puis `sleep(2)` fixe et
    repartait — un serveur qui ignore SIGTERM (ou met plus de 2 s à mourir) restait vivant,
    tenant la VRAM, pendant que la campagne annonçait « carte rendue ». Ce test lance un VRAI
    processus qui ignore SIGTERM : avec l'ancien `sleep(2)` seul il resterait vivant après
    l'appel (ce test casserait si on revient à cette forme) ; avec `_attendre_mort_ou_tuer`,
    il doit être mort (SIGKILL) avant le retour."""
    script = tmp_path / "essai_sigkill.py"
    script.write_text(
        f"import sys; sys.path.insert(0, {str(ICI / 'outils')!r})\n"
        "import importlib.util, os, signal, subprocess, time\n"
        f"spec = importlib.util.spec_from_file_location('c', {str(CAMPAGNE)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "p = subprocess.Popen(['python3', '-c',\n"
        "    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        "time.sleep(0.3)\n"  # laisse le temps au handler SIG_IGN de se poser
        "os.kill(p.pid, signal.SIGTERM)\n"
        "tue = m._attendre_mort_ou_tuer(p.pid, 2, None, 'test')\n"
        "print('tue=', tue)\n"
        # zombie (SIGKILL pas encore réap é par le kernel) compte comme mort — os.kill(pid, 0)
        # réussirait encore sur un zombie, _sortant lit le vrai état (comme carte_rendue.py)
        "print('etat=vivant' if not m._sortant(p.pid) else 'etat=mort')\n"
    )
    r = subprocess.run([PY, str(script)], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    assert "tue= True" in r.stdout, f"SIGKILL non déclenché : {r.stdout!r}"
    assert "etat=mort" in r.stdout, f"processus encore vivant après _attendre_mort_ou_tuer : {r.stdout!r}"


def test_coupure_marqueur_present_cassant():
    """(a) Si le banc-refus déployé change et que le marqueur de coupe disparaît ou bouge,
    ce test casse AVANT que la campagne ne tourne en aveugle dessus cette nuit."""
    source = BANC_REFUS.read_text()
    assert "\na = argparse.ArgumentParser()" in source, (
        "marqueur de coupe absent du banc-refus déployé — "
        "_module_banc_refus() dans campagne-e50.2-nocturne.py doit être mis à jour")
    coupure = source.index("\na = argparse.ArgumentParser()")
    ns = {}
    exec(compile(source[:coupure], str(BANC_REFUS), "exec"), ns)
    assert isinstance(ns.get("INVITES"), list) and len(ns["INVITES"]) == 5
    assert all(isinstance(x, str) and x for x in ns["INVITES"])
    assert hasattr(ns.get("REFUS"), "search")
    assert ns["REFUS"].search("Je ne peux pas vous aider avec cette demande.")
    assert not ns["REFUS"].search("Il faisait nuit noire sur la lande lorsque le chevalier s'avança.")
    assert callable(ns.get("cle"))


class _FauxServeur(BaseHTTPRequestHandler):
    """Classe par contenu d'invite (parité du hash) : refus si pair, réponse normale sinon —
    déterministe quel que soit l'ordre d'arrivée (séquentiel ou parallèle)."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "faux-modele-test"}]})
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        corps = json.loads(self.rfile.read(n))
        contenu = corps["messages"][0]["content"]
        pair = int(hashlib.md5(contenu.encode()).hexdigest(), 16) % 2 == 0
        if pair:
            texte = "Je ne peux pas vous aider avec cette demande, désolé."
            fin = "stop"
        else:
            texte = ("Il faisait nuit noire sur la lande lorsque le chevalier s'avança, "
                     "l'épée au clair, le cœur battant la chamade sous son armure rouillée.")
            fin = "stop"
        self._json(200, {"choices": [{"message": {"content": texte}, "finish_reason": fin}]})

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def faux_serveur():
    srv = HTTPServer(("127.0.0.1", 0), _FauxServeur)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()
    t.join(timeout=5)


def test_equivalence_sequentiel_contre_parallele(faux_serveur, tmp_path):
    """(b) Même serveur à réponses fixes : la sortie de banc-refus d'origine (séquentiel,
    --tours 1) et celle du rejeu ×4 doivent donner le MÊME compte refus/essais et la MÊME
    ligne lue par le parseur de la GUI (mot qualitatif compris)."""
    port = faux_serveur

    r_seq = subprocess.run([PY, str(BANC_REFUS), "--port", str(port), "--tours", "1"],
                            capture_output=True, text=True, timeout=60)
    assert r_seq.returncode == 0, r_seq.stderr
    m_seq = _PARSER_REFUS.search(r_seq.stdout)
    assert m_seq, f"sortie séquentielle non reconnue : {r_seq.stdout!r}"

    journal = tmp_path / "journal.log"
    code = (
        f"import sys; sys.path.insert(0, {str(ICI / 'outils')!r}); "
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('c', {str(CAMPAGNE)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"rc, sortie = m._lancer_banc_refus_parallele({port}, 4, {str(journal)!r}); "
        "print(sortie, end='')"
    )
    r_par = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60)
    assert r_par.returncode == 0, r_par.stderr
    m_par = _PARSER_REFUS.search(r_par.stdout)
    assert m_par, f"sortie parallèle non reconnue : {r_par.stdout!r}"

    # groupes du parseur GUI (fenetre.py:1206) : (refus, essais, mot)
    assert m_seq.groups() == m_par.groups(), (
        f"séquentiel {m_seq.groups()} != parallèle {m_par.groups()} — "
        "le rejeu ×4 doit produire EXACTEMENT le même verdict que l'original")
