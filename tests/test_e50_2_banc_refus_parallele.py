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
