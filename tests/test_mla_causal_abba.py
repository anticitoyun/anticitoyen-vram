"""zzs : instrument ABBA en service `outils/gpu/mesure/mla-causal-abba.{sh,py}` (poste5, 01/10). Hermétique : faux serveur
HTTP sur un port libre (/v1/models, /metrics, /v1/completions en flux et sans flux), nvidia-smi remplacé, aucune carte.
Teste la PLOMBERIE et les contrôles qui doivent pouvoir rendre « faux » (régime, témoins, NaN, divergence, bandes) — jamais
le cœur MLA lui-même (tests/test_mla_causal_zzs.py)."""
import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

ICI = Path(__file__).resolve().parent.parent
MESURE = ICI / "outils" / "gpu" / "mesure"
_spec = importlib.util.spec_from_file_location("mla_causal_abba", MESURE / "mla-causal-abba.py")
abba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abba)

# Le faux serveur suit l'interrupteur comme le vrai : régime (regime.py:737 et :808) et coût du préfill (B = 0,6 × A).
FAUX = textwrap.dedent('''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port, nom = int(sys.argv[1]), sys.argv[2]
    temoin = os.environ.get("ACVRAM_MLA_CAUSAL", "1") == "0"
    regime = ("[régime] ACVRAM_MLA_CAUSAL=0" if temoin else "[régime] défaut") + " extension=oui torch=faux" \\
        + (" mla_causal=0(temoin)" if temoin else "") + " mla_prep=grille"
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _json(self, d):
            b = json.dumps(d).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._json({"regime_ligne": regime} if self.path == "/metrics" else {"data": [{"id": nom}]})
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            p = req["prompt"]
            time.sleep(len(p) * 4e-6 * (1.0 if temoin else 0.6))
            n, s = req["max_tokens"], sum(p)
            if req.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(b'data: {"choices": [{"text": "", "index": 0}]}\\n\\n')
                self.wfile.write(b"data: [DONE]\\n\\n"); return
            toks = [f"t{(s + i) % 97}" for i in range(n)]
            self._json({"choices": [{"text": "".join(toks), "logprobs": {
                "tokens": toks, "token_logprobs": [-0.01 * i for i in range(n)]}}]})
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bras(temoin, ms8, ms32, jetons=None, lp=None):
    regime = ("[régime] ACVRAM_MLA_CAUSAL=0" if temoin else "[régime] défaut") + " torch=x" \
        + (" mla_causal=0(temoin)" if temoin else "")
    m = {}
    for L, ms in (("8192", ms8), ("32768", ms32)):
        m[L] = [{"k": k, "mur_prefill_ms": ms + k * 0.01, "jetons": list(jetons or ["a", "b", "c"]),
                 "logprobs": list(lp or [-0.1, -0.2, -0.3])} for k in range(3)]
    return {"regime": regime, "mesures": m}


def _comparer(tmp_path, a1, b1, b2, a2, capsys):
    ch = []
    for nom, x in (("A1", a1), ("B1", b1), ("B2", b2), ("A2", a2)):
        p = tmp_path / f"{nom}.json"
        p.write_text(json.dumps(x))
        ch.append(str(p))
    rc = abba.comparer(*ch)
    return rc, capsys.readouterr().out


def _quatre(ms8a=1400.0, ms32a=16000.0, ms8b=1100.0, ms32b=9500.0):
    return _bras(True, ms8a, ms32a), _bras(False, ms8b, ms32b), _bras(False, ms8b, ms32b), _bras(True, ms8a, ms32a)


def test_comparer_identiques_et_dans_les_bandes(tmp_path, capsys):
    rc, out = _comparer(tmp_path, *_quatre(), capsys)
    assert rc == 0, out
    assert "E1(b) : jetons ET logprobs IDENTIQUES" in out
    assert out.count("dans la bande") == 2, out


def test_comparer_faux_si_le_gain_a_32k_est_sous_20_pourcent(tmp_path, capsys):
    rc, out = _comparer(tmp_path, *_quatre(ms32b=14000.0), capsys)
    assert rc == 9 and "FAUX (gain < 20 %)" in out, out


def test_comparer_refuse_un_a_sans_temoin_ou_un_b_avec(tmp_path, capsys):
    a1, b1, b2, a2 = _quatre()
    rc, out = _comparer(tmp_path, a1, b1, b2, _bras(False, 1400.0, 16000.0), capsys)
    assert rc == 5 and "A2 sans" in out, out
    rc, out = _comparer(tmp_path, a1, _bras(True, 1100.0, 9500.0), b2, a2, capsys)
    assert rc == 5 and "bras inversés" in out, out


def test_comparer_refuse_un_regime_qui_differe_par_autre_chose(tmp_path, capsys):
    a1, b1, b2, a2 = _quatre()
    b2["regime"] = b2["regime"].replace("défaut", "ACVRAM_GLUE_COMPACT=0")
    rc, out = _comparer(tmp_path, a1, b1, b2, a2, capsys)
    assert rc == 5 and "autre chose que mla_causal" in out, out


def test_comparer_temoin_non_deterministe_aucun_verdict(tmp_path, capsys):
    a1, b1, b2, a2 = _quatre()
    a2 = copy.deepcopy(a2)
    a2["mesures"]["32768"][1]["jetons"][2] = "z"
    rc, out = _comparer(tmp_path, a1, b1, b2, a2, capsys)
    assert rc == 6 and "mur du préfill" not in out, out


def test_comparer_nan_dans_les_logprobs(tmp_path, capsys):
    a1, b1, b2, a2 = _quatre()
    b1["mesures"]["8192"][0]["logprobs"][1] = float("nan")
    rc, out = _comparer(tmp_path, a1, b1, b2, a2, capsys)
    assert rc == 8 and "NaN" in out, out


def test_comparer_nomme_le_premier_pas_divergent(tmp_path, capsys):
    a1, _, _, a2 = _quatre()
    b = _bras(False, 1100.0, 9500.0, jetons=["a", "x", "c"])
    rc, out = _comparer(tmp_path, a1, b, copy.deepcopy(b), a2, capsys)
    assert "PREMIÈRE DIVERGENCE au pas 1" in out and "E1(b) TOMBE" in out, out


def test_comparer_meme_texte_logprob_different_fait_tomber_e1(tmp_path, capsys):
    """Le serveur ne rend pas les ids : deux ids au même texte ne se voient qu'au logprob."""
    a1, _, _, a2 = _quatre()
    b = _bras(False, 1100.0, 9500.0, lp=[-0.1, -0.25, -0.3])
    rc, out = _comparer(tmp_path, a1, b, copy.deepcopy(b), a2, capsys)
    assert "IDENTIQUES" in out and "E1(b) TOMBE" in out, out


def test_invites_distinctes_et_hors_chauffe():
    ks = [k for L in abba.LONGUEURS for k in abba._cles(L)]
    chauffes = [100 * i for i in range(len(abba.LONGUEURS))]
    assert len(set(ks)) == len(ks) and not set(ks) & set(chauffes)


def test_abba_de_bout_en_bout_contre_le_faux_serveur(tmp_path):
    faux = tmp_path / "faux.py"
    faux.write_text(FAUX)
    smi = tmp_path / "smi"
    smi.write_text("#!/bin/sh\necho 500\n")
    smi.chmod(0o755)
    attendu = subprocess.run(["git", "-C", str(ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {**os.environ, "ATTENDU": attendu, "PY_ACVRAM": sys.executable, "MLA_ABBA_LANCEUR": f"{sys.executable} {faux}",
           "MLA_ABBA_PORT": str(_port()), "MLA_ABBA_REPS": "1", "MLA_ABBA_DELAI_S": "20", "BRAS_NVIDIA_SMI": str(smi),
           "SORTIE": str(tmp_path / "o"), "HOME": str(tmp_path)}
    env.pop("ACVRAM_MLA_CAUSAL", None)
    out, err = tmp_path / "sortie", tmp_path / "erreurs"
    try:
        with open(out, "w") as o, open(err, "w") as e:
            rc = subprocess.run(["bash", str(MESURE / "mla-causal-abba.sh")], stdout=o, stderr=e, env=env,
                                timeout=180).returncode
    finally:
        subprocess.run(["pkill", "-f", str(faux)], capture_output=True)
    sortie = out.read_text()
    assert rc == 0, (rc, sortie[-2000:], err.read_text()[-2000:])
    assert "étape 0 et nsys OMIS" in sortie
    assert "RÉGIMES : A = B + « mla_causal=0(temoin) »" in sortie
    assert "TÉMOINS A1 = A2 et B1 = B2" in sortie and "E1(b) : jetons ET logprobs IDENTIQUES" in sortie
    for bras in ("A1", "B1", "B2", "A2"):
        d = json.loads((tmp_path / "o" / f"{bras}.json").read_text())
        assert (abba.TEMOIN in d["regime"]) == (bras[0] == "A"), d["regime"]
        assert [len(s["jetons"]) for L in d["mesures"] for s in d["mesures"][L]] == [abba.N_GEN, abba.N_GEN]
    # le coût suit l'interrupteur jusque dans le serveur : B (0,6 × A) plus court que A aux deux longueurs
    assert sortie.count("mur du préfill : A") == 2
    assert not (tmp_path / "o" / "A1.json.pid").exists()
