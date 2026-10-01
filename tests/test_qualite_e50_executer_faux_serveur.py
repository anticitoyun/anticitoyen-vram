"""e50.3 § 7 (poste4, 01/10) : `outils/qualite-e50.sh --executer` de bout en bout, contre un
FAUX serveur HTTP (`/v1/models`, `/v1/chat/completions`) et un FAUX `lm_eval` (module Python
injecté par PYTHONPATH, répond au même contrat CLI que `python -m lm_eval run`, écrit un
`results_*.json`/`samples_*.jsonl` de la même forme que le vrai paquet) — aucune carte, aucun
GPU, pas de `.venv-panel`. Teste la PLOMBERIE (lancement/arrêt du harnais `serveur-bras.sh`,
agrégation MMLU/GSM8K, score HumanEval via `qualite-e50-humaneval-score.py` sous bac à sable,
S composite, barème, écriture TSV + `ecrire_note`) — jamais la qualité d'un vrai modèle, jamais
le comportement réel du paquet lm-eval (REGLES §6 : rien de ceci ne lit la sortie d'un modèle
non censuré, les complétions du faux serveur sont écrites par ce test).

Entièrement isolé du parc réel : `ACVRAM_PARC_CONFIG` pointe un `parc.toml` sous `tmp_path`,
jamais `~/TSV` ni `~/.kimi-code/config.toml`."""
import json
import os
import socket
import subprocess
import textwrap
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent.parent
SCRIPT = ICI / "outils" / "qualite-e50.sh"
ALIAS = "acvram-faux-e50-test-nvfp4"


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


FAUX_SERVEUR = textwrap.dedent('''
    import json, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port, nom = int(sys.argv[1]), sys.argv[2]
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"data": [{"id": nom}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
            # complétion fixe : satisfait le filtre MMLU ("answer ... A") ET contient un nombre
            # pour GSM8K (flexible-extract) ; HumanEval est scoré à part (jamais via ce texte).
            texte = "Reasoning...\\nThe answer is (A).\\n#### 42"
            rep = {"id": "faux", "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": texte}, "finish_reason": "stop"}]}
            b = json.dumps(rep).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')

# Faux paquet `lm_eval` : répond au contrat CLI `python -m lm_eval run --tasks ... --output_path
# ... --model_args ...` — écrit un results_*.json (MMLU+GSM8K) ou un samples_humaneval_e50_*.jsonl
# (HumanEval), dans la forme que `qualite-e50.sh`/`qualite-e50-humaneval-score.py` attendent déjà
# (mêmes clés que `outils/panel-taches-resume.py` sait lire : "<métrique>,<filtre>").
FAUX_LM_EVAL_MAIN = textwrap.dedent('''
    import json, os, sys, time, urllib.request

    def _arg(nom, args):
        for i, a in enumerate(args):
            if a == nom:
                return args[i + 1]
            if a.startswith(nom + "="):
                return a.split("=", 1)[1]
        return None

    def run(args):
        tasks = (_arg("--tasks", args) or "").split(",")
        sortie = _arg("--output_path", args)
        model_args = _arg("--model_args", args) or ""
        base_url = next((kv.split("=", 1)[1] for kv in model_args.split(",") if kv.startswith("base_url=")), None)
        assert base_url, "model_args sans base_url"
        # preuve que le faux serveur est bien joignable (pas seulement configuré) :
        with urllib.request.urlopen(base_url.rsplit("/v1/", 1)[0] + "/v1/models", timeout=5) as r:
            assert json.loads(r.read())["data"][0]["id"]
        os.makedirs(sortie, exist_ok=True)
        horodate = str(int(time.time() * 1000))
        if "humaneval_e50" in tasks:
            # pass@1 est calculé À PART (§2 bis) : seule la forme samples_*.jsonl importe ici ;
            # une complétion triviale, valide syntaxiquement mais fausse (pass_at_1 attendu = 0).
            chemin = os.path.join(sortie, f"samples_humaneval_e50_{horodate}.jsonl")
            doc = {"prompt": "def f(x):\\n    \\"\\"\\"docstring\\"\\"\\"\\n", "entry_point": "f",
                   "test": "def check(candidate):\\n    assert candidate(1) == 2\\n"}
            with open(chemin, "w") as fh:
                fh.write(json.dumps({"doc": doc, "filtered_resps": ["    return x\\n"]}) + "\\n")
        else:
            chemin = os.path.join(sortie, f"results_{horodate}.json")
            resultats = {
                "mmlu_e50_hsm": {"exact_match,get-answer-v275": 0.6},
                "mmlu_e50_law": {"exact_match,get-answer-v275": 0.5},
                "mmlu_e50_ccs": {"exact_match,get-answer-v275": 0.7},
                "gsm8k_e50": {"exact_match,strict-match": 0.1, "exact_match,flexible-extract": 0.4},
            }
            with open(chemin, "w") as fh:
                json.dump({"results": resultats, "configs": {}}, fh)
        print(f"faux lm_eval : écrit {chemin}")
        return 0

    def main():
        args = sys.argv[1:]
        assert args and args[0] == "run", f"sous-commande inattendue : {args}"
        raise SystemExit(run(args[1:]))
''')


def _ecrire_faux_lm_eval(racine):
    pkg = racine / "lm_eval"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAUX_LM_EVAL_MAIN + "\nif __name__ == '__main__':\n    main()\n")
    return racine


def _parc_isole(tmp_path):
    """Un parc minimal, isolé (ACVRAM_PARC_CONFIG), avec un seul alias acvram résolvable."""
    tsv_dir = tmp_path / "TSV"
    tsv_dir.mkdir()
    kimi_dir = tmp_path / "kimi"
    kimi_dir.mkdir()
    dossier_modele = tmp_path / "modele-faux"
    dossier_modele.mkdir()

    (tsv_dir / "acvram-chemins.tsv").write_text(f"{ALIAS}\t{dossier_modele}\t4096\n")
    (tsv_dir / "gguf-chemins.tsv").write_text("")
    (tsv_dir / "vllm-chemins.tsv").write_text("")
    (tsv_dir / "vision-modeles.tsv").write_text("")
    (tsv_dir / "notes-modeles.tsv").write_text(f"{ALIAS}\tnon\t?\tnon mesuré\tchat\n")
    (kimi_dir / "config.toml").write_text(
        f'[models."{ALIAS}"]\n'
        f'provider = "acvram"\n'
        f'model = "FauxModele"\n'
        f'max_context_size = 4096\n'
        f'capabilities = []\n'
    )
    parc_toml = tmp_path / "parc.toml"
    parc_toml.write_text(
        f'[chemins]\n'
        f'kimi_dir = "{kimi_dir}"\n'
        f'tsv_dir = "{tsv_dir}"\n'
    )
    return parc_toml, tsv_dir


def _gi_ok():
    r = subprocess.run(["/usr/bin/python3", "-c",
                       "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _gi_ok(), reason="GTK4/libadwaita absent (parc.py en dépend même pour ecrire_note)")


def test_executer_bout_en_bout_faux_serveur_faux_lmeval(tmp_path):
    parc_toml, tsv_dir = _parc_isole(tmp_path)
    faux_racine = _ecrire_faux_lm_eval(tmp_path / "faux-lm-eval-lib")
    faux_serveur_py = tmp_path / "faux-serveur.py"
    faux_serveur_py.write_text(FAUX_SERVEUR)
    port = _port()

    env = {
        **os.environ,
        "ACVRAM_PARC_CONFIG": str(parc_toml),
        "ACVRAM_LMEVAL_PY": "/usr/bin/python3",
        "PYTHONPATH": str(faux_racine) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "ACVRAM_E50_LANCEUR": f"/usr/bin/python3 {faux_serveur_py}",
        "ACVRAM_E50_PORT": str(port),
        "ACVRAM_E50_SERVED_NAME": ALIAS,
        "ACVRAM_E50_DELAI_S": "20",
        "ACVRAM_E50_TSV": str(tmp_path / "qualite-e50.tsv"),
        "ACVRAM_E50_SCRATCH": str(tmp_path / "scratch"),
    }

    r = subprocess.run(["bash", str(SCRIPT), ALIAS, "--executer", "--je-sais-que-la-carte-est-libre"],
                       env=env, capture_output=True, text=True, timeout=60, cwd=str(ICI))
    assert r.returncode == 0, "stdout:\n" + r.stdout + "\nstderr:\n" + r.stderr

    assert "faux lm_eval : écrit" in r.stdout
    assert "->" in r.stdout  # ligne "S=... -> ★..."

    tsv_e50 = tmp_path / "qualite-e50.tsv"
    assert tsv_e50.exists(), "ACVRAM_E50_TSV jamais écrit"
    derniere = [l for l in tsv_e50.read_text().splitlines() if l.startswith(ALIAS + "\t")][-1]
    champs = derniere.split("\t")
    assert champs[0] == ALIAS
    s = float(champs[4])
    # S attendu = (acc_mmlu moyenne(0.6,0.5,0.7)=0.6 + gsm8k flexible 0.4 + pass_he 0/1) / 3
    assert abs(s - (0.6 + 0.4 + 0.0) / 3) < 1e-3, derniere  # TSV tronqué à 4 décimales

    note = (tsv_dir / "notes-modeles.tsv").read_text()
    assert ALIAS in note and "non mesuré" not in note.split(ALIAS, 1)[1].split("\n", 1)[0]
