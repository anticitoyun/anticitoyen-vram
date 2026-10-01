#!/usr/bin/env python3
"""e50.3 § 2 bis / § 7 (poste3, 01/10) — pass@1 de humaneval_e50, calculé À PART de lm-eval (voir
le commentaire de `outils/lm_eval_taches/humaneval_e50.yaml`) : lit les échantillons journalisés
(`--log_samples`), reconstruit <prompt><complétion>\n<test HumanEval>\ncheck(<entry_point>) pour
chaque item, fait passer le programme complet dans `outils/bac-a-sable-humaneval.sh` (jamais nu),
et rend la fraction qui passe (rc 0).

Usage : qualite-e50-humaneval-score.py <samples_humaneval_e50_*.jsonl>
Imprime une seule ligne JSON : {"n": N, "pass": P, "pass_at_1": P/N}.
"""
import json
import os
import subprocess
import sys
import tempfile

ICI = os.path.dirname(os.path.abspath(__file__))
BAC_A_SABLE = os.path.join(ICI, "bac-a-sable-humaneval.sh")


def _programme(doc, completion):
    """prompt + complétion générée + test HumanEval + l'appel `check(entry_point)` — le
    format standard openai_humaneval (`test` est un `def check(candidate): ...`)."""
    entry_point = doc.get("entry_point", "")
    return f"{doc['prompt']}{completion}\n\n{doc['test']}\n\ncheck({entry_point})\n"


def noter(chemin_jsonl):
    n = 0
    reussis = 0
    with open(chemin_jsonl, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            item = json.loads(ligne)
            doc = item.get("doc", item)
            completion = (item.get("filtered_resps") or item.get("resps") or [[""]])[0]
            if isinstance(completion, list):
                completion = completion[0]
            n += 1
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f_py:
                f_py.write(_programme(doc, completion))
                chemin_py = f_py.name
            try:
                r = subprocess.run(["bash", BAC_A_SABLE, chemin_py],
                                   capture_output=True, text=True, timeout=20)
                if r.returncode == 0:
                    reussis += 1
            finally:
                os.unlink(chemin_py)
    return {"n": n, "pass": reussis, "pass_at_1": (reussis / n) if n else 0.0}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: qualite-e50-humaneval-score.py <samples_humaneval_e50_*.jsonl>", file=sys.stderr)
        raise SystemExit(64)
    print(json.dumps(noter(sys.argv[1])))
