"""Pièce 37 : pourquoi `acvram eval` rend PPL 27 713-59 316 sur Gemma 4 31B
alors que le service décode juste (scellé E, KL 0,87 sur 280 jetons) ?
Test discriminant (verdict croisé duck.ai) : teacher-forcing court, SANS
fenêtrage, NLL jeton par jeton, et la première position qui diverge.

Trois bras sur les MÊMES ids ([BOS] + texte connu, 300 jetons par défaut) :
  eval   : le chemin de `perplexity` — ForwardBatch d évaluation,
           `model(batch, return_hidden=True)` puis `_pertes_par_tranches`
  serve  : le chemin du serveur au préfill — `Engine.add_request` + un
           `step()` (mêmes noyaux que /v1/completions), logits de la dernière
           position seulement, et `forward(logits_positions=arange)` pour
           toutes (même forward que le service, logits à chaque position)
  hf     : transformers bf16 (offload, `HF_PYTHON` + `MAXMEM` comme
           decode-pas), NLL par position — la référence
Sortie : NLL médiane par bras, |Δ| eval−serve et eval−hf par position,
première position où |Δ| > 1 nat ; PPL(ctx ≥ 32) de chaque bras.

Prédictions (écrites avant) :
  P1 eval ≈ serve ≈ hf (|Δ| < 0,1 nat, PPL 8-20) → le forward est sain : la
     pathologie vient du fenêtrage/corpus de `perplexity` (d : fenêtre
     locale 1 024 au-delà de 1 024 jetons, ou corpus) → rejouer avec
     --jetons 1500 : divergence attendue APRÈS la position 1 024.
  P2 eval ≠ serve dès la position 1 → la ForwardBatch d évaluation manque
     un champ que le service pose (à lire dans l écart : seq_ids, images,
     positions) ; c est un défaut d eval, corrigeable à sec.
  P3 eval ≈ serve ≠ hf dès le début → le forward acvram diverge sur ce
     modèle hors gabarit de conversation (le scellé E ne l a vu que sous
     gabarit) — défaut moteur, à nommer par position et par couche.
Contrôle qui rend « faux » : PPL(ctx ≥ 32) du bras eval ≤ 30 sur ce texte
connu ; > 30 = eval faux, quel que soit le reste.

Usage : outils/carte.sh python outils/gpu/mesure/diag-eval-nll.py ALIAS [--source HF_DIR]
        [--jetons 300] [--texte FICHIER] [--json SORTIE]         (≤ 10 min avec hf ; ≤ 2 min sans)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import torch  # noqa: E402

SCRIPT_HF = r'''
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
ids = json.load(open(sys.argv[2]))
maxmem = sys.argv[3]
kw = {"dtype": torch.bfloat16}
if maxmem != "cuda":
    g, c = maxmem.split(",")
    kw.update(device_map="auto", max_memory={0: g, "cpu": c})
else:
    kw.update(device_map="cuda")
try:
    m = AutoModelForImageTextToText.from_pretrained(sys.argv[1], **kw)
except Exception:
    m = AutoModelForCausalLM.from_pretrained(sys.argv[1], **kw)
m.eval()
x = torch.tensor([ids])
dev = next(m.parameters()).device
with torch.no_grad():
    out = m(input_ids=x.to(dev))
lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
nll = -lp.gather(1, x[0, 1:].to(lp.device).unsqueeze(1)).squeeze(1)
json.dump({"nll": nll.cpu().tolist()}, open(sys.argv[4], "w"))
'''


def nll_eval(loaded, ids: list[int]) -> list[float]:
    from acvram.engine.model import ForwardBatch
    from acvram.evaluate import _pertes_par_tranches
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    n = len(ids)
    alloc = BlockAllocator((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1, enable_prefix_cache=False)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)], dtype=torch.long)
    batch = ForwardBatch(tokens=torch.tensor(ids, dtype=torch.long), positions=torch.arange(n, dtype=torch.long),
                         seq_lens=[n], query_lens=[n], block_tables=[torch.tensor(blocks, dtype=torch.long)],
                         slot_mapping=slots, is_prefill=True)
    with torch.inference_mode():
        h = loaded.model(batch, return_hidden=True)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=h.device)
        pertes = _pertes_par_tranches(loaded.model, h, targets, 0)
    return pertes.float().cpu().tolist()


def nll_serve(loaded, tok, ids: list[int]) -> tuple[list[float], dict]:
    """Le forward du service au préfill : `Engine` construit la ForwardBatch
    comme /v1/completions (`_build_batch`), le même `model(batch)` est
    appelé avec `logits_positions` = toutes les positions."""
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    eng = Engine(loaded, tok, max_batch_size=1, max_model_len=len(ids) + 32, enable_cuda_graphs=False)
    eng.pipeline_actif = False
    eng.add_request(list(ids), SamplingParams(temperature=0.0, max_tokens=1), request_id="diag")
    eng._admit()                                    # alloue les blocs, comme au premier pas du service
    seq = eng.running[0]
    batch = eng._build_batch([seq], prefill=True)
    champs = {k: (type(getattr(batch, k)).__name__ if getattr(batch, k) is not None else None)
              for k in ("seq_ids", "gdn_store", "images", "mrope_positions", "deepstack") if hasattr(batch, k)}
    with torch.inference_mode():
        logits = loaded.model(batch, logits_positions=torch.arange(len(ids)))
        lp = torch.log_softmax(logits.float(), dim=-1)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=lp.device)
        nll = -lp[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.cpu().tolist(), champs


def nll_hf(source: str, ids: list[int]) -> list[float] | None:
    py = os.environ.get("HF_PYTHON", "/opt/ia/vLLM/.venv/bin/python")
    maxmem = os.environ.get("MAXMEM", "26GiB,80GiB")
    with tempfile.TemporaryDirectory() as d:
        s, i, o = os.path.join(d, "hf.py"), os.path.join(d, "ids.json"), os.path.join(d, "nll.json")
        open(s, "w").write(SCRIPT_HF)
        json.dump(ids, open(i, "w"))
        r = subprocess.run([py, s, source, i, maxmem, o], capture_output=True, text=True, timeout=1500)
        if r.returncode != 0:
            print(f"[diag] bras hf en échec : {r.stderr[-800:]}", flush=True)
            return None
        return json.load(open(o))["nll"]


def ppl(nll: list[float], depuis: int = 32) -> float:
    v = nll[depuis:] if len(nll) > depuis else nll
    return math.exp(sum(v) / len(v)) if v else float("nan")


def premiere_divergence(a: list[float], b: list[float], seuil: float = 1.0):
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > seuil:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("alias")
    ap.add_argument("--source", help="dossier HF bf16 pour le bras hf (sinon deux bras)")
    ap.add_argument("--jetons", type=int, default=300)
    ap.add_argument("--texte", default=None, help="fichier texte connu (défaut : acvram/data/calibration-anglais.txt)")
    ap.add_argument("--json")
    a = ap.parse_args()
    from acvram.engine.loader import load_model
    from acvram.server.chat import load_tokenizer
    from racine_modeles import racine_modeles
    chemin = a.alias if os.path.isdir(a.alias) else os.path.join(racine_modeles(), a.alias)
    tok = load_tokenizer(chemin)
    texte = open(a.texte or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../acvram/data/calibration-anglais.txt"),
                 encoding="utf-8").read()
    ids = tok.encode(texte)[: a.jetons - 1]
    bos = tok.bos_id()
    if bos is not None:
        ids = [bos] + ids
    loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=len(ids) + 32, max_concurrent_seqs=1)
    r = {"alias": chemin, "jetons": len(ids), "bos": bos, "bos_en_tete": ids[0] == bos if bos is not None else None}
    e = nll_eval(loaded, ids)
    s, champs = nll_serve(loaded, tok, ids)
    r.update({"nll_eval": e, "nll_serve": s, "champs_batch_serve": champs,
              "ppl_eval": round(ppl(e), 3), "ppl_serve": round(ppl(s), 3),
              "div_eval_serve": premiere_divergence(e, s),
              "delta_eval_serve_max": round(max(abs(x - y) for x, y in zip(e, s)), 4)})
    if a.source:
        h = nll_hf(a.source, ids)
        if h is not None:
            r.update({"nll_hf": h, "ppl_hf": round(ppl(h), 3), "div_eval_hf": premiere_divergence(e, h),
                      "div_serve_hf": premiere_divergence(s, h),
                      "delta_eval_hf_med": round(sorted(abs(x - y) for x, y in zip(e, h))[len(h) // 2], 4)})
    r["controle_eval_ppl_le_30"] = r["ppl_eval"] <= 30
    r["verdict"] = verdict(r)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[diag] {r['jetons']} jetons, bos {bos} en tête {r['bos_en_tete']} ; PPL(ctx≥32) eval {r['ppl_eval']} · serve {r['ppl_serve']}"
          + (f" · hf {r['ppl_hf']}" if "ppl_hf" in r else "") + f" ; champs serve {champs}")
    print(f"  première divergence eval/serve : {r['div_eval_serve']} (|Δ| max {r['delta_eval_serve_max']})"
          + (f" ; eval/hf : {r['div_eval_hf']} ; serve/hf : {r['div_serve_hf']} ; |Δ| eval−hf médian {r['delta_eval_hf_med']}" if "ppl_hf" in r else ""))
    print(f"  contrôle PPL eval ≤ 30 : {r['controle_eval_ppl_le_30']} ; verdict : {r['verdict']}")
    return 0


def verdict(r: dict) -> str:
    if "ppl_hf" in r:
        if r["div_eval_hf"] is None and r["div_serve_hf"] is None and r["ppl_eval"] <= 30:
            return "P1 : forward sain (eval ≈ serve ≈ hf) — la pathologie est dans le fenêtrage/corpus de perplexity : rejouer --jetons 1500 (fenêtre locale 1 024)"
        if r["div_eval_serve"] is not None and r["div_eval_serve"] < 8:
            return f"P2 : eval ≠ serve dès la position {r['div_eval_serve']} — la ForwardBatch d évaluation diffère du service (champs {r['champs_batch_serve']})"
        if r["div_serve_hf"] is not None:
            return f"P3 : eval ≈ serve mais ≠ hf dès la position {r['div_serve_hf']} — défaut du forward acvram hors gabarit, à nommer par couche"
        return "non tranché : divergences tardives, à lire par position"
    if r["div_eval_serve"] is not None:
        return f"P2 (sans hf) : eval ≠ serve dès la position {r['div_eval_serve']}"
    return ("P1 ou P3 (sans hf) : eval = serve ; " + ("PPL ≤ 30 : forward plausible, fenêtrage/corpus en cause" if r["ppl_eval"] <= 30
                                                     else "PPL > 30 sur un texte connu : les deux chemins sont faux — bras hf requis"))


if __name__ == "__main__":
    sys.exit(main())
