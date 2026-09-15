#!/usr/bin/env python3
"""PPL teacher-forcing À b=12 — voyant `narrow_gemm` corrigé (poste7, revue/
poste7-narrow-b12-16-09.md, 16/09, amendant le b=1 de `ppl-decode-mma-
coder30b.py` : 1,000000 exact = chemin non pris, pas jugé).

Lu dans `kernels/__init__.py:544,681` : `_NARROW_MIN(2) <= n <= 16` — à b=1,
n=1, FAUX quel que soit `ACVRAM_NARROW_GEMM` : jamais appelé, le contrôle ne
peut pas rendre faux (REGLES §5). CE script décode 12 séquences EN PARALLÈLE
(n=12 par pas, dans la fenêtre 2-16) pour RÉELLEMENT solliciter le chemin
étroit côté int8 (côté NVFP4 reste coupé par `_NARROW_NVFP4=0`, non touché
ici) — 12 × 2 048 = 24 576 jetons NOTÉS EN DÉCODAGE (un jeton par séquence
par pas, jamais un prefill : `--window` sur un prefill prendrait M=2048, qui
NE prend PAS narrow — piège nommé par poste7).

TROIS mesures, un seul montage, trois bras (A, B, A-eager) :

1. PPL teacher-forcing B/A, seuil scellé (chef) **1,000 ± 0,002**.
2. PREUVE que le chemin a été pris : compte de lancements `narrow_gemm`,
   publié dans RESULTAT ; DOIT valoir (projections éligibles) × (pas de
   décodage) pour B, 0 pour A — sinon **non jugé**, exactement comme le b=1.
3. Taux de divergences top-1 (argmax du modèle, PAS le jeton forcé) sur les
   24 576 positions, comparé HORS LIGNE (`compare-narrow-b12-16-09.py`) à un
   témoin A-graphes/A-eager (`TIES_EAGER=1`, même script, NARROW=0) : seuil
   scellé **taux(B vs A) ≤ 1,2 × taux(témoin)** — règle le s3@3 de poste3
   (`verdict-narrow-voyants-15-09.md` § (2)) sur les mêmes positions et le
   même critère top-1 plutôt qu'un critère à ulp inapplicable au décodage.

    ACVRAM_NARROW_GEMM=0 outils/carte.sh python outils/ppl-narrow-b12-coder30b.py A.json
    ACVRAM_NARROW_GEMM=1 outils/carte.sh python outils/ppl-narrow-b12-coder30b.py B.json
    ACVRAM_NARROW_GEMM=0 TIES_EAGER=1 outils/carte.sh python outils/ppl-narrow-b12-coder30b.py A-eager.json
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("ACVRAM_MOE_DECODE_MMA", "1")

MODEL = "/mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3-Coder-30B-A3B-nvfp4"
CORPUS = "/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt"
N_SEQ = 12
CHUNK = 2048                      # 12 x 2048 = 24 576 jetons DE DECODAGE
MAX_MODEL_LEN = CHUNK + 64


def main() -> int:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer
    from acvram.evaluate import _load_corpus
    from acvram.kernels import get_extension

    sortie = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ppl-narrow-b12.json"
    assert os.environ.get("ACVRAM_MOE_DECODE_MMA") in ("0", "1"), "flag pas pose avant import"
    assert os.environ.get("ACVRAM_NARROW_GEMM") in ("0", "1"), "flag pas pose avant import"
    eager = os.environ.get("TIES_EAGER", "0") == "1"

    ext = get_extension()
    compte = {"narrow_gemm": 0}
    if ext is not None and hasattr(ext, "narrow_gemm"):
        orig = ext.narrow_gemm

        def compte_et_appelle(*a, **kw):
            compte["narrow_gemm"] += 1
            return orig(*a, **kw)

        ext.narrow_gemm = compte_et_appelle

    tokenizer = load_tokenizer(MODEL)
    texte = _load_corpus(CORPUS)
    ids_plats = tokenizer.encode(texte)[: N_SEQ * CHUNK]
    assert len(ids_plats) == N_SEQ * CHUNK, (
        f"corpus trop court : {len(ids_plats)} < {N_SEQ * CHUNK}")
    chunks = [ids_plats[k * CHUNK:(k + 1) * CHUNK] for k in range(N_SEQ)]
    print(f"  corpus : {N_SEQ} x {CHUNK} jetons, {os.path.basename(CORPUS)}", flush=True)

    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=MAX_MODEL_LEN)
    engine = Engine(loaded, tokenizer, max_batch_size=N_SEQ,
                    max_model_len=MAX_MODEL_LEN, enable_cuda_graphs=not eager)
    engine._eos = set()
    if not eager:
        engine.warm_graphs()

    etat = {f"s{k}": {"pos": 0, "nll": 0.0, "n": 0} for k in range(N_SEQ)}
    top1: dict[str, list] = {f"s{k}": [] for k in range(N_SEQ)}
    pas_petits, pas_total = 0, 0

    def sample_force(logits, seqs):
        nonlocal pas_petits, pas_total
        pas_total += 1
        if logits.shape[0] <= 32:
            pas_petits += 1
        toks, lps = [], []
        for i, s in enumerate(seqs):
            k = int(s.request_id[1:])
            e = etat[s.request_id]
            p = e["pos"]
            chunk = chunks[k]
            l32 = logits[i].to(torch.float32)
            top1[s.request_id].append(int(l32.argmax()))
            cible = chunk[p + 1] if p + 1 < CHUNK else chunk[p]
            lp = torch.log_softmax(l32, dim=-1)
            e["nll"] += float(-lp[cible])
            e["n"] += 1
            e["pos"] = p + 1
            toks.append(cible)
            lps.append(float(lp[cible]))
        tok = torch.tensor(toks, device=logits.device, dtype=torch.long)
        logprob = torch.tensor(lps, device=logits.device, dtype=torch.float32)
        return tok, logprob

    engine._sample_only = sample_force
    for k in range(N_SEQ):
        engine.add_request([chunks[k][0]],
                           SamplingParams(temperature=0.0, max_tokens=CHUNK - 1),
                           request_id=f"s{k}")

    n_pas = 0
    while engine.running or engine.waiting:
        engine.step()
        n_pas += 1
        if n_pas > CHUNK + 20:
            raise RuntimeError("le lot ne se termine pas")

    nll_total = sum(e["nll"] for e in etat.values())
    n_total = sum(e["n"] for e in etat.values())
    ppl = math.exp(min(nll_total / n_total, 60.0))
    proof = {
        "acvram_moe_decode_mma": os.environ["ACVRAM_MOE_DECODE_MMA"],
        "acvram_narrow_gemm": os.environ["ACVRAM_NARROW_GEMM"],
        "eager": eager,
        "ppl": ppl, "n_jetons_notes": n_total,
        "pas_t_le_32": f"{pas_petits}/{pas_total}", "n_pas": n_pas,
        "lancements_narrow_gemm": compte["narrow_gemm"],
    }
    print(f"RESULTAT {json.dumps(proof, ensure_ascii=False)}", flush=True)
    json.dump({"preuve": proof, "top1": top1}, open(sortie, "w"))
    print(f"[INFO] ecrit {sortie}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
