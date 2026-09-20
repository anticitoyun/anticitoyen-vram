#!/usr/bin/env python3
"""Arbitre prefill, générique graphes/eager — Coder-30B b=4 W4A16 (16/09).

Ordre poste7 §12 (`revue/poste7-duel-verdict-16-09.md`) : Coder-30B, b=4
(`ACVRAM_HYBRID_SLOTS=4`), W4A16 (`ACVRAM_MOE_DECODE_MMA=0`), 84 points
(pas 28), sous GRAPHES et EAGER séparément, jugés contre le PREFILL du
même texte (arbitre §9 : prompt réel ≥ 128+k jetons, jamais un autre
chemin batché comme référence). On compte les points HORS
quasi-égalité : `top1_test != top1_ref` ET
`ref[top1_ref] - ref[top1_test] > 1,0` logit (mesuré sur la RÉFÉRENCE,
pas sur le test — un swap que la référence elle-même juge presque à
égalité n'est pas une divergence).

Scellé : eager ≈ 0 et graphes ≥ 3 → défaut générique confirmé, localisé
par k (le plus petit pas hors quasi-égalité) ; alors, même processus,
comparaison bit à bit de `x` d'entrée du pas 2 (graphes contre eager) :
jetons, embedding, positions. Graphes ≤ 1 et eager ≤ 1 → bruit
d'échantillon, générique clos.

Réutilise le montage de `outils/ppl-decode-mma-coder30b.py` (`Engine`,
`_sample_only` détourné pour forcer le jeton réel du corpus) et de
`outils/equivalence-glm-2couches.py` (préfixe croissant, `b=1`, sans
graphes, comme référence prefill).
"""
import json
import math
import os
import sys
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Lus au chargement du module acvram : AVANT tout import acvram.
os.environ.setdefault("ACVRAM_MOE_DECODE_MMA", "0")     # W4A16 explicite
os.environ.setdefault("ACVRAM_HYBRID_SLOTS", "4")        # b=4

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
CORPUS = "/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt"
N_PROMPT = 128
N_POINTS = 84
MAX_MODEL_LEN = N_PROMPT + N_POINTS + 64
SEUIL_LOGIT = 1.0


def main() -> int:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer
    from acvram.evaluate import _load_corpus

    assert os.environ.get("ACVRAM_MOE_DECODE_MMA") == "0", "W4A16 pas pose avant import"

    tokenizer = load_tokenizer(MODEL)
    texte = _load_corpus(CORPUS)
    ids = tokenizer.encode(texte)[: N_PROMPT + N_POINTS + 8]
    assert len(ids) >= N_PROMPT + N_POINTS, "corpus trop court"
    print(f"  corpus : {len(ids)} jetons pris, prompt={N_PROMPT} points={N_POINTS}",
         flush=True)

    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=MAX_MODEL_LEN)

    # --- référence : préfixe croissant, b=1, sans graphes (arbitre §9) ---
    print("[1/3] référence prefill (b=1, sans graphes, préfixe croissant)...", flush=True)
    ref_logits = []
    for k in range(N_PROMPT, N_PROMPT + N_POINTS):
        engine = Engine(loaded, tokenizer, max_batch_size=1,
                        max_model_len=MAX_MODEL_LEN, enable_cuda_graphs=False)
        capture = {}
        orig_emit = engine._emit
        def espion(logits, seqs, _c=capture):
            _c["v"] = logits[0].to(torch.float32).clone()
            return orig_emit(logits, seqs)
        engine._emit = espion
        engine.add_request(ids[:k], SamplingParams(temperature=0.0, max_tokens=1),
                           request_id="ref")
        for _ in range(3):
            if not engine.running and not engine.waiting:
                break
            engine.step()
        if "v" not in capture:
            raise RuntimeError(f"référence : aucune capture au pas k={k}")
        ref_logits.append(capture["v"])
        del engine
    print(f"  {len(ref_logits)} positions de référence capturées", flush=True)

    def decoder(sous_graphes: bool) -> list:
        engine = Engine(loaded, tokenizer, max_batch_size=1,
                        max_model_len=MAX_MODEL_LEN, enable_cuda_graphs=sous_graphes)
        engine._eos = set()
        captures = []
        etat = {"pos": N_PROMPT - 1}

        def sample_force(logits, seqs):
            captures.append(logits[0].to(torch.float32).clone())
            p = etat["pos"]
            cible = ids[p + 1] if p + 1 < len(ids) else ids[p]
            etat["pos"] = p + 1
            lp = torch.log_softmax(logits[0].to(torch.float32), dim=-1)
            tok = torch.tensor([cible], device=logits.device, dtype=torch.long)
            return tok, lp[cible].reshape(1)

        engine._sample_only = sample_force
        engine.add_request(ids[:N_PROMPT], SamplingParams(temperature=0.0, max_tokens=N_POINTS),
                           request_id="d")
        while engine.running or engine.waiting:
            engine.step()
        return captures[:N_POINTS]

    print("[2/3] décodage EAGER (sans graphes)...", flush=True)
    logits_eager = decoder(False)
    print("[3/3] décodage GRAPHES (avec graphes)...", flush=True)
    logits_graphes = decoder(True)

    def juger(nom: str, logits_test: list) -> dict:
        hors = []
        for i in range(N_POINTS):
            ref = ref_logits[i]
            test = logits_test[i]
            top1_ref = int(ref.argmax())
            top1_test = int(test.argmax())
            if top1_test == top1_ref:
                continue
            delta = float(ref[top1_ref] - ref[top1_test])
            if delta > SEUIL_LOGIT:
                hors.append({"k": i + 1, "pos": N_PROMPT + i,
                            "top1_ref": top1_ref, "top1_test": top1_test,
                            "delta_logit_sur_ref": delta})
        print(f"{nom}: {len(hors)}/{N_POINTS} hors quasi-égalité — "
             f"{[h['k'] for h in hors]}", flush=True)
        return {"n_hors": len(hors), "points": hors}

    r_eager = juger("EAGER", logits_eager)
    r_graphes = juger("GRAPHES", logits_graphes)

    n_eager, n_graphes = r_eager["n_hors"], r_graphes["n_hors"]
    if n_eager <= 1 and n_graphes >= 3:
        verdict = "défaut générique CONFIRMÉ"
    elif n_eager <= 1 and n_graphes <= 1:
        verdict = "bruit d'échantillon — générique CLOS"
    else:
        verdict = "zone grise (ni eager<=1&graphes>=3, ni les deux <=1) — à trancher par poste7"
    print(f"\nVERDICT: eager={n_eager} graphes={n_graphes}  {verdict}")

    out = Path("/tmp") / "arbitre-prefill-coder30b-resultat.json"
    with open(out, "w") as fh:
        json.dump({"eager": r_eager, "graphes": r_graphes, "verdict": verdict}, fh, indent=2)
    print(f"\nécrit: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
