#!/usr/bin/env python3
"""PPL bf16 de GLM-4.7-Flash par HF transformers direct (pas acvram) --
juge final ≤ ×1,01, demande de Jerome (15/09) apres que ni la conversion
bf16 par defaut ni `--host-exec stream` n'aient laisse assez de marge
sur la carte pour `acvram eval` (voir revue/verdict-glm-awq-mla-15-09.md,
section P6).

`device_map="auto"` (accelerate, installe ce soir dans le venv vLLM --
absent avant) delegue le delestage hote a accelerate plutot qu'au
planificateur acvram : plus lent (copies host<->device par couche), mais
sans le probleme de marge insuffisante puisqu'accelerate CONNAIT la VRAM
reelle disponible au moment du placement, pas une estimation figee a la
conversion.

Meme regime que `acvram eval` (evaluate.py::perplexity) : fenetre et pas
2048, min_context 256, corpus wiki-gptq.txt -- meme calcul de
`first_new` (ici constant a 256, puisque window==stride rend le terme
`(window-stride)-1` negatif). Encodage : `add_special_tokens=False`,
comme `acvram.server.chat.Tokenizer.encode` (memes fichiers
tokenizer.json/tokenizer_config.json, donc memes identifiants).

    outils/carte.sh /opt/ia/vLLM/.venv/bin/python outils/glm-ppl-bf16-hf.py
"""
import math
import sys
from pathlib import Path

SOURCE = "/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16"
CORPUS = "/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt"
WINDOW = 2048
STRIDE = 2048
MIN_CONTEXT = 256
MAX_TOKENS = 8192


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SOURCE)
    texte = Path(CORPUS).read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer.encode(texte, add_special_tokens=False)[:MAX_TOKENS]
    print(f"  corpus : {len(ids)} jetons", flush=True)

    modele = AutoModelForCausalLM.from_pretrained(
        SOURCE, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "26GiB", "cpu": "80GiB"})
    modele.eval()

    total_nll = 0.0
    counted = 0
    n_windows = max(1, (len(ids) - 1 + STRIDE - 1) // STRIDE)
    for w, start in enumerate(range(0, len(ids) - 1, STRIDE)):
        chunk = ids[start:start + WINDOW]
        if len(chunk) < 2:
            break
        input_ids = torch.tensor([chunk], dtype=torch.long)
        with torch.no_grad():
            logits = modele(input_ids).logits[0, :-1].to(torch.float32)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=logits.device)

        first_new = 0 if start == 0 else max(0, (WINDOW - STRIDE) - 1)
        first_new = max(first_new, MIN_CONTEXT)
        if first_new >= logits.shape[0]:
            continue
        nll = torch.nn.functional.cross_entropy(
            logits[first_new:], targets[first_new:], reduction="sum")
        total_nll += float(nll)
        counted += int(targets[first_new:].numel())
        print(f"  fenetre {w + 1}/{n_windows} : {int(targets[first_new:].numel())} "
             f"jetons notes, cumul {counted}", flush=True)
        if start + WINDOW >= len(ids):
            break

    ppl = math.exp(min(total_nll / counted, 60.0))
    print(f"RESULTAT ppl_bf16_hf={ppl:.4f} n_jetons_notes={counted} "
         f"windows={n_windows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
