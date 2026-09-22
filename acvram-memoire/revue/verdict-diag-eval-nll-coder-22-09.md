# diag-eval-nll sur Coder nvfp4 (bras eval/serve) — SAIN, défaut propre à Gemma 4 — 22/09 (poste2)

* instrument : `outils/carte.sh .venv/bin/python outils/gpu/mesure/diag-eval-nll.py Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c --jetons 300`, sans `--source` (2 bras, ≤2 min)
* commit : main à jour
* contrôle qui rend faux : PPL(ctx≥32) bras eval ≤ 30 sur ce texte connu — **TENU** (8,434)
* mesuré : `PPL eval 8,434 · serve 8,434` — identiques ; première divergence eval/serve : aucune (`|Δ| max 0,002` nat).
* verdict : **SAIN** — sur le Coder, `acvram eval` rend une PPL plausible et identique au chemin serve. Comparé au rejeu Gemma 4 31B de tout à l'heure (`verdict-diag-eval-nll-37-22-09.md`, PPL 227 232, contrôle FAUX) : **le défaut est propre à Gemma 4**, pas un bug général de `logits_positions`/ForwardBatch commun à tous les modèles (l'option « faux → logits_positions hors dernière position cassé pour tous » du groupe est donc écartée). Reste à isoler ce qui distingue Gemma 4 (fenêtrage du corpus ? gabarit de conversation ? spécificité de tokenizer/BOS déjà nommée dans `ae519990`).
* durée : ~2 min de carte

## Suite
Bras hf sur le 31B (--source bf16, ≤10 min) pour trancher P1/P3 sur Gemma 4 spécifiquement — enchaîné maintenant.
