# P3 (4) duel Qwen3-VL-30B sur 665eeacc (main, moteur 0.6.34 final) — 21/09

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-4.sh`
* commit : 665eeacc (worktree poste2-w-21-09)
* régime : NOMINAL, graphes=on(hybrides≤4), kv=int8, vision=bf16(eager), deepstack=3, mrope=[24,20,20]
* scellé : TTFT ≤ 0,094 s, J ≤ 40,0
* mesuré : 20/20 images décodées pleinement — TTFT médian **4,2845 s** (témoin 0,0938 s) ; J net médian **175,9** (témoin 40,2) ; top1 identique au témoin sur **17/20** (3 divergences)
* verdict : **RÉFUTÉ, nettement** — TTFT ×45 le seuil (4,28 s contre 0,094 s), J ×4,4 le seuil (175,9 contre 40,0), pas de marge d'incertitude possible ici. 3/20 divergences top1 avec le témoin llama.cpp (préfixe non identifié dans le verdict — sha référence `d6f55e0c293e`). `-lgc` posé puis `-rgc` relâché proprement (225→787 MHz).
* durée : 370 s (16:01:51–16:08:01), prévu ≤ 30 min

nvidia-smi propre avant/après (seul PID 4286).

## Suite
Écart massif (TTFT et J) — probablement le préfill vision (déquantification/offload, pas d'accélération Marlin pour ce modèle) ; hors mon domaine de diagnostic causal, à qui touche le chemin de préfill vision. Les 3 divergences top1 méritent d'être nommées (quel(s) des 20 flux) avant d'écarter une cause de justesse numérique.
