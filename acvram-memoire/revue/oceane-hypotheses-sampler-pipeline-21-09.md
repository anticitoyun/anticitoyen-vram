# Sampler vectorisé : −33,5 µs isolé, −2,2 % en service — trois hypothèses à sec (Océane, 21/09)

instrument : lecture de `verdict-banc-sampler-665eeacc-21-09` (lent 55,1 µs / lot 21,6 µs), `verdict-sampler-b12-6f-v2-21-09` (A 1 541,9 · 1 538,9 ; B 1 506,2 · 1 471,9 · 1 522,5), `scratchpad/laurine-b12-21-09/chaine-sampler-6f-v2.sh`, `engine/pipeline.py`, `engine/sampler.py` ; 0 min de carte
commit : d12b2a08 (cellule), 665eeacc (banc)

Fait de départ : le chemin pris en service est le glouton (harnais `banc-llamacpp-16-09.py:151` temperature 0.0 → `SamplingParams.greedy`) — B = `argmax(bf16)` + zéros (sampler.py, chemin lot), A = cast fp32 + argmax + gather + logsumexp (`_sample_lent`). Côté carte, B fait 33,5 µs de MOINS par pas = −0,4 % d un pas de 7,8 ms ; la cellule dit +2,2 % de temps : cinq fois plus, et du mauvais signe. Le chiffre de la cellule n est pas celui du noyau.

## H1 — la plus probable : confusion d ordre et de dérive, pas le sampler
* `chaine-sampler-6f-v2.sh` joue A1 A2 A3 puis B1 B2 B3 (lignes finales : `passe A1…A3` puis `passe B1…B3`), 405 s au total, sans horloge SM par fenêtre dans le verdict (REGLES § 3 : « toute cellule b=12 porte l horloge SM moyenne ») ; A1 rejetée (pic étranger), B2 = 1 471,9 s écarte de 50 t/s (3,3 %) de B3 — la dispersion INTRA-bras de B dépasse l écart A/B (34 t/s).
* mesure qui tranche (Laurine/Manon, ≈ 8 min) : même chaîne, ordre entrelacé **A B B A A B B A** (le témoin encadre le bras, REGLES § 3 « A B B A »), `nvidia-smi dmon` ou `clocks.sm` médian par fenêtre dans l en-tête, fenêtres 20 s. Prédit : |B/A − 1| ≤ 0,5 % (le sampler pèse 0,4 % du pas dans les deux sens) ; H1 réfutée si B < A de ≥ 1,5 % en entrelacé ET horloges égales à 1 %.

## H2 — recouvrement du pipeline : la frontière est liée au processeur, un noyau de moins ne rend rien
* `engine/pipeline.py` `_pipeline_suite` : rejeu n enfilé → `_sample_only` (2 lancements en B, 6 en A) → `event.record()` ; le processeur prépare le lot n+1 (`_build_batch_device`, `preparer`) pendant que la carte finit rejeu n + échantillonnage ; si le processeur est le chemin critique, retirer 33 µs de carte ne change rien — et B ajoute ≈ 2-4 µs de Python (`sampler_lot_actif`, `any(p.logprobs…)`), pas 170. H2 explique un gain nul, pas une perte de 2,2 % : elle ne tient que si H1 tombe.
* mesure (Laurine, ≤ 3 min de carte) : nsys de 50 pas en A et en B (instrument `laurine-b12-21-09/nvtx-glue-h2.py`), lire le TROU de carte entre le dernier noyau du pas n et le premier du pas n+1 : prédit égal à ±20 µs ; H2 si le trou est plus long en B de ≥ 150 µs.

## H3 — argmax bf16 lu directement dans le tampon statique du graphe
* B lit `logits` = sortie statique du GraphRunner sans la copie fp32 que A faisait d abord ; ordre de flux respecté (ids au bit tenus), coût mémoire identique (le tampon vient d être écrit, il est en L2). Peu probable ; le banc isolé sur un tenseur frais ne le voit pas.
* mesure (≤ 1 min) : variante de `banc-sampler.py` où les logits sortent d un graphe capturé (`torch.cuda.CUDAGraph` d un GEMM bf16 [12, V]) : prédit lot toujours ≤ lent.

Décision proposée : H1 d abord (8 min) ; si elle tient, le sampler vectorisé n est ni gain ni perte mesurable à b=12 (0,4 %) — le remettre en défaut n apporte rien, le laisser en opt-in ne coûte rien ; l écart à vLLM (3,5 %) est ailleurs (H2 de Laurine : frontière 0,43 ms = 5,5 % du pas, dont l échantillonnage n est que 55 µs).
durée : 0 min de carte · verdict : H1 à jouer avant tout code
