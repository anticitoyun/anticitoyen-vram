# 182 — verdict (poste1, 25/09) : z sans cast TENU (au bit, +0,40 % méd.) ; attention GQA au bit, TENUE en processus, NEUTRE au banc

* instrument : banc chat `scratchpad/poste2-piece102-etalon-hf-24-09/banc-chat-openai.py` (HTTP, serveur neuf par passe, fenêtre 25 s, `energie.py`) ; `scratchpad/poste1-p182-25-09/profil-ctx.py` (en processus, graphes) ; juges `tests/test_gdn_z_bf16_182.py`, `tests/test_attn_gqa_182.py`
* commit : 4f27a940 (branche poste1-mtp, origin/main fusionné), ATTENDU vérifié par les deux scripts
* régime : Qwen3.8-27B-unsloth-mixte-i8c, b = 8, carte 0 seule, -lgc 2700 (horloge moyenne 2 645 MHz), ACVRAM_ECO=off, cpu-safe 100 début/fin, graphes on, repli_eager=0 sur toutes les passes, aucune PASSE NULLE ; PID hors verrou : 4242 llama-server (permanent) seul
* scellé : `scratchpad/poste1-p182-25-09/scelle.md` (7d4c2d72, écrit avant toute mesure)
* mesuré : z B/A +0,40 % (méd.) ; GQA Δ −0,18 ms (ctx 600), −0,42 ms (ctx 2 000), banc +0,12 % (méd.)
* verdict : (1) TENU ; (2) au bit, en processus TENU (bas des fourchettes), banc NEUTRE (sous +0,2 %) → TENU composite NON atteint, FAUX non plus
* durée : prévu ≤ 15 min par prise ; tenu 9 min (ABBA z 07:34-07:43), 2,5 min (ctx 07:45-07:47), 9 min (ABBA gqa 07:48-07:57)

## Juges au bit (avant les prises)
33 verts. Premier passage : 12 288 écarts à q_len = 3 (lens [1, 17, 40]). Cause : faute de MONTAGE du test, pas du noyau.
Une séquence de longueur 1 à q_len = 3 donne slen ≤ 0 pour deux requêtes. Ni le noyau d'origine ni la variante n'écrivent
ces lignes : la sortie `torch::empty` (`acvram_kernels.cu:6300`) reste non initialisée. Le test comparait donc du
non-initialisé. Contrôle capable de rendre faux (`diag-lignes-slen0.py`, 16 cas) : **0 écart sur les lignes valides**,
tous les écarts sur les lignes slen ≤ 0. En service, seq_len compte les q_len jetons vérifiés, donc seq_len ≥ q_len
(duck.ai 3/3 via poste4 : conséquence de construction chez vLLM, pas un invariant imposé). Montage corrigé
(`lens = max(n, q_len)`) ; second défaut de montage : dtype `bfloat16` → `bf16` (clé de `kvcache.py:327`).

## (1) z — ABBA servi, 10 passes (A = ACVRAM_GDN_Z_BF16=0)
| bras | t/s par passe | médiane | J/jeton net (méd.) |
|---|---|---|---|
| A | 327,4 328,5 327,7 326,6 327,0 | 327,4 | 0,8529 |
| B | 328,7 329,4 326,4 328,7 326,3 | 328,7 | 0,8547 |

Médiane +0,40 % : prédite +0,4 %, seuil +0,2 % → TENU au critère scellé. Réserve : écart des moyennes +0,14 %, SE ≈ 0,22 %,
donc non résolu au sens de Welch. Le gain de 96 µs/pas est sous la résolution de 5 passes.

## (2) attention GQA — en processus (A B B A A B), puis ABBA servi (A = ACVRAM_PA_GQA=0)
| contexte | A (ms/pas, 3 passes) | B | Δ | prédit (fourchette) |
|---|---|---|---|---|
| ≈ 600 | 19,364 19,336 19,361 | 19,149 19,185 19,186 | −0,18 | −0,25 (−0,15 / −0,35) |
| ≈ 2 000 | 21,565 21,529 21,533 | 21,100 21,146 21,126 | −0,42 | −0,6 (−0,4 / −0,9) |

Contrôle de bras : compteur `paged_attn_gqa_lancements` = 0 en A, 48 (captures) puis 144 en B. Étendue intra-bras
≤ 0,05 ms, Δ ≥ 3,6 × l'étendue.
Banc chat : A 331,0 328,8 329,4 331,4 325,2 (méd. 329,4, σ 2,46) ; B 329,8 332,2 329,2 330,9 329,4 (méd. 329,8, σ 1,25).
Médiane +0,12 % (prédit +0,4 %), moyennes +0,35 %, SE 0,38 % → NEUTRE, non résolu. J/jeton net 0,8584 → 0,8529.

Issue gênante (latence de la chaîne par jeton, risque d'occupation avec 32 blocs signalé par Luna) : non réalisée. B < A
aux deux contextes, et le gain croît avec le contexte (0,18 → 0,42), comme un gain en blocs/lectures. Il reste pourtant en
bas de fourchette : la sérialisation des 6 têtes dans le warp en reprend une partie.

## Suite proposée (décision au chef)
* Les deux sont au bit, donc garder les deux défauts ne change aucune sortie ; aucun n'est négatif.
* Le banc court (contextes 256-512) ne résout pas des gains ≤ 0,4 % en 5 passes. Pour une cellule publiée, il faut ≥ 10
  passes par bras ou un contexte ≥ 1 024.
* Levier suivant de la 182 à sec : (3) porte d'attention (−0,05 ms), écartée par le feu de chef.

Journaux (ignorés par git) : `prise-abba-z.log`, `prise-ctx.log`, `prise-abba-gqa.log`, `juges-au-bit.log`,
`diag-lignes-slen0.log` ; cellules `cellule-{z,gqa}-Qwen3.8-27B-unsloth-mixte-i8c-b8.jsonl`.
