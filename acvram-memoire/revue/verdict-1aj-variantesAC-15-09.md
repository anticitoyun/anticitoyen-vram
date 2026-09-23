# Verdict — 1aj décodage, variantes A et C (test par projection)

Manon, 15/09. Protocole Laurine (`revue/1aj-decodage-14-09.md`), même
régime `acvram eval` que le témoin (`window 2048/2048/min-context 256`,
`wiki-gptq.txt`). Témoin int8-promu (4 classes, snr_floor 25) : PPL
9,3369. Seuils : ≤ ×1,01 tient, > ×1,015 réfute.

## Résultat

| variante | promues (int8) | en NVFP4 | PPL | ratio/témoin | verdict |
|---|---|---|---:|---:|---|
| A | q_proj, k_proj | v, o, lm_head | 9,7802 | **1,04748** | RÉFUTÉ (>1,015) |
| C | v_proj, o_proj, lm_head | q, k | 9,2465 | **0,99032** | TIENT (≤1,01) |

96 tenseurs promus en A (48×2, q+k), 97 en C (48×2, v+o, +1 lm_head) —
comptes attendus, vérifiés dans le journal de conversion.

## Conclusion : l'issue « q/k refusent » est RÉFUTÉE — c'est v/o(/lm_head)

Exactement la condition de réfutation nommée par Laurine (« A > ×1,015
ET C ≤ ×1,01, c'est v/o ») : quand q_proj/k_proj restent en NVFP4 (C),
la PPL TIENT ; quand v_proj/o_proj/lm_head restent en NVFP4 (A), elle
casse. **q/k tolèrent le W4A4 NVFP4 sans problème** ; le coupable est du
côté v/o ou lm_head (les trois sont promus ensemble dans C, non
distingués ici — variante A' de la note, non testée ce soir, séparerait
lm_head des deux autres si besoin).

## Conséquence pour 1aj

La variante recommandable par la qualité est **C** (v,o,lm_head int8 ;
q,k NVFP4) — mais Laurine notait que le gain d'octets/ms de C (1,88
Go/jeton, ~3,4 ms prédits) est PLUS PETIT que celui d'A (2,30→1,97,
3,6 ms) puisque q/k pèsent moins que v/o+lm_head. La variante qui
économise le plus (A) est aussi celle qui casse la qualité — 1aj décodage
n'a donc PAS de variante gagnante sur les deux tableaux à la fois avec
seulement 2 des 4 classes protégées.

## Suite

Rendu à Jérôme/Sage : C est la seule des deux à tenir la qualité,
gain moindre que prédit pour A (qui casse). Reste ouvert (non fait ce
soir, hors du périmètre demandé) : mesurer le débit/ms réel de C pour
voir si son gain plus modeste vaut le coût de conversion, et éventuellement
la variante A' (note Laurine) pour isoler lm_head de v/o.
