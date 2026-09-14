# Verdict — 1aj décodage, variantes D et E : lm_head est le coupable

poste2, 15/09. Suite de `verdict-1aj-variantesAC-15-09.md` (q/k innocentés,
coupable dans {v,o,lm_head}). Prédiction scellée avant mesure :
`prediction-1aj-variantesDE-15-09.md` — **D tient, E casse, lm_head
coupable**. Même régime que A/C (window 2048/2048/min-context 256,
wiki-gptq.txt), même témoin 9,3369. Conversions sur le vrai HDD
(`models_acvram_hdd/`, après correction du piège de lien symbolique
signalé par chef).

## Résultat

| variante | promue (int8) | en NVFP4 | PPL | ratio/témoin | verdict |
|---|---|---|---:|---:|---|
| D | lm_head seul | q,k,v,o | 9,2192 | **0,98739** | TIENT (≤1,01) |
| E | v_proj,o_proj | q,k,lm_head | 9,8174 | **1,05146** | CASSE (>1,015) |

## Conclusion : PRÉDICTION CONFIRMÉE — lm_head est le coupable, v/o innocents

D (protège UNIQUEMENT lm_head, tout le reste NVFP4) TIENT. E (protège
v_proj+o_proj, laisse lm_head en NVFP4) CASSE, dans la même fourchette
que la variante A (v/o/lm_head tous NVFP4, ratio 1,047). **v_proj et
o_proj tolèrent le NVFP4 sans problème** — c'est bien lm_head, et lui
seul parmi {v, o, lm_head}, qui refuse le W4A4.

Chaîne complète, les quatre variantes :

| variante | q | k | v | o | lm_head | ratio |
|---|---|---|---|---|---|---:|
| A | int8 | int8 | NVFP4 | NVFP4 | NVFP4 | 1,047 CASSE |
| C | NVFP4 | NVFP4 | int8 | int8 | int8 | 0,990 TIENT |
| D | NVFP4 | NVFP4 | NVFP4 | NVFP4 | int8 | 0,987 TIENT |
| E | NVFP4 | NVFP4 | int8 | int8 | NVFP4 | 1,051 CASSE |

Le motif est net : **le ratio suit uniquement l'état de lm_head**, pas
q/k/v/o (comparer A vs D : identiques sauf v/o, mais A casse — parce que
lm_head y est aussi en NVFP4 — et D tient parce que lm_head seul est
protégé, même avec v/o en NVFP4 ; comparer C vs E : identiques sauf
lm_head, C tient (lm_head protégé) et E casse (lm_head exposé)).

## Conséquence pour 1aj

**D est la variante gagnante** : elle tient la qualité (0,987) ET donne
le MEILLEUR gain d'octets prédit (1,85 Go/jeton, ~3,38 ms) puisque
lm_head seul est protégé, tout le reste (le plus gros volume : q, k, v,
o ET les experts) reste en NVFP4 économique. C'est exactement la lecture
« repli lm_head seul » que poste7 avait posée comme option de repli
propre — elle s'avère être la MEILLEURE option, pas un simple repli.

## Suite

Rendu à chef/poste7 : D recommandée pour 1aj décodage. Débit réel
(ms/jeton mesuré, pas seulement prédit par la borne octets) non mesuré
ce soir — hors du périmètre demandé (conversion + PPL seulement).
