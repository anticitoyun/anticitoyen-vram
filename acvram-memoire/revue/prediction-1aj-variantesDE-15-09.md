# Prédiction — 1aj décodage, variantes D et E (départage v/o vs lm_head)

Manon, 15/09, avant mesure (carte après P1 de Laurine, ~2 h). Suite de
`verdict-1aj-variantesAC-15-09.md` : A (q,k protégés) RÉFUTÉ 1,047 ; C
(v,o,lm_head protégés) TIENT 0,990 — q/k innocentés, coupable dans
{v_proj, o_proj, lm_head} sans les distinguer. D et E séparent lm_head
de v/o.

## Commandes (--promotion-classes, --format nvfp4 --no-awq --snr-floor 25,
## mêmes options que A/C, mêmes source/HDD)

```
acvram convert /mnt/4TO_SATACMR_2022/Modeles/models/Qwen3-Coder-30B-A3B-Instruct \
  --name Qwen3-Coder-30B-A3B-srcbf16-1ajvarD --format nvfp4 --no-awq --snr-floor 25 \
  --promotion-classes lm_head \
  -o /mnt/4TO_SATACMR_2022/Modeles/models_acvram/Qwen3-Coder-30B-A3B-srcbf16-1ajvarD

acvram convert /mnt/4TO_SATACMR_2022/Modeles/models/Qwen3-Coder-30B-A3B-Instruct \
  --name Qwen3-Coder-30B-A3B-srcbf16-1ajvarE --format nvfp4 --no-awq --snr-floor 25 \
  --promotion-classes v_proj,o_proj \
  -o /mnt/4TO_SATACMR_2022/Modeles/models_acvram/Qwen3-Coder-30B-A3B-srcbf16-1ajvarE
```

Éval identique à A/C : `--window 2048 --stride 2048 --min-context 256`,
`wiki-gptq.txt`, contre le témoin 9,3369.

## Octets/jeton prédits (mêmes données que la note de Laurine, q+o et
k+v décomposés par moitié — q_proj et o_proj ont la même taille
[hidden×hidden], k_proj et v_proj la même taille [hidden×kv_dim] par
construction GQA, donc partage égal du couple mesuré)

| classe | int8 (Go) | NVFP4 (Go) |
|---|---:|---:|
| q_proj | 0,41 | 0,225 |
| o_proj | 0,41 | 0,225 |
| k_proj | 0,05 | 0,03 |
| v_proj | 0,05 | 0,03 |
| lm_head | 0,32 | 0,17 |
| MoE (inchangé) | — | 1,02 |

**D** (lm_head int8, q/k/v/o NVFP4) : 0,225+0,225+0,03+0,03 (attn NVFP4)
+ 0,32 (lm_head int8) + 1,02 (MoE) = **1,85 Go/jeton**, ≈ **3,38 ms**
(borne 550 Go/s, même règle que la note de Laurine).

**E** (v_proj+o_proj int8, q/k/lm_head NVFP4) : 0,225 (q NVFP4) + 0,41
(o int8) + 0,03 (k NVFP4) + 0,05 (v int8) + 0,17 (lm_head NVFP4) + 1,02
(MoE) = **1,905 Go/jeton**, ≈ **3,48 ms**.

D économise plus qu'E (0,055 Go, ~0,1 ms) — attendu, lm_head seul pèse
moins que v_proj+o_proj ensemble.

## Prédiction PPL, avec le raisonnement qui la soutient

**Je prédis D TIENT (≤ 1,010) et E CASSE (> 1,015)** — c'est-à-dire
**lm_head est le coupable**, pas v_proj/o_proj. Raison : la sortie
`lm_head` projette directement vers 150k+ classes de vocabulaire ;
une petite erreur relative sur cette seule matrice se propage sans
amortissement dans le softmax final, alors que v_proj/o_proj sont
suivis d'une somme sur les têtes puis d'une couche résiduelle qui
peut partiellement moyenner un bruit local. C'est aussi le seul des
trois qui n'a PAS d'attention entre lui et la perte — l'erreur y est
la plus directe. Argument secondaire, plus faible : le témoin
int8-promu du parc (celui qui fixe 9,3369) protège déjà lm_head parmi
ses 4 classes, cohérent avec une pratique où lm_head est traité comme
sensible par défaut dans ce dépôt (voir `--lm-head-format` dédié dans
le CLI, absent pour q/k/v/o).

**Issue qui me gênerait** : si E tient ET D casse (lecture inverse de
Jérôme, « c'est v/o »). Je la trouve moins probable a priori, mais je
ne l'écarte pas — v_proj/o_proj sont deux fois plus gros ensemble que
lm_head (0,82 Go int8 contre 0,32), et un tenseur plus gros absorbe
généralement MIEUX le bruit par canal (plus de canaux pour diluer une
échelle mal choisie), ce qui plaiderait plutôt pour ma prédiction —
mais un contre-argument existe : plus de paramètres en NVFP4 veut
aussi dire plus de bruit total injecté dans le résidu, pas seulement
plus de dilution.

**Issue qui gênerait le plus** : les deux cassent (>1,015) — aucune
des trois classes seule ne suffit à expliquer A, il faudrait une
interaction (deux classes ensemble aggravent plus que la somme de
leurs effets seuls). Improbable pour une PPL (les tenseurs ne
partagent pas de calcul direct entre q/k/v/o/lm_head), mais pas
impossible.

## Plan

1. Préparé maintenant, à sec.
2. Sur carte (après Laurine P1, signal Jérôme) : les deux conversions,
   les deux éval, comparaison au témoin.
3. Rendu avec le même format que verdict-1aj-variantesAC-15-09.md.
