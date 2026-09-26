# Banc de prefill MoE : GEMM groupée vs déquant+grouped_mm

Date : 12/09/2026 — poste4, chantier 2.
Modèle : Qwen3-Coder-30B-A3B-nvfp4, RTX 5090.

## Résultat initial (INVALIDE)

Le premier passage mesurait un ratio de 1,00 partout. Le contrôle a montré
que le banc mesurait le **mauvais chemin** : après le premier step, le moteur
active le prefill adaptatif (tranche ≤ 32 jetons), qui appelle
`_forward_grouped` → `nvfp4_gemv_grouped_gateup` (GEMV par expert). Ce chemin
**n'est pas contrôlé** par `ACVRAM_PREFILL_DEQUANT` — les deux colonnes
exécutaient le même code.

## Contrôle 1 : le noyau est bien appelé

Interception de `_gemm` sur chaque `MoEBlock` (model.py:717) :

- Sans `ACVRAM_PREFILL_DEQUANT` : **144 appels** (48 couches × 3 GEMM) ✓
- Avec `ACVRAM_PREFILL_DEQUANT=1` : **0 appels** (repli _pile_bf16) ✓

## Contrôle 2 : profil par événements CUDA

Premier prefill (moteur frais, pas de chauffe, médian de 5), L=64/256/512 :

| L    | Direct (ms) | _gemm (ms) | % pas | Repli (ms) | _pile_bf16 (ms) | % pas | ratio |
|-----:|------------:|-----------:|------:|----------:|----------------:|------:|------:|
|   64 |         539 |         21 |  4,0  |       551 |             206 | 37,4  |  0,98 |
|  256 |         619 |         51 |  8,3  |       656 |             202 | 30,7  |  0,94 |
|  512 |         653 |         92 | 14,1  |       728 |             215 | 29,6  |  0,90 |

### Dénominateurs

- **Direct** = `_forward_prefill_grouped`, chemin NVFP4 (`_gemm` →
  `nvfp4_gemm_grouped`, model.py:719).
- **Repli** = même fonction, chemin dequant (`_pile_bf16` →
  `dequantize_nvfp4` + `torch._grouped_mm`, model.py:763).
- **Durée** = `torch.cuda.Event.elapsed_time` sur un `Engine.step()` complet
  (le premier, non découpé).
- **Moteur frais** à chaque répétition : `load_model` + `Engine()` + un
  `step()`. Le JIT CUDA (compilation des noyaux) est inclus : ~500 ms de
  plancher commun aux deux chemins.
- 5 répétitions, valeur médiane.

### Lecture

1. La déquantification `_pile_bf16` pèse **30-37 % du premier pas** — c'est
   le coût que le noyau NVFP4 supprime (il lit les poids en 4 bits
   directement).
2. Le noyau NVFP4 `_gemm` ne pèse que **4-14 %** du pas : il est plus
   léger que la déquantification, mais la différence est masquée par le
   plancher JIT (~500 ms).
3. Le gain net est de **~10 % à L=512** (0,90×). En régime chaud, sans le
   plancher JIT, la part MoE est plus visible et le gain plus marqué.

### Pourquoi le banc initial ne voyait rien

Le banc initial faisait 2 passes de chauffe sur le même `Engine` avant de
mesurer. Le prefill adaptatif (runner.py:542) découpait les requêtes
suivantes en tranches ≤ `_MOE_GROUPED_MAX=32` jetons (model.py:875), qui
empruntent `_forward_grouped` → `nvfp4_gemv_grouped_gateup` (model.py:789).
Ce chemin GEMV est indépendant de `ACVRAM_PREFILL_DEQUANT` : les deux
colonnes mesuraient le même code.

## Régime chaud — première version INVALIDE (12/09 soir)

La première mesure « a/b/c à ±0,6 % » était fausse deux fois : (1) le pas
durait ~27 ms à L=512 **et** à L=2048 — signature du cache de préfixe
(`enable_prefix_cache=True` par défaut, runner.py:229) sur un prompt
identique à chaque répétition : seul un résidu était précalculé, t ≤ 32,
GEMV pour tous les bras ; (2) la condition est `t <= _MOE_GROUPED_MAX` →
GEMV (model.py:875), donc MAX=99999 force le **GEMV**, pas le GEMM. Les
compteurs (`_forward_prefill_grouped=0` partout) le disaient ; je l'ai
attribué à la lecture unique de la variable de module au lieu de le lire
comme un résultat. Section retirée, remplacée par la suivante.

## Régime chaud (13/09) — cache de préfixe coupé, chemins prouvés

Moteur chaud (2 passes de chauffe, 7 répétitions, médian ± σ), un processus
par bras, `enable_prefix_cache=False`, prompt distinct à chaque répétition,
`generate(max_tokens=1)` chronométré entre deux `synchronize`. Compteurs sur
`_forward_prefill_grouped` (fpg), `_forward_grouped` (fg), `_gemm`,
`_pile_bf16` (pile), cumulés sur les 7 répétitions (48 couches × 7 = 336).

| bras | réglage | chemin |
|------|---------|--------|
| (b) GEMM direct | défaut (t > 32) | `_forward_prefill_grouped` → `_gemm` |
| (c) déquant | `ACVRAM_PREFILL_DEQUANT=1` | `_forward_prefill_grouped` → `_pile_bf16` |
| (a) GEMV plein | `ACVRAM_MOE_GROUPED_MAX=99999` | `_forward_grouped` sur L jetons |
| (d) tranches 32 | `ACVRAM_BUDGET_JETONS=32` | runner découpe, `_forward_grouped` × L/32 |
| (e) GEMM forcé | `ACVRAM_MOE_GEMM_MAX=99999` | comme (b), sans repli déquant |

### L=512 (≈ 32 jetons par expert)

| bras | j/s | σ | ms/pas | fpg | fg | gemm | pile | /b |
|------|----:|--:|-------:|----:|---:|-----:|-----:|---:|
| (b) GEMM direct | **3 916** | 22 | 130,7 | 336 | 0 | 1008 | 0 | 1,00 |
| (c) déquant | 3 034 | 20 | 168,8 | 336 | 0 | 0 | 1008 | 0,77 |
| (a) GEMV plein | 1 644 | 6 | 311,4 | 0 | 336 | 0 | 0 | 0,42 |
| (d) tranches 32 | 962 | 83 | 532,1 | 0 | 5376 | 0 | 0 | 0,25 |

### L=2048 (≈ 128 jetons par expert)

| bras | j/s | σ | ms/pas | fpg | fg | gemm | pile | /b |
|------|----:|--:|-------:|----:|---:|-----:|-----:|---:|
| (b) défaut | **7 607** | 25 | 269,2 | 336 | 0 | **0** | **1008** | 1,00 |
| (c) déquant | 7 592 | 202 | 269,8 | 336 | 0 | 0 | 1008 | 1,00 |
| (e) GEMM forcé | 4 860 | 4 | 421,4 | 336 | 0 | 1008 | 0 | 0,64 |
| (a) GEMV plein | 1 661 | 2 | 1232,9 | 0 | 336 | 0 | 0 | 0,22 |
| (d) tranches 32 | 894 | 4 | 2290,2 | 0 | 21504 | 0 | 0 | 0,12 |

### Prédictions écrites avant la mesure, et verdict

- (b) ≈ 6 400 j/s à L=512 : **faux**, 3 916 (la référence 6 420 vient du banc
  serveur, autre dénominateur — lot, TTFT exclu) ; ≈ 8 000 à L=2048 : 7 607, proche.
- c/b ≈ 0,75 : **0,77 à L=512, confirmé**. À L=2048 c = b parce que (b) était
  déjà en déquant (compteurs `gemm=0`) — le seuil `_MOE_GEMM_MAX=64`
  (model.py:737) a basculé le chemin ; sans compteur cette égalité aurait
  été lue comme « noyau noyé ».
- a/b ≈ 0,5-0,8 : **0,42 / 0,22**, plus sévère que prévu.
- d/b ≈ 0,5-0,8 : **0,25 / 0,12**, bien plus sévère : 16 (resp. 64) pas
  runner de 32 jetons, chacun un GEMV par expert.
- Issue gênante (b ≈ c) : **ne s'est pas produite** à 32 jetons/expert.

### Lecture

1. **Le noyau GEMM NVFP4 gagne 29 % à 32 jetons/expert** (L=512) et **perd
   36 % à 128 jetons/expert** (L=2048, bras e). Le seuil `_MOE_GEMM_MAX=64`
   est du bon côté ; le croisement exact est entre 32 et 128, non mesuré.
2. **Le GEMV par expert n'est jamais le bon chemin au-delà de 32 jetons** :
   ×0,42 en pleine longueur, ×0,25 en tranches. Le tranchage
   `ACVRAM_BUDGET_JETONS` coupe la latence des autres séquences, il ne peut
   pas être vendu comme un gain de débit.
3. Le levier réel est `_MOE_GEMM_MAX` (jetons par expert), pas
   `_MOE_GROUPED_MAX` ni le tranchage du runner : mesurer 48/64/96 pour
   placer le croisement.

## Croisement GEMM / déquant en jetons par expert (13/09)

Même protocole, bras (e) GEMM forcé (`ACVRAM_MOE_GEMM_MAX=99999`) contre
(c) déquant, L ∈ {512, 768, 1024, 1536, 2048} soit par_expert = L × 8 / 128.
Compteurs vérifiés à chaque passage (gemm=1008 ou pile=1008, jamais mêlés).

Prédiction écrite avant : GEMM gagne à 32 et 48, croisement 64-96, perd à 128.

| j/expert | L | GEMM j/s | σ | déquant j/s | σ | GEMM/déquant |
|---------:|-----:|---------:|--:|------------:|--:|-------------:|
|  32 |  512 | **3 913** | 22 | 3 022 | 69 | 1,29 |
|  48 |  768 | **4 260** | 40 | 4 173 | 25 | 1,02 |
|  64 | 1024 | 4 511 | 34 | **5 075** | 26 | 0,89 |
|  96 | 1536 | 4 749 |  9 | **6 592** | 27 | 0,72 |
| 128 | 2048 | 4 864 |  5 | **7 592** | 32 | 0,64 |

Verdict : croisement vers **50 jetons/expert**, plus bas que prédit (et que
le commentaire du 3/09 qui le plaçait entre 64 et 128 : « 1024 j : +5 % »
est aujourd'hui −11 %). **Le défaut 64 était du mauvais côté** : à 64
j/expert il coûtait 11 %. Défaut abaissé à 48 (model.py, `_MOE_GEMM_MAX`),
tableau dans le commentaire. À 48 l'écart est +2 % (≈ 2 σ) : le seuil
pourrait être 48 ou 56, pas 64.

Le GEMM plafonne (3 913 → 4 864 de 32 à 128 j/expert, +24 %) là où la
déquant fait ×2,5 : le noyau relit les poids par tuile de 16 jetons, son
coût croît avec t ; la déquant paie un coût fixe puis un `grouped_mm` bf16
qui exploite pleinement les tensor cores. Un noyau GEMM NVFP4 à tuile plus
haute (32 ou 64 jetons) déplacerait le croisement — chantier distinct.

## Question ouverte — le tranchage runner (chef, 13/09)

Le tranchage `ACVRAM_BUDGET_JETONS` coûte ×4 à ×8 en débit de prefill
(bras d). C'est un outil de latence : il rend la main aux slots qui
décodent entre deux tranches. Piste : ne trancher que lorsque d'autres
slots décodent réellement, et précalculer d'une pièce sinon. Non mesuré.

## Conclusion

Le noyau `nvfp4_gemm_grouped` vaut +29 % de débit de prefill à ≈ 32 jetons
par expert (L=512 sur Qwen3-Coder-30B, 8 actifs / 128) et −36 % à ≈ 128
(L=2048) : le repli déquant au-delà de `_MOE_GEMM_MAX=64` est justifié, le
croisement reste à placer entre 32 et 128. Le GEMV par expert (chemin
`_forward_grouped`) est 2,4 à 8× plus lent que le GEMM dès que t > 32 ; le
tranchage du runner en 32 est un outil de latence, pas de débit.

Le noyau NVFP4 reste utile au **décodage** (lots petits, memory-bound), où il
évite de lire 4× plus d'octets. Le banc de décodage MoE est un chantier
distinct.
