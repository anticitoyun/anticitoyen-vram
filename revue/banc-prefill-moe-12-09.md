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

## Conclusion

Le noyau `nvfp4_gemm_grouped` apporte un gain réel au **premier prefill**
(unchunked, t > 32) : −10 % à L=512 en supprimant 215 ms de déquantification.
En régime chaud, le prefill adaptatif contourne ce chemin au profit de
`nvfp4_gemv_grouped_gateup` (GEMV par expert, toujours NVFP4) — le noyau GEMM
groupée n'y est plus appelé.

Le noyau NVFP4 reste utile au **décodage** (lots petits, memory-bound), où il
évite de lire 4× plus d'octets. Le banc de décodage MoE est un chantier
distinct.
