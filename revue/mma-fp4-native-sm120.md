# La RTX 5090 a une MMA FP4 native — et notre GEMM groupée ne l'utilise pas

Date : 13/09/2026 — poste4, chantier 2. Test : `outils/test_mxf4_sm120.cu`.
Origine : campagne duck.ai (`acvram-memoire/revue/duck-poste4-13-09.md`),
Luna et GPT-5.4 mini contre Haiku ; tranché par exécution.

## Prédiction écrite avant

- Compile avec `-arch=sm_120a` ou échoue à ptxas (issue : « mxf4nvf4
  réservé aux sm_100 », Haiku).
- Si compile : un MMA 16×8×64 avec échelles E4M3 non-puissances de 2 est
  bit-identique à la référence CPU en fp32, sinon mon layout de fragments
  est faux (écart > 0 sur des éléments précis).
- Débit : entre 4× et 8× le bf16 m16n8k16 sur la même boucle (fiche
  Blackwell : FP4 dense 1 676 TFLOPS contre bf16 210 → 8×). Issue qui me
  gênerait : < 2× (instruction émulée), le chantier tombe.

## Résultat

```
$ nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o test_mxf4 outils/test_mxf4_sm120.cu
$ cuobjdump --dump-sass test_mxf4 | grep -oE "[A-Z]+MMA[.A-Z0-9_]*" | sort | uniq -c
    132 HMMA.16816.F32.BF16
    133 OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
$ CUDA_VISIBLE_DEVICES=0 outils/carte.sh ./test_mxf4
EXACTITUDE elements_bit_identiques=128/128 ecart_max_abs=0.000e+00 max_ref=542.750
BANC mxf4nvf4 m16n8k64 : 170 SM, 1360 blocs x 128 fils, 4000 iters x 4 chaines : 0.692 ms, 2059.6 TFLOPS
BANC bf16 m16n8k16    : 170 SM, 1360 blocs x 128 fils, 4000 iters x 4 chaines : 1.371 ms, 260.1 TFLOPS
```

- **`-arch=sm_120a` seul ne suffit pas** : nvcc 13.4 émet `.target sm_120`
  et ptxas refuse (« Instruction 'mma with block scale' not supported on
  .target 'sm_120' »). Il faut `-gencode arch=compute_120a,code=sm_120a`.
  Conséquence pour `kernels/__init__.py` : le JIT devra passer ce gencode
  pour ce noyau, et le `.so` ne chargera que sur sm_120 (la 3080 Ti garde
  son chemin AWQ — c'est déjà le cas).
- L'instruction SASS est `OMMA.SF.16864` : un MMA distinct de HMMA, avec
  facteurs d'échelle (« SF ») en opérande. Ce n'est pas une émulation.
- **128/128 bit-identiques** avec des échelles 1, 2, 0,5, 1,25, 2,5, 0,25 :
  le layout des fragments (décodé depuis `mma_traits_sm120.hpp` de CUTLASS)
  est juste ; le calcul interne est bien E2M1 × UE4M3 par bloc de 16 en
  accumulation f32.
- **×7,9 en débit de registre** (2 060 contre 260 TFLOPS). Les valeurs
  absolues dépassent la fiche (1 676 / 210 à 2 407 MHz) parce que la carte
  tourne vers 2,9 GHz ; le rapport 7,9 ≈ 8 est celui de la fiche.

## Dénominateurs

- Micro-banc en registres : aucune lecture mémoire, 4 chaînes de
  dépendance par warp, 8 blocs de 4 warps par SM. Il mesure le **pic de
  l'unité**, pas un GEMM. Un GEMM réel restera borné par la lecture des
  poids (1 050 Go/s) : à 4,5 bits par poids, 1 050 Go/s nourrissent
  ~1,9 T poids/s ; à 32 jetons par expert cela vaut ~120 TFLOPS utiles —
  le MMA n'est plus le goulot, la mémoire l'est. C'est exactement ce qu'on
  veut : aujourd'hui la déquant en shared + wmma bf16 (260 TFLOPS pic,
  moins la conversion) est le goulot dès 48 jetons/expert.
- Le test ne dit rien de la qualité : l'instruction exige **A en E2M1**
  aussi. Voir ci-dessous.

## Ce que ça change pour `nvfp4_gemm_grouped_kernel` (acvram_kernels.cu:1599)

Aujourd'hui : W4A16 logiciel — poids E2M1 → bf16 en shared (`ws`),
activations bf16 (`xs`), wmma bf16 16×16×16, tuile 64 lignes × 16 jetons.
Le plafond (+24 % de 32 à 128 j/expert quand la déquant + `grouped_mm`
fait ×2,5) vient de là : chaque tuile de 16 jetons redéquantise 64×K
poids en shared.

Avec `OMMA.SF` : **W4A4** — poids E2M1 lus tels quels (aucune conversion,
4× moins de shared par tuile de poids), échelles E4M3 telles quelles (nos
`bscale` par bloc de 16 sont déjà au format UE4M3 attendu, à vérifier :
signe absent, biais 7), super-échelle fp32 appliquée à la sortie comme
aujourd'hui. **Les activations doivent être quantifiées en E2M1 par bloc
de 16 avec une échelle E4M3** — à la volée, au chargement de `xs`
(amax du bloc / 6 → E4M3, puis arrondi E2M1). C'est ce que font vLLM et
TensorRT-LLM pour les checkpoints « NVFP4 » (W4A4).

Deux voies :

1. **W4A4 pur** : A quantifié en E2M1 blocs de 16. Débit maximal, risque
   qualité (les activations ont des valeurs aberrantes ; la fiche NVIDIA
   annonce < 1 % de perte sur les tâches usuelles, à mesurer chez nous).
2. **W4A16 sur MMA FP4** : impossible — l'instruction n'a pas de variante
   A en bf16. La voie « mxf8f6f4 » (m16n8k32, A en E4M3 8 bits, B en E2M1)
   existe sur sm_120 (`SM120_16x8x32_TN_VS`) : **W4A8**, activations en
   FP8 E4M3 par bloc de 32 — compromis qualité/débit (4× bf16 au lieu de
   8×). À garder comme repli si W4A4 dégrade la PPL.

## La mesure qui tranche

1. **Qualité** (sans GPU pour la référence) : PPL Llama-2-7B protocole
   GPTQ (étalon fp16 5,4141, tout-nvfp4 5,6102) avec activations des MLP
   quantifiées E2M1 bloc 16 (simulation torch, fake-quant) — seuil :
   ≤ +1 % relatif sur 5,6102 (≤ 5,666). Idem W4A8 E4M3 bloc 32. Si W4A4
   dépasse, W4A8 ; si W4A8 dépasse, le chantier s'arrête et on garde la
   déquant.
2. **Débit** : même balayage que `banc-prefill-moe-12-09.md`, par_expert
   32/48/64/96/128, 7 rép, compteurs — le noyau doit battre la déquant +
   `grouped_mm` **à tous les points** (≥ 7 592 j/s à 128, ≥ 3 913 à 32),
   sinon `_MOE_GEMM_MAX` reste et on documente le croisement.
3. **Équivalence** : test `tests/kernels/test_gemm_grouped.py` étendu —
   sortie du nouveau noyau contre déquant bf16 + matmul avec A fake-quant
   E2M1 : cosinus > 0,999 et erreur max bornée, sur 15 formes.

## Bead

`anticitoyen-vram-brd` réécrit : le plan « tuile 32/64 » est retiré, il
optimisait un chemin de conversion qui n'a pas lieu d'être.
