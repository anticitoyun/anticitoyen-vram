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

## Le noyau (13/09, plus tard) — `nvfp4_gemm_grouped_mma`, main 6c99fa5

Écrit et mesuré le même jour. Fragments A (jetons quantifiés E2M1 bloc 16
+ UE4M3 par `nvfp4_quant_act`) et B (nos poids tels quels) chargés depuis
la mémoire globale dans le layout de la MMA ; aucune shared, double tampon
de registres sur K ; piles adressées par deux tables `[E]` int64 (contrat
bead pds). Coupé par défaut (`ACVRAM_MOE_MMA=1`) tant que la PPL W4A4
n'est pas sous +1 % (poste2 : +2,58 % sans lissage).

### Équivalence (52 tests, `tests/test_gemm_grouped_mma.py`)

- Quantification des activations **bit-identique** à la référence torch
  sur 4 formes — après une correction : sous `--use_fast_math` la division
  est un `MUFU.RCP` approché qui basculait 5 égalités E2M1 sur 5 376 ;
  `__fdiv_rn` les rend exactes. Le matériel (`F2FP.SATFINITE.E2M1.F32`)
  arrondit les égalités au code pair.
- GEMM contre référence float64 sur les mêmes activations quantifiées :
  45 cas (5 répartitions d'experts × 3 formes × bt 16/32/64), toutes
  valeurs dans la tolérance fp32/bf16, > 90 % bit-identiques à bf16(réf).
- Tables d'adresses : experts dispersés dans deux piles = même sortie ;
  table fausse ≠ sortie.
- Cosinus W4A4 / bf16 : 0,9954 — l'information, pas le verdict.

### Débit (moteur chaud, cache de préfixe coupé, 7 rép, médian ± σ, j/s)

Prédiction écrite avant : bat déquant + `grouped_mm` à tous les points
(≥ 3 913 à 32, ≥ 7 592 à 128). Issue gênante : chargements directs sans
shared → MMA < déquant à 128.

| j/expert | L | GEMM bf16 | déquant | **MMA bt64** | MMA bt32 | MMA / meilleur ancien |
|---------:|-----:|------:|------:|------:|------:|-----:|
|  32 |  512 | 3 913 | 3 022 | **5 888 ± 29** | 5 900 ± 14 | 1,50 |
|  48 |  768 | 4 260 | 4 173 | **7 065 ± 31** | — | 1,66 |
|  64 | 1024 | 4 511 | 5 075 | **7 820 ± 27** | — | 1,54 |
|  96 | 1536 | 4 749 | 6 592 | **8 551 ± 68** | — | 1,30 |
| 128 | 2048 | 4 864 | 7 592 | **8 945 ± 53** | 8 219 ± 58 | 1,18 |

Compteurs : `mma=1008`, `gemm=0`, `pile=0` à chaque point. Prédiction
tenue partout ; l'issue gênante ne s'est pas produite. bt=64 ≥ bt=32
(égal à 32 j/expert, +9 % à 128).

### Lecture

1. Le plafond a sauté : de 32 à 128 j/expert le débit fait +52 % (l'ancien
   noyau : +24 %), et le croisement avec la déquant n'existe plus —
   `_MOE_GEMM_MAX` n'a plus d'objet sur ce chemin.
2. Le pas entier à L=2048 passe de 269,8 ms à 228,9 ms ; la part MoE n'est
   plus le premier poste — mesurer le reste (attention, quantification des
   activations, routage) avant d'optimiser le noyau davantage.
3. Ce que ce chiffre ne dit pas : la qualité. 5 888 j/s à +2,58 % de PPL
   ne se livre pas. Le lissage statique de poste2 décide.

### Ce qui reste

- Chargements directs : un `cp.async` en shared double-tamponné gagnerait
  sur la latence ; à mesurer seulement si le MoE redevient le poste dominant.
- `nvfp4_quant_act` est un lancement séparé par entrée (2 par couche) : à
  fusionner dans l'épilogue de gate·up pour `down_proj`.

## Profil du pas et décodage b=12 (13/09, nuit)

Prédictions écrites avant : attention paginée premier poste (≥ 35 %),
MoE MMA 15-25 %, `quant_act` < 5 %, projections denses 10-20 % ; au
décodage b=12, MMA ≈ GEMV à ±10 %, issue gênante MMA plus lent.

### Profil `torch.profiler`, prefill chaud L=2048, `ACVRAM_MOE_MMA=1`

Pas mesuré 226 ms ; somme des `self_device_time` 289 ms (les entrées
`aten::` recomptent leurs noyaux, lire les lignes de noyaux).

| poste | ms | appels | part du pas |
|-------|---:|------:|-----:|
| `nvfp4_gemm_grouped_mma_kernel<64>` | **114,4** | 144 | **≈ 50 %** |
| routage + index/gather/copies/mul/sum (glue Python du MoE) | ≈ 55 | ~1 500 | ≈ 25 % |
| GEMM bf16 cutlass (projections attention, INT8 promu) | 19,1 | 192 | 8 % |
| `int8_dequant_kernel` | 11,0 | 192 | 5 % |
| flash attention | 11,5 | 48 | 5 % |
| `nvfp4_quant_act_kernel` | 1,6 | 96 | 0,7 % |

- **Prédiction attention fausse** : 5 %, pas 35 %. À L=2048 sur ce MoE le
  pas est le MoE, pas l'attention.
- **`quant_act` = 0,7 %** : la fusion dans l'épilogue gate·up n'a pas
  d'objet, retirée du plan.
- **Le noyau MMA est encore le premier poste, et loin de sa borne** :
  poids MoE par couche ≈ 300 Mo en 4 bits, lus 2× à 128 j/expert avec
  bt=64 → 28 ms sur 48 couches à 1 050 Go/s ; calcul 1,5·10¹³ FLOP →
  7 ms à 2 000 TFLOPS. Mesuré 114 ms : **~25 % de la borne mémoire**.
  Cause : fragments chargés depuis la mémoire globale un pas de K en
  avance seulement, aucun `cp.async`, un bloc de 128 fils par 64 lignes.
  C'est la retouche suivante : étage en shared par `cp.async` à 3-4
  étapes, BM=128 pour réutiliser A, 8 warps.
- La glue Python (25 %) est le second poste : `x[flat_t[ordre]]`,
  `d[inv]`, `.to(float32)`, `sum(dim=1)` — un noyau de dispersion /
  réduction pondérée par expert la ferait disparaître.

### Décodage MoE b=12 (Qwen3-Coder-30B, 62 pas de 12 séquences, 5 rép, médian)

| chemin | ms / pas | j/s | compteurs |
|--------|---------:|----:|-----------|
| GEMV par expert (défaut, `_forward_grouped`) | **19,51** | **615** | fg=2976 |
| MMA (`ACVRAM_MOE_GROUPED_MAX=0`, `ACVRAM_MOE_MMA=1`, bt=16) | 39,61 | 303 | fpg=2976, mma=8928 |

**MMA ×0,49 au décodage** : l'issue gênante. 96 affectations sur 128
experts → une tuile de 16 jetons par expert en tient ~1 ; les octets de
poids lus sont les mêmes qu'en GEMV, mais le GEMV fusionné gate·up tient
634 Go/s effectifs quand le noyau MMA plafonne à ~25 % de la borne. Le
GEMV reste le chemin du décodage ; la MMA n'y a d'intérêt qu'une fois son
étage mémoire refait — et même alors le gain attendu est celui de la
lecture, pas du calcul.

## Glue du prefill MoE en deux noyaux (13/09, nuit) — `moe_act`, `moe_reduce_trie`

Prédiction écrite avant : MMA L=2048 +15-20 %, déquant L=2048 +14 %,
GEMM directe L=512 +10 % ; issue gênante < +5 %.

| chemin | L | glue torch | **glue noyaux** | gain |
|--------|--:|----------:|----------:|----:|
| MMA bt64 | 2048 | 9 000 ± 65 | **10 535 ± 94** | +17 % |
| déquant + `grouped_mm` | 2048 | 7 559 ± 33 | **8 615 ± 40** | +14 % |
| GEMM directe (wmma) | 512 | 3 879 ± 16 | **4 003 ± 10** | +3 % |

(j/s, moteur chaud, cache de préfixe coupé, 7 rép ; compteurs inchangés.)

Deux des trois prédictions tenues ; à L=512 la glue ne pèse que ~3 % du
pas (le GEMM y domine), prédiction +10 % trop haute. Le pas MMA à L=2048
passe de 227,6 à 194,4 ms : la glue restante (index/gather de `xs`,
construction des tuiles ≈ 10 lancements par couche) et la quantification
sont maintenant sous 10 %.

Équivalence (`tests/test_moe_glue.py`, 87 passed avec les tests GEMM) :
`moe_act` SiLU identique à ≥ 99 % en bf16 ; GELU-tanh a demandé de
reproduire la suite d'opérations fp32 de `F.gelu` avec un `tanh` en
double — `tanhf` sous `--use_fast_math` est l'approximation MUFU et 0,2 %
des sorties différaient de 1-2 ulp ; en double avec la même formule
qu'en fp32, 93,7 % identiques seulement (l'arrondi des intermédiaires) ;
formule fp32 de torch + `tanh` double : ≥ 99 %. `moe_reduce_trie`
≥ 98 % identique (ordre de somme de torch).

État du pas L=2048 (MMA) : 194 ms dont noyau MMA ≈ 114 — il pèse
maintenant ≈ 60 %, et il est à 25 % de sa borne. C'est la retouche (a),
`cp.async`, si la qualité W4A4 passe.

## Étage `cp.async` du noyau MMA (14/09) — `nvfp4_gemm_grouped_mma2_kernel<BT, S>`

BM=128 lignes, 8 warps, tuiles A/B/échelles en mémoire partagée à foulée
48 o (32 utiles + 16 : les 32 lectures de 4 o d'un warp tombent sur 32
bancs distincts), S étapes en pipeline (`cp.async.cg` 16 o pour les
données, `.ca` 4 o pour les échelles), même ordre de somme que la variante
directe. Sélection par l'argument `etages` (0 = directe, 2-4), réglage
`ACVRAM_MOE_MMA_ETAGES`.

Prédiction écrite avant : L=2048 10 535 → 14-16 000 j/s (noyau 114 →
40-55 ms, ×2-3) ; issue gênante < +20 %.

| variante | L | j/s | σ | ms/pas |
|----------|--:|----:|--:|------:|
| directe (étages 0) | 2048 | 10 411 | 98 | 196,7 |
| étages 2 | 2048 | 16 655 | 24 | 123,0 |
| étages 3 | 2048 | 16 851 | 32 | 121,5 |
| **étages 4** | 2048 | **16 938** | 41 | **120,9** |
| directe | 512 | 5 888 | 29 | 87,0 |
| **étages 4** | 512 | **8 701** | 49 | **58,8** |

Compteurs mma=1008 partout. Équivalence : 128 tests, et **bit-identique**
à la variante directe pour bt ∈ {16, 32, 64} × étages ∈ {2, 3, 4}
(`test_bt32_et_etages_identiques`). Suite complète : 774 passed.

Lecture : le pas L=2048 perd 76 ms ; le noyau passe de ≈ 114 à ≈ 38 ms
(×3, la borne mémoire estimée est 28 ms → il est maintenant à ~75 % de sa
borne). ×1,63 sur le pas entier, ×1,48 à L=512. Deux étapes suffisent
presque (−1,7 %) : la latence est couverte dès qu'une tuile est en vol
pendant le calcul de la précédente ; 4 est retenu par défaut.

Bilan du chantier 2 sur le prefill chaud L=2048 de Coder-30B, en j/s :
déquant + `grouped_mm` 7 592 → GEMM directe wmma (ancienne) 4 864 → MMA
FP4 native 8 945 → + glue en noyaux 10 535 → + pipeline cp.async
**16 938**, soit ×2,23 sur le meilleur chemin d'hier. Qualité : A4 sur les
experts mesurée par poste2 sur le noyau réel : PPL +0,919 % (seuil 1 %),
`ACVRAM_MOE_MMA=1` par défaut (poste2).

## Face à vLLM (14/09, soir) — décodage b=12, profil, tuile 128

Contexte : poste2 a mesuré vLLM 0.29 sur la même carte, Coder-30B NVFP4 :
décodage 12 séquences 1 198 t/s (10 ms/pas), prefill pp2048 34 788 j/s,
GEMM groupée CUTLASS FP4 14,3 ms par pas contre nos 38 ms.

### (1) Décodage b=12 : MMA2 sur le chemin décodage

`outils/banc_decodage_moe.py`, Coder-30B, 62 pas, 5 rép, eager :

| chemin | ms/pas | j/s | compteurs |
|--------|-------:|----:|-----------|
| GEMV par expert (défaut) | 19,17 | 626 | fg=2976 |
| MMA2 forcée (GROUPED_MAX=0, bt 16, S=4) | 37,68 | 319 | fpg=2976, mma=8928 |

**×0,51** (×0,49 avant cp.async) : le pipeline n'y change rien. À ~1
jeton par tuile de 16, ce n'est pas le noyau qui coûte mais le chemin
(quant_act + 3 GEMM + glue par couche) contre un GEMV gate·up fusionné.
Verdict tenu : GEMV au décodage.

### (2) Profil du pas de décodage b=12 (chemin par défaut, eager)

GPU 12,25 ms par pas ; sous graphes CUDA le pas mesuré est **16,54 ms
(726 j/s)** — 4,3 ms par pas hors rejeu (ordonnanceur, échantillonnage,
`bind`/`fill`), 26 % du pas.

| poste | ms | part GPU |
|-------|---:|----:|
| `nvfp4_gemv_grouped_gateup` (MoE gate·up) | 3,35 | 27 % |
| `nvfp4_gemv_grouped_warp` (MoE down) | 2,63 | 21 % |
| `int8_gemv_kernel` (projections d'attention, ×192) | 2,99 | 24 % |
| `paged_attn_partial` | 0,72 | 6 % |
| `int8_gemv` lm_head | 0,88 | 7 % |
| routage, élémentaires, reste | ~1,7 | 14 % |

Lecture : le MoE (6 ms) est proche de sa borne (≈ 68 experts distincts ×
2,36 Mo × 48 couches ≈ 7,7 Go → 7,3 ms si tous distincts, moins en
pratique). **Le GEMV INT8 des projections d'attention est à ~27 % de sa
borne** (0,86 Go → 0,8 ms, mesuré 3,0) : premier levier noyau au décodage
(−2 ms, −16 % du GPU). Le second est hors GPU : 4,3 ms par pas d'ordonnan-
cement, à profiler côté CPU. vLLM fait le pas entier en 10 ms.

### (3) Tuile de 128 jetons (bead 0si) — RÉFUTÉ

Prédiction scellée : +13 à +18 % à L=2048 (les poids lus une fois au lieu
de deux) ; réfutation : < +5 %.

| bt | L=2048 | L=512 |
|---:|-------:|------:|
| 64 (S=4) | 17 009 ± 75 | 8 701 ± 49 |
| 128 (S=4) | 16 989 ± 51 | 8 111 ± 49 |

**±0 % à L=2048, −7 % à L=512** (l'issue gênante : 1 bloc de 8 warps par
SM). Réfuté : la seconde lecture des poids de bt=64 était déjà servie par
le L2 (deux tuiles du même expert se suivent, 300 Mo par couche pour 96 Mo
de L2 mais lues de près). La borne à 28 ms de la note précédente était
donc fausse : la vraie est 15,4 Go → 14,7 ms, et le noyau est à **39 %**
de sa borne, pas 75 %. CUTLASS (14,3 ms) est à la borne. Ce qui reste
entre nous et lui n'est pas dans les octets : `Shape<128,128,128>` avec
**K = 128 par étape** (moitié de synchronisations), **TMA** (`cp.async.bulk
.tensor`, un fil émet le chargement d'une tuile entière) et **warps
producteurs/consommateurs** ; nos `cp.async` de 16 o par fil et nos
`__syncthreads` par pas de 64 sont la différence. Bead 0si à réécrire :
TMA + K=128, prédiction sur le 39 %.

## INT8 GEMV des projections d'attention, NV=12 (14/09, soir)

Le profil b=12 montrait `int8_gemv<4,8>` + `<4,4>` par couche : NV
plafonnait à 8, les poids étaient relus. Version à NV ≤ 12 (activations
chargées par mot de 4 poids, même ordre d'accumulation ; 243 registres
bf16, pas de débordement), tests bit-identiques à N=1 pour N=1..16.

Prédiction scellée : `int8_gemv` 2,99 → ~1,7 ms par pas (−1,3 ms) ;
réfutation : < −0,5 ms.

| mesure | avant | après |
|--------|------:|------:|
| `int8_gemv` projections, profil eager (GPU) | 2,99 ms (<4,8>+<4,4>) | **2,62 ms** (<4,12>) |
| GPU du pas, eager | 12,25 ms | 11,94 ms |
| pas sous graphes, banc b=12 | 16,54 ms (726 j/s) | **14,54 ms (826 j/s)** |

**Réfuté sur le noyau** (−0,37 ms < −0,5) : la seconde lecture des poids
était servie par le L2 — la même leçon que la tuile de 128 le même jour,
deux fois en une soirée : « lu deux fois » ne coûte que si les deux
lectures sortent de la HBM. Les 2 ms gagnés sur le pas sous graphes ne
sont PAS attribuables à ce changement sans A/B dans les mêmes conditions
(le 16,54 datait d'une autre session de mesure, après le duel de poste2) ;
ils sont rapportés, pas revendiqués.

Pourquoi le noyau reste à ~31 % de sa borne (0,86 Go en 2,62 ms) : à
K=2048, `threads_for(K)` donne 128 fils pour `nloads = 128` — **chaque fil
fait exactement un chargement de 16 o par ligne** puis réduit : aucune
latence recouverte, le bloc attend sa seule rafale. Le levier est là
(plusieurs chargements en vol par fil, moins de fils par ligne ou plus de
lignes par bloc), à mesurer par A/B propre. Bead à ouvrir si chef le
retient devant 0si.

### INT8 GEMV « un warp par ligne » (bead z5q) — RÉFUTÉ deux fois

Prédiction : 2,62 → 1,0-1,4 ms ; réfutation > 2,0 ms.

| variante | `int8_gemv` par pas (profil eager b=12) |
|----------|----:|
| blocs 4 lignes, NV=12 (défaut) | 2,62 ms (q/k/v ≈ 1,29 + o ≈ 1,33) |
| warp, 1 ligne/warp, x en shared, 4 uint4 en vol | **2,79 ms** |
| warp, 4 lignes/warp (32/bloc), K ≤ 2048 (q/k/v seuls) | **1,69 ms** sur q/k/v (vs ≈ 1,29) + 1,33 o |

Ce que le compte a montré, après coup : à N = 12 et K = 2048, une ligne
lit **2 Ko de poids et 48 Ko d'activations** (12 × 2048 bf16) — le
« GEMV » est un GEMM étroit dont le trafic dominant est x, pas W. Copier x
en shared par bloc (48 Ko × 160-640 blocs = 8-31 Mo) coûte plus que ce
que la réduction par shuffles rapporte ; le noyau à blocs lit x depuis le
L1 (hits) et ses 12 réductions par la shared ne sont pas le goulot. Ce qui
reste est le plancher d'un lancement de ~9 Mo (rampe, queue, ~10 µs sur
27) : le levier serait de lancer moins (q/k/v sont déjà fusionnés ; o ne
peut pas l'être avec eux). Variante gardée derrière
`ACVRAM_INT8_GEMV_WARP=1`, défaut = noyau à blocs. Bead z5q fermé.

## K = 128 par étage (bead 0si, première étape) — 14/09 nuit

Prédiction scellée : +10-20 % sur le noyau seul ; réfutation < +5 %.
Même pipeline cp.async, foulée de ligne 80 o (20g+tq : 32 bancs distincts),
deux MMA par fragment et par étage, échelles en 8 o, moitié de
`__syncthreads`. Bit-identique à la variante directe (166 tests, bt ×
étages × ks).

| ks | L=2048 | ms/pas | L=512 | ms/pas |
|---:|-------:|-------:|------:|-------:|
| 64 | 16 911 ± 36 | 121,1 | 8 761 ± 42 | 58,4 |
| **128** | **19 148 ± 23** | **107,0** | **9 853 ± 44** | **52,0** |

**+13 % en j/s aux deux longueurs** ; sur le pas, −14 ms à L=2048 : le
noyau passe d'environ 38 à ≈ 24 ms (−37 %, au-delà de la prédiction), soit
~60 % de la vraie borne (14,7 ms) contre 39 % avant. `ACVRAM_MOE_MMA_KS=128`
par défaut. Bilan chantier 2 à L=2048 : 7 592 → 19 148 j/s (×2,52) ;
vLLM pp2048 est à 34 788, le noyau CUTLASS à 14,3 ms. Reste dans 0si : TMA
(`cp.async.bulk.tensor`, un fil par tuile) et warps producteurs/
consommateurs pour les derniers 10 ms.
