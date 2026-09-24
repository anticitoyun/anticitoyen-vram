# Pièce 126 bis — linéaires nvfp4 à petit M : NInfer contre acvram, à sec, aucun code — 24/09 02 h 4x (poste1)

Source NInfer : clone local `/mnt/AI_GENERATOR/ninfer`, commit 594930e (Apache-2.0). Acvram : main fb82bd14.
Cellule à expliquer (registre 02 h 24, poste2, 94e08f8d) : Qwen3.8-27B-nvfp4, b=8 → NInfer +56,3 % de débit, −35,2 % de J/jeton ; b=1 → égalité (+1,55 %).

## 1. Le fait qui domine : NInfer ne calcule pas la même arithmétique à b=8
* L'artefact mesuré (`models/qwen3_8_27b_nvfp4.ninfer`) vient de la recette officielle `tools/convert/official_recipes.py:150-172` :
  MLP des couches < 56 en **nvfp4 avec `activation_policy="AllowA4"`** ; toutes les autres projections (attention, GDN,
  MLP 56-63) en **FP8 avec `AllowA8`**. Comptage dans l'artefact : 168 usages `AllowA4` (= 56 couches × 3 projections
  MLP), 347 `AllowA8`, 329 `A16Only`.
* `AllowA4` = **W4A4** : l'activation est quantifiée à la volée en E2M1, par bloc de 16 (`nvfp4_codec.cuh:35-52`,
  `cvt.rn.satfinite.e2m1x2.f32`), puis produit par la MMA FP4 native
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X` (`src/ops/common/mma.cuh:89`, noyau
  `nvfp4_w4a4_mma.cuh:208`).
* Seuils A4 par forme : gate‖up `n34816_k5120.cu:49` A4 **toujours**, mais la SwiGLU fusionnée prend la route A16 à
  T = 1 (`linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:37`, DecodeFusedA16), A16 à T ≤ 4 (`:38`), **W4A4
  fusionnée de T = 5 à 128** (`:39`) ; down `n5120_k17408.cu:68` A4 **dès T ≥ 8**.
* D'où la forme de la cellule : **à b=1, A16 des deux côtés → égalité ; à b=8, NInfer passe en W4A4 sur tout le MLP
  → +56 %**. Acvram à b=8 reste **W4A16** (activations bf16). Le comparatif de la 102 oppose donc deux formats
  numériques, pas seulement deux jeux de noyaux. Il doit le dire (REGLES § 3 : « une cellule du comparatif se mesure
  avec le harnais des concurrents », et même format).

## 2. Comparaison noyau par noyau (b=8, M = T = 8)

| point | NInfer | acvram (défaut) |
|---|---|---|
| aiguillage | par FORME (N, K) figée : `nvfp4_dispatch.cpp:9-17`, tables `shapes/*.cu` (formes inconnues → exception) | générique : `acvram/kernels/__init__.py:591` (n ≤ 32 = `_NVFP4_GEMV_MAX`, :537) |
| MLP gate‖up | W4A4 MMA FP4 + SwiGLU fusionnée (`nvfp4_linear_swiglu_w4a4.cu`), tuile T32R128 (`n34816_k5120.cu:26,43`) | W4A16 : `gemm_dense_etroit` Triton (`__init__.py:606`, `ACVRAM_DENSE_NVFP4=triton` :945) ; activation SwiGLU à part |
| MLP down | W4A4 MMA, tuile T32R64 dès T ≥ 8 (`n5120_k17408.cu:37,60,68`) | idem gate‖up (`gemm_dense_etroit`) |
| instruction | **MMA FP4 native** `kind::mxf4nvf4` m16n8k64 : aucune déquantification des poids | **MMA bf16** (`tl.dot`, `gemm_dense_etroit.py:93`) après décodage E2M1 → bf16 **par table en registres** (lut4/lut8, `:54`, en-tête :8-10) ; M = 8 rembourré en BM = 16 (moitié de la tuile MMA perdue) |
| décodage des poids | aucun en W4A4 ; en A16 : conversion matérielle `__nv_fp4x2_e2m1` (`nvfp4_codec.cuh:14-15`), SIMT, ordonnancements EXACTS par T (`n5120_k17408.cu:9-12,46`) | table constante E2M1 → bf16 + produit par l'échelle E4M3, par élément (`gemm_dense_etroit.py` en-tête) |
| split-K | **aucun** dans `nvfp4_w4a4_mma.cuh` : down N = 5 120 en tuiles de 64 → 80 CTA pour 170 SM | **oui** : tranches K pour couvrir la carte (`gemm_dense_etroit.py:205-215`), partiels fp32 + réduction (compteur atomique :103-111) |
| fusion | gate‖up + SwiGLU en un noyau ; quantification A4 de l'entrée à chaque linéaire (espace de travail `nvfp4_w4a4_plan.h:36-53`) | gate, up, activation en appels séparés (dense) |
| attention / GDN | FP8 **W8A8** (`AllowA8`) — 1 o/paramètre | nvfp4 W4A16 (0,5625 o/paramètre) — hypothèse à vérifier sur le manifeste de `Qwen3.8-27B-nvfp4` |
| W4A4 chez nous | — | la MMA existe déjà : `acvram_kernels.cu:3390-3395` (même `mxf4nvf4`), `fp4_gemm.py:162` `nvfp4_mm_tensorcore`, servie seulement en PRÉFILL et en opt-in (`ACVRAM_PREFILL=w4a4`, `__init__.py:653-657`) |

## 3. Octets lus par pas (b=8, poids seulement, à confirmer au manifeste — REGLES § 3 : jamais un octet de mémoire)
* MLP d'une couche (H = 5 120, I = 17 408) : 3 × 5 120 × 17 408 = 267,4 M paramètres ; nvfp4 (4 bits + E4M3 / 16)
  = 0,5625 o → **150,4 Mo** ; FP8 → ≈ 267 Mo.
* NInfer : 56 × 150,4 + 8 × 267 ≈ **10,6 Go** de MLP + attention FP8. Acvram (si tout nvfp4) : 64 × 150,4 ≈
  **9,6 Go** + attention nvfp4.
* **NInfer lit donc AUTANT ou PLUS d'octets que nous, et va 1,56 × plus vite** : à b=8, notre pas n'est pas borné par
  les octets. Le coût est dans le calcul autour des octets (décodage E2M1 par table, MMA bf16 à moitié vide). C'est
  exactement ce que W4A4 supprime, et c'est aussi ce qui explique les −35 % de J (moins d'ALU par octet).

## 4. Prédiction d'un « port », et ce qui la réfuterait
Le port utile n'est pas de copier leurs noyaux, mais d'ouvrir notre MMA FP4 existante (§ 2, dernière ligne) aux
petits M, avec une tuile T ≤ 32 et la SwiGLU fusionnée. **Ça change la sortie** (activation E2M1) : c'est un régime
nommé, OPT-IN, jugé par la KL et la PPL, jamais un défaut sans porte de qualité (REGLES § 1, « rapide ± 1 ulp », qui
ne couvre même pas ce cas : W4A4, c'est plus qu'un ulp).
* **Prédit** (le MLP porte l'essentiel des octets ; le décodage par table borne `gemm_dense_etroit` à 1,1-1,4 To/s,
  mesuré au 17/09) : à b=8, **débit +25 à +40 %**, **J/jeton −15 à −25 %**. Pas les +56 % entiers : l'attention
  W8A8 et la fusion en portent une part.
* **Qualité** : NInfer protège lui-même les 8 dernières couches du W4A4 (recette `:165`, `layer < 56`) — indice que
  le W4A4 coûte en qualité sur ce modèle. Prédit : KL b=8 au-dessus de celle du W4A16, à mesurer contre le témoin.
* **Réfuté si** (dans cet ordre, du moins cher au plus cher) :
  1. la **126 de poste3** (nsys b=8 par famille) montre le MLP dense à moins de 50 % du pas, ou `gemm_dense_etroit`
     déjà ≥ 1,5 To/s → l'écart est ailleurs (attention, hôte) ;
  2. une **cellule NInfer `A16Only`** (même artefact reconverti, leurs outils le permettent, `tools/convert`) rend
     b=8 ≈ acvram → tout l'écart vient du format, et notre noyau A16 n'est pas en cause ;
     si NInfer A16Only reste ≫ acvram, leurs ordonnancements SIMT exacts (`nvfp4_simt.cuh`, conversion
     matérielle) sont le levier, pas le W4A4 ;
  3. un micro-banc à sec de notre `nvfp4_mm_tensorcore` à M = 8 sur les formes de Qwen3.8 ne bat pas
     `gemm_dense_etroit` d'au moins 1,3 × ;
  4. la KL du W4A4 sur le MLP dépasse la porte (0,74 à b=1, relative au lot mêlé) → régime non servable, quel que
     soit son débit.

## Ordre proposé (au chef)
(a) Requalifier la cellule 102 b=8 : « NInfer W4A4 MLP + W8A8 attention contre acvram W4A16 ». (b) Cellule NInfer
A16Only (poste2), c'est le réfutateur n° 2, le moins cher des mesures neuves. (c) Selon (b) et la 126 : micro-banc W4A4
à petit M, puis opt-in sous scellé qualité.
