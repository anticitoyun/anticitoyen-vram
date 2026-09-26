# P1 — étalon FlashInfer b12x (SM120 W4A4 fused MoE) sur nos formes (15/09, poste4)

Ordre de poste7 (`poste7-lancement-14-09.md` § 3) : le banc `bench_b12x_mxfp4_moe.py`
de flashinfer 0.6.18.post1 (`/opt/ia/flashinfer`, venv séparé : torch
2.14+cu130, cutlass-dsl 4.7.1) sur la forme Coder-30B (2048/768/128/top-8)
et GLM-4.7-Flash (2048/1536/64/top-4), 1/8/12/16 jetons, 20 s, sous
carte.sh, compteur NVML.

## Ce que le banc d'origine mesure — deux écarts corrigés dans la copie

`outils/banc_flashinfer_b12x.py` (copie Apache-2.0 avec ajouts) :

1. **Routage** : l'original route `(t·k + j) % E` → à 12 jetons top-8, 96
   experts DISTINCTS (255 Mo de poids par couche Coder) là où le modèle en
   touche ~30 (`bras experts`, 14/09 : 30,7 par couche, 80 Mo). Le seuil
   de poste7 (80-100 µs/couche à 12 jetons) suppose ~30 experts ; sous le
   routage d'origine il est inatteignable (255 Mo à 1 To/s = 255 µs).
   `--experts-pool 30` (Coder) / `--experts-pool 34` (GLM, calcul de poste7)
   : chaque jeton tire top-k experts distincts dans les N premiers.
2. **L2** : `cold_l2_cache=False` en dur → 80 Mo par couche tiennent dans
   les 96 Mo de L2, la boucle mesure une relecture L2 (P7). `--cold-l2`
   purge entre deux mesures ; les deux sont publiés, le froid est le
   chiffre du pas (48 couches différentes par pas).

Plus l'énergie : `energie.py` (compteur) sur la fenêtre `repeat-ms`, repos
10 s avant, W brut/net et horloge par ligne.

## Prédictions scellées (avant la carte)

| forme, jetons, routage | poste7 | moi (L2 froid) | moi (L2 chaud, banc d'origine) |
|---|---|---|---|
| Coder, 12, pool 30 (80 Mo) | 80-100 µs (réfuté > 110 ou < 70) | 85-110 µs | 40-60 µs (le L2 sert) |
| Coder, 12, 96 distincts (255 Mo) | — | 250-300 µs | 220-280 (ne tient pas en L2) |
| Coder, 1, 8 experts (21 Mo) | 30-40 µs | 30-45 µs (plancher de lancement) | 20-30 |
| GLM, 12, pool 34 (160 Mo) | 170-210 µs | 165-220 µs | 150-200 (ne tient pas) |
| W pendant la boucle | — | ≤ 345 W brut (pas de plafond : 0,18 inst/oct) | — |

Ce que ça achète : notre chemin MMA décodage (B) fait le MoE de Coder en
≈ 3,6 ms + glue par pas = **75 µs de GEMM par couche à 12 jetons** (3
lancements + 2 quant_act + moe_act + reduce ≈ 100-110 µs tout compris). Si
b12x fait 80-100 µs tout compris (une seule fusion), le levier de la
fusion vaut ≤ 10-30 µs/couche = 0,5-1,4 ms/pas ; si b12x est à 40-60 µs
à L2 froid, notre noyau a un défaut de 2× à chercher.

## Mesures (15/09, 13h17-13h30, carte exclusive)

En-tête (REGLES §3) : 5090 seule (`CUDA_VISIBLE_DEVICES=0` posé par carte.sh,
`[]` hors verrou), plafond 400 W (relevé avant/après), horloge libre (3 135
max), 35 °C au départ, `nvidia-smi --query-compute-apps` vide avant et
après, venv séparé `/opt/ia/flashinfer/.venv` (torch 2.14+cu130, flashinfer
0.6.18.post1, cutlass-dsl 4.7.1), backend `static` (choisi par
`select_sm120_moe_backend`), NVFP4 (une ligne MXFP4 témoin), fenêtres
20 s + 2 s de chauffe, repos 8 s au compteur avant chaque ligne (72-82 W,
carte chaude), médiane des rejeux de graphe.

**Le `--cold-l2` de FlashInfer est inopérant ici** (chaud = froid = 31,7 µs
au bit) : sa rotation copie les ARGUMENTS de `fn` (tenseurs passés), or
`run()` les capture par fermeture, et elle compte les octets alloués
(604 Mo) et non les 74 Mo touchés. Le froid ci-dessous = 4 (Coder) ou 2
(GLM) pools d'experts DISJOINTS dans un même graphe (297 / 282 Mo > 96 Mo
de L2), temps de rejeu ÷ appels — ce qu'un pas fait sur 48 couches.

| forme, jetons | experts distincts | L2 | µs/couche | Go/s effectif (poids) | W brut | MHz |
|---|---:|---|---:|---:|---:|---:|
| Coder, 1 | 8 (21 Mo) | chaud | 22,5 | 940 | 243 | 3 000 |
| Coder, 1 | 8 ×4 pools | froid | **26,8** | 790 | 253 | 2 977 |
| Coder, 12, pool 30 | 28 (74 Mo) | chaud | 31,7 | (2 340 : L2) | 398 | 2 872 |
| Coder, 12, pool 30 | 28 ×4 pools | **froid** | **62,7** | **1 180** | **401** | 2 910 |
| Coder, 16, pool 30 | 30 ×4 pools | froid | 81,1 | 980 | 399 | 2 887 |
| Coder, 12, routage d'origine | 96 (255 Mo) | (> L2) | 174,7 | 1 460 | 400 | 2 512 |
| Coder, 12, pool 30, **MXFP4** | 28 ×4 pools | froid | 60,5 | 1 220 | 400 | 2 872 |
| GLM, 1 | 4 (19 Mo) | chaud | 22,5 | 840 | 244 | 2 985 |
| GLM, 12, pool 34 | 23 (108 Mo) | chaud (> L2 ?) | 89,0 | 1 210 | 400 | 2 580 |
| GLM, 12, pool 30 | 25 ×2 pools (118 Mo) | froid | **112,2** | **1 050** | 400 | 2 752 |
| GLM, 16, pool 30 | 26 ×2 pools | froid | 125,4 | 970 | 399 | 2 767 |

(E=64 ne permet pas 2 pools disjoints de 34 : 30 → 25 distincts ; à 34
experts, 160 Mo, extrapolation à 1,05 To/s ≈ 150 µs.)

## Contre les seuils scellés

| | poste7 | moi | mesuré | verdict |
|---|---|---|---|---|
| Coder 12 jetons, ~30 experts | 80-100 µs (réfuté < 70 : « ma borne d'octets est fausse ») | 85-110 froid, 40-60 chaud | **62,7 froid, 31,7 chaud** | **réfuté par le bas, les deux** : b12x tient 1,18 To/s sur 74 Mo (1,46 sur 255) — la borne de 1 050 Go/s est celle de NOS noyaux, pas de la carte (REGLES §9, déjà annoté) |
| Coder 1 jeton | 30-40 | 30-45 | 26,8 froid / 22,5 chaud | sous le seuil : plancher de lancement ~20 µs, pas 30 |
| GLM 12 jetons, ~34 experts | 170-210 | 165-220 | 112 à 25 experts → ≈ 150 extrapolé à 34 | sous les deux (≈ −12 %) |
| W en boucle | — | ≤ 345 (0,18 inst/oct) | **400, au plafond** (2 900 MHz) | réfuté : même à 0,18 inst/octet, 1,2 To/s de DRAM + tensor cores saturent 400 W ; ce qui distingue b12x de nos GEMV n'est pas la puissance mais l'horloge tenue (2 900 contre 1 600-1 800) et le temps |

## Ce que ça ordonne

Notre chemin B (MMA décodage) : 3 GEMM ≈ 75 µs + glue (2 quant_act,
moe_act, reduce, argsort/gather) ≈ 105-110 µs par couche à 12 jetons ;
b12x fusionné à L2 froid : **62,7 µs**, même octets, même W. Écart ≈ 45
µs/couche = **2,1 ms par pas de 13,75 (−15 %)** et, à W égal, −15 % de
J/jeton. Deux voies : (a) brancher b12x (venv séparé : cutlass-dsl 4.7.1 +
torch 2.14 — notre venv est en torch 2.13 ; une passerelle .so ou une
mise à niveau, décision poste7) ; (b) fusionner nous-mêmes gate·up + down
+ activation en un noyau (la glue est 30 µs, les 3 lancements ~10 µs) —
prédiction : −25 à −35 µs/couche, soit la moitié de l'écart, sans la
dépendance. À b=1 le fusionné ne rend que 26,8 µs contre nos ≈ 4,18 ms /
48 = 87 µs par couche tout compris (GEMV 21 Mo) — l'écart à b=1 est
ailleurs (projections int8, lm_head : 1aj).

## Voie (c) de poste7 — contrôle de dépendance du cubin (15/09, 1 h, hors carte)

Ce que produit `cute.compile(..., --enable-tvm-ffi)` et que FlashInfer met en
cache (`~/.cache/flashinfer/0.6.18.post1/120f/cached_ops/b12x_moe_sm120a_cute_dsl/`) :
**un objet ELF relocatable `.o` de 160 Ko par forme** —
`static_m12_k2048_n768_t8_r96_<sha>.o` (Coder, b=12), `static_m16_..._r128`
(godet 16), `static_m12_k2048_n1536_t4_r48` (GLM), `micro_m1_...` (b=1). Il
contient le cubin (fatbin chargé par `_cudaLibraryLoadData` + `cuda_load`) ET
le code hôte MLIR de lancement (grille, shared, `_cudaLaunchKernelEx`) ; pas
de `cuModuleLoad` à faire nous-mêmes, ni de signature de kernel à
reconstituer : l'entrée est `__tvm_ffi_b12x_moe_static_<forme>` (convention
TVM-FFI : tableau de `TVMFFIAny`, tenseurs en `DLTensor*`, flux pris dans
l'environnement).

Symboles non définis du `.o` (nm) : 11 enveloppes `_cuda*`/`_cu*`
(`_cudaLibraryLoadData`, `_cudaLibraryGetKernel`, `_cudaLaunchKernelEx`,
`_cudaFuncSetAttribute`, `_cudaKernelSetAttributeForDevice`,
`_cuKernelGetAttribute`, `_cudaGetDevice`, `_cudaSetDevice`,
`_cudaDeviceGetAttribute`, `cuda_dialect_init/unload_library_once`,
`CuteDSLRT_TVMFFISetRaisedCudaError`) fournies par
`libcute_dsl_runtime.so` (43 Mo, C pur, ne dépend que de libc/libdl, fait
`dlopen`/`dlsym` de libcudart — pas de Python, pas de torch) ; et 3
symboles TVM-FFI (`TVMFFIEnvGetStream`, `TVMFFIErrorSetRaisedFromCStr[Parts]`)
de `libtvm_ffi.so` (2 Mo). **Aucune dépendance au runtime Python
cutlass-dsl ni à torch 2.14 à l'exécution** : la réfutation structurelle de
poste7 ne tient pas — sous réserve du point 3.

Ce qu'il faut de notre côté :
1. lier le `.o` dans notre extension (ninja : un objet de plus), avec les
   11 enveloppes ÉCRITES PAR NOUS (renvois directs vers cudart/cuda
   driver : `_cudaLaunchKernelEx` → `cudaLaunchKernelExC`, etc.) et les 3
   stubs TVM-FFI (flux courant de torch, message d'erreur) — sans embarquer
   `libcute_dsl_runtime.so` ;
2. un appel C++ qui remplit les 24 `DLTensor` dans l'ordre de
   `moe_dispatch.py:1722-1750` (a, ids compacts, poids de routage, 6
   espaces de travail `packed_a/scale/barrier`, w13/down FP4 + leurs
   échelles en disposition MMA native (32,4,m_tiles,4,k_tiles,E), row_counts,
   active_expert_count, weight_expert_ids, global_to_local, input_gs,
   alpha ×2, down_input_scale, sortie scatter, token_map, token_weights) ;
   tailles des espaces lues dans `B12xMoEWrapper.__init__` ;
3. **vérifier les signatures des enveloppes** : `libcute_dsl_runtime` les
   résout par `dlsym` sur cudart (vu à l'objdump : trampolines), mais si
   une enveloppe ajoute un argument, l'édition de liens passe et l'appel
   plante — contrôle par désassemblage des sites d'appel du `.o` (registres
   chargés avant `call`) avant de lier, 1 h ;
4. le pré-noyau `compact_topk_ids` (Triton, 30 lignes,
   `triton_compact.py`) réécrit en CUDA : ids globaux → compacts
   [0, actifs), table compact→global, compte ; les fantômes (-1) à traiter
   (`other=-1` dans le Triton : un id -1 devient un expert « −1 » —
   à masquer avant, poids 0) ;
5. conversion de disposition des experts au chargement : `w13_fp4`
   = [E, 2I, K/2] (gate et up concaténés par lignes, à vérifier
   intercalés ou empilés), `down_fp4` [E, K, I/2], échelles E4M3 bloc 16
   en tuiles MMA (128 lignes × 64 colonnes de blocs) — nos piles
   `("nvfp4", qw, bs, gs, K, M)` ont les mêmes octets E2M1, seules les
   échelles changent de disposition : conversion hors ligne ou au
   chargement (poste1 : placement, tables d'adresses — le `.o` prend des
   pointeurs contigus [E,…], pas une table par expert : **incompatible
   avec l'exil par expert** tel quel).

**Bloquant à trancher AVANT le jour de travail (utilisateur) : la licence.**
`nvidia_cutlass_dsl` 4.7.1 est sous « NVIDIA Software License Agreement »
(pas BSD comme CUTLASS) : distribuables = « python files in the Software
package in source format » ; « unless a developer tool is identified as
distributable, it is delivered for your internal use only » ; 2.2 interdit
de distribuer « any portion of the Software or Derivatives ». Donc
`libcute_dsl_runtime.so` ne va PAS dans le `.deb` (d'où les enveloppes à
nous, point 1) ; le `.o` compilé depuis le code de FlashInfer (Apache-2.0)
par le DSL : sortie de compilation, pas « Software » à ma lecture — mais
c'est une lecture, pas un droit ; usage interne (bancs, duel) sans
question. Si le `.deb` ne peut pas l'embarquer, (c) sert au duel et aux
mesures, et le produit garde (b) (fusion à nous) — ordre inversé par
rapport à poste7.

Prédiction si (c) aboutit (rappel des seuils de poste7) : 62,7 µs ± 10 % à
froid avec fantômes ; pas Coder b=12 −1,8 à −2,3 ms sur le bras B de
c6377d5 ; J −12 % ; PPL b12x/A ≤ 1,010 (poste2, teacher forcing).

## Plan de découpe chiffré du port (b) — lecture de `moe_static_kernel.py` (15/09, hors carte)

Ce que fait b12x « static » (Apache-2.0, `moe_static_kernel.py:1-134, 340-590`,
`moe_dispatch.py:295-321, 1634-1637`) à 96 lignes routées (12 × 8) :

* **Un seul noyau résident**, deux phases séparées par une barrière de
  grille : (1) frontend — un leader de CTA par paire routée : `atomicAdd`
  sur `row_counts[expert]`, écrit jeton source + poids de routage, quantifie
  la ligne en FP4 dans le stockage expert-major `[max_rows, K, E]` ;
  (2) compute — unité de travail = **(m-tile, tranche d'intermédiaire de
  128, expert)** prise dans une file linéaire (`_compact_static_get_work_tile`),
  tuile MMA (64, 128) sous 128 lignes routées, K par pas de 128, 4 warps
  MMA + 1 warp TMA (160 fils), 2 étages, `MmaMXF4NVF4Op` (le même
  `kind::mxf4nvf4` que le nôtre).
* **FC1 gate et up ensemble** sur la tranche (mêmes fragments A, poids lus
  une fois), SwiGLU dans les registres, **quantification de la tranche en
  FP4 directement dans la shared** (A du FC2), puis **FC2 balaie les 16
  tuiles de sortie** (2048/128) avec cette tranche de K=128 et **accumule
  par `atomicAdd` bf16x2 dans la sortie token-major**, pondéré par le
  routage : l'intermédiaire ne touche jamais la mémoire globale, il n'y a
  ni `moe_act`, ni `quant_act` de l'intermédiaire, ni `reduce`, ni
  scatter — FC2 est un split-K par tranche d'intermédiaire (6 tranches pour
  I=768) résolu par atomiques.
* Travail à b=12 Coder : ≈ 28 experts × 1 m-tile × 6 tranches = **168
  unités** pour 170 SM — une par CTA (échelle « MAC » 64-148 CTAs
  résidentes selon les lignes, `_STATIC_MAC_LADDER`).

Notre chemin B, en regard : 3 lancements de GEMM (gate, up, down :
`nvfp4_gemm_grouped_mma2`, BM=128 de sortie × BT=16 jetons, K pipeliné
cp.async 4 étages), `quant_act` ×2, `moe_act`, `moe_reduce_trie`, plus
argsort/gather/scatter torch — l'intermédiaire fait 5 passages en global
(g, u écrits ; lus par moe_act ; act écrit ; lu par quant ; aq écrit ; lu
par down) : 96 × 768 × 2 o = 150 Ko par passage, négligeable en octets,
**≥ 7 lancements par couche** contre 1.

### Découpe proposée (CUDA C++, notre extension, sans DSL)

| élément | b12x | port |
|---|---|---|
| CTA | (m-tile 64, tranche 128, expert), 160 fils | (expert, m-tile **16** — b=12 tient dans une tuile m16, godet 16 aussi —, tranche d'intermédiaire 128), **256 fils = 8 warps**, chaque warp 2 tuiles n8 sur les 128 colonnes |
| FC1 | gate + up, TMA 2 étages | gate + up dans la même boucle K (fragments A partagés), cp.async 4 étages, KS=128 (notre pipeline existant) : 2 × [16×2048]·[128×2048]ᵀ |
| activation + quant | registres → shared FP4 (coopératif) | registres → shared : 16 lignes × 64 o de nibbles + 8 o d'échelles UE4M3 (bloc 16), **même formule que `nvfp4_quant_act` (ties-to-even, `__fdiv_rn`) : bit-identique au chemin B** — c'est le test |
| FC2 | 16 tuiles n128, atomicAdd bf16x2 token-major × poids | idem : [16×128]·[128×128]ᵀ par tuile depuis down[e][:, tranche], **`atomicAdd` fp32** dans un tampon [t, 2048] fp32 (évite la perte bf16 des atomiques : 6 tranches × 8 experts = 48 sommes par sortie) puis une conversion bf16 (1 lancement, ou dans le noyau d'après) ; fantômes : poids 0 |
| frontend | route/pack en noyau + barrière de grille | **phase 2 seulement d'abord** : on garde argsort/scatter_add_/gather torch + `quant_act` de x (capturables, ~15 µs) ; la barrière de grille et le pack en noyau = marche 2 |
| grille | file linéaire persistante | grille fixe **E_max × 6 tranches** = 128 × 6 = 768 blocs (tuiles n=0 sautées : `if (nt <= 0) return`, déjà là) — capturable, 4,5 blocs par SM, un seul lancement |

Octets : par unité (e, tranche) 128 Ko gate + 128 Ko up + 128 Ko down
(+ échelles 24 Ko) = 408 Ko ; × 28 experts × 6 = **68,5 Mo** = exactement
les poids des experts touchés, lus une fois (aujourd'hui aussi ; le gain
n'est pas en octets). Temps prédit à L2 froid : 68,5 Mo à 1,18 To/s
= 58 µs + queue (168 CTAs actives sur 768 lancées, dernière vague) →
**60-75 µs/couche** (poste7 : ≤ 75, réfuté > 90) contre 3 GEMM 75 + glue 30
aujourd'hui ; pas b=12 : −(105 − 70) × 48 ≈ **−1,7 ms** (−12 % sur 13,75).
Risques nommés : (a) 6 tranches × 8 experts d'atomiques fp32 par sortie
(96 × 2048 × 48 = 9,4 M atomiques par couche ≈ 10-15 µs si sérialisés
sur le L2 — b12x le fait en bf16x2 et vit avec) ; (b) l'occupation : 8
warps × (fragments gate+up 2×[16×128] acc = 64 fp32/fil) tient ; (c)
bit-identité avec B non garantie sur FC2 (ordre des sommes du split-K
par atomiques ≠ réduction triée) → le test au bit se fait sur FC1+quant
(déterministe), FC2 contre float64 à tolérance, jetons A/B et PPL poste2.

Coût : 2-3 j (noyau 1 j, tests 0,5 j, câblage graphe + mesures 0,5-1 j).
Ce que le ncu (1) doit dire avant : si la glue mesure ≥ 20 µs/couche,
poste7 ordonne la marche de glue d'abord (quant dans l'épilogue de moe_act,
réduction dans down) — elle est CONTENUE dans ce port (activation+quant
dans l'épilogue de FC1, réduction par atomiques de FC2) : la faire
séparément ne sert que si le port glisse.

## (1) ncu du chemin B au décodage (15/09, 14:37-16:07, carte, deux passes)

Un pas b=12 sous graphes, MMA décodage (défaut 0.6.1) : **3 617 lancements
par pas** (A : 1 506 — le routage torch en ajoute ~2 100). ncu rejoue chaque
noyau avec sauvegarde des 15 Gio : 46 min par passe (leçon écrite dans le
script : `NCU_LANCEMENTS=2000` par défaut, une passe par appel carte.sh — le
processus est root, il ne se tue pas ; incident signalé, REGLES §6).

| noyau (par pas, ÷48 = par couche) | lanc. | ms ncu chaud | ms ncu purgé | Go DRAM chaud | Go purgé |
|---|---:|---:|---:|---:|---:|
| 3 GEMM `mma2` (gate, up, down) | 144 | 4,112 (**86 µs/couche**) | 4,117 | 4,285 | 4,303 |
| `quant_act` ×2 | 96 | 0,270 | 0,307 | 0,000 | 0,036 |
| `moe_act` | 48 | 0,093 | 0,110 | 0,000 | 0,019 |
| `moe_reduce_trie` | 48 | 0,104 | 0,126 | 0,000 | 0,026 |
| routage torch : `index_elementwise` ×5 | 240 | 0,629 | 0,897 | 0,006 | 0,017 |
| `scatter_gather` ×2 | 97 | 0,221 | 0,262 | 0,003 | 0,006 |
| `searchsorted` | 48 | 0,130 | 0,188 | 0,001 | 0,001 |
| `gather` | 48 | 0,096 | 0,118 | 0,000 | 0,003 |
| **glue totale** | ~625 | **1,54 (32 µs/couche)** | **2,01 (42)** | 0,010 | 0,108 |
| pas entier | 3 617 | 18,09 | 19,48 | 5,694 | 5,974 |

Lecture : les octets ne changent pas (les GEMM lisent leurs 4,3 Go une
fois ; l'intermédiaire g/u/act/aq — 5 passages de 150 Ko — est servi par le
L2 à chaud, coûte 0,1 Go purgé) ; **la glue est du temps de lancement, pas
des octets** : 32-42 µs/couche sous ncu (surestimé ~45 % sur ces noyaux
de 2-13 µs → réel ≈ 20-30), dont **22-30 de routage torch** (argsort,
scatter_add_, `_tuiles`, gather, inv : ~45 lancements/couche) et 10-12 de
quant/act/reduce. Règle de poste7 (≥ 20 → glue d'abord) : glue d'abord, et
la marche utile est le routage (§ 2 rescellé : route+pack CUDA en un
lancement, ≤ 5 µs/couche, lancements/pas ≤ 1 700, pas −0,6/−1,0 ms,
réfuté > −0,3).

## Les 3 GEMM du chemin B à b=1/6/12 (15/09, 16:57-17:00, ncu une passe bornée, cache chaud)

Ordre de poste7 (`poste7-mma-lot-15-09.md` § 2) : le coût fixe de B hors glue
(2,0-2,3 ms/pas, courbe de poste3) serait un plancher de latence par GEMM
(une seule vague, chaque CTA parcourt K entier) — confirmé si
b=6 ≤ 1,3 × b=1, réfuté si ≥ 2×. Mesure : `nvfp4_gemm_grouped_mma2` seul,
144 lancements = 1 pas, MMA forcée (`MIN_T=1`), grille fixe, route+pack.

| b | ms/pas (3 GEMM, ncu) | µs par GEMM | Go lus/pas | Go/s effectif | grille (blocs, moy. 3 GEMM) | warps actifs | MHz |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,425 | 9,9 | 1,020 | **715** | 1 204 | 16,5 % | 2 517 |
| 6 | 3,291 | 22,9 | 3,417 | 1 038 | 1 232 | 16,6 % | 2 456 |
| 12 | 4,010 | 27,8 | 4,285 | 1 069 | 1 269 | 16,6 % | 2 447 |

b=6 / b=1 = **2,31×** : le critère de poste7 réfute son plancher tel quel
(≥ 2×). Mais le mécanisme est à moitié là : à b=1 chaque GEMM lit 7,1 Mo
(8 experts) en 9,9 µs = 715 Go/s, contre 1 069 à b=12 — **un excès de
≈ 3,5 µs par GEMM à b=1** (7,1 Mo à 1,07 To/s = 6,6 µs), soit 0,5 ms par
pas de ncu sur 144 lancements (réel ≈ 0,3-0,4 avec la surestimation) ;
le reste du coût fixe de B à b=1 (2,0-2,3 ms mesurés par poste3) n'est PAS
dans les GEMM : il est dans la glue et les lancements (route+pack, 2
quant_act, moe_act, reduce_trie : 7 lancements de 2-13 µs chacun = 30-50
µs/couche = 1,5-2,4 ms/pas, indépendants de b — c'est le « fixe »). Grille
lancée 1 204-1 269 blocs quel que soit b (grille fixe : 129-134 tuiles ×
6 ou 16 tuiles N), mais à b=1 seules 8 tuiles portent du travail (48 CTA
utiles sur gate/up, 128 sur down) : occupation 16,5 % de warps actifs
partout — le pipeline cp.async à 8 warps est borné par la mémoire, pas par
l'occupation, dès b=6.

Conséquences : (1) route+pack (7 → 1 lancement de routage) est bien la
marche qui rend à TOUS les lots (le fixe est en lancements) ; (2) pour le
port, l'exigence « ≥ 170 CTA à b=1 » (split-K) ne vaut que 3,5 µs × 3
par couche = 0,5 ms/pas à b=1 — moins que la fusion des 7 lancements en
un (b12x fait tout en un noyau : 26,8 µs à b=1 contre nos ≈ 30 + 30 µs).

## Route+pack — A/B du pas b=12 sous graphes (15/09, 18:04-18:14, une carte)

En-tête : 5090 seule (carte.sh), 400 W, horloge libre, 40 °C, compteur NVML
(22 s), venv acvram, Coder-30B NVFP4, v0.6.3 + poste4 27ae762
(`ACVRAM_MOE_ROUTE_PACK=0` = A témoin torch, `1` = B), MMA décodage (godet 16
≥ 9).

| | A (routage torch) | B (route+pack) | Δ |
|---|---:|---:|---:|
| jetons (12 séquences, 229 pas) | 12 empreintes | **identiques** | — |
| lancements par pas (profil sous rejeu) | 3 677 (57 noyaux) | **1 517 (39)** | −59 % (seuil ≤ 1 700 tenu) |
| Σ noyaux GPU (profil) | 13,62 ms | 11,82 ms | −1,80 |
| ms/pas, banc 62 pas ×2 ABAB | 13,84 / 13,84 | **12,05 / 12,07** | **−1,79 ms (−12,9 %)**, 867 → 996 t/s |
| pas complet 22 s au compteur | 14,46 ms, 361 W, 5,22 J/pas | 12,80 ms, **398 W**, 5,09 J/pas | −11,5 % ms ; W +37 |
| J/jeton brut | 0,435 | 0,425 | **−2,4 %** |

Seuil de poste7 (−0,6 à −1,0 ms, réfuté > −0,3) : **tenu et dépassé
(−1,8 ms)** — la glue torch valait 1,8 ms de pas, pas 1,0 : ce sont les
lancements (2 160 de moins) et les trous entre eux, pas les µs de noyau.
Le J/jeton, lui, ne bouge presque pas : le pas plus dense remonte au plafond
de 400 W (361 → 398 W) — le temps gagné est payé en watts. Le levier
énergie reste dans les noyaux (b12x : 400 W aussi, mais 63 µs).
Défaut : `ACVRAM_MOE_ROUTE_PACK=1` ; test de lancements par pas
`tests/test_lancements_par_pas.py` (modèle + carte, sauté sinon) et par
couche (`test_moe_route_pack.py`, ≤ 8).

## Port (b) — noyau `nvfp4_moe_fused`, mesures (15/09, 18:58-19:54, carte)

Livré (poste4, `ACVRAM_MOE_DECODE_FUSED=1`, coupé par défaut) : CTA =
(tranche TN de l'intermédiaire, tuile de 16 jetons d'un expert), gate+up
dans la même boucle K (cp.async S étages × 128), act+quant en shared
(formules de moe_act/quant_act : FC1+quant bit-identique à B), FC2 par
tranches ; deux épilogues : split-K SÉRIEL (partiels fp32 en ws, dernier
CTA somme dans l'ordre fixe, `d` → reduce_trie ; bit-reproductible) et
témoin ATOMIQUES (y32 par jeton, cast bf16 ; non reproductible au bit).
Tests : jumelles au bit (sériel), référence float64, ≥ 95 % identique à B,
fantômes, TN 64/128.

| variante | ms/pas banc (62 pas ×5) | µs/couche ncu (cache chaud) | Go/s | SM actifs |
|---|---:|---:|---:|---:|
| B (3 GEMM + act + quant + reduce, route+pack) | 11,45-11,48 | 83 (3 GEMM seules) | 1 094 | 82 % |
| sériel TN128 S3 | 13,25 | 112 | 965 | 68 % (partiels : 22 Mo/couche ÉCRITS en DRAM — le L2 est write-back) |
| sériel TN64 S2/S3 | 19,6 | 203 (S3) | 701 | 52 % |
| atomiques TN128 S3 | 11,09 | 109 | 937 | 67 % |
| atomiques TN64 S2 | 11,45 | 77 | 1 082 | 87 % |
| **atomiques TN128 S2** | **10,93 (−4,6 %)** | **69,7** | **1 165** | 82 % |

Le sériel coûte ≈ 45 µs/couche (seuil de poste7 ≤ 5, > 10 → atomiques
acceptés avec contrôles). Ce qui a débloqué le noyau : **2 étages au lieu
de 3** (shared 48 → 2-3 CTA par SM : la latence d'un CTA seul à 16 lignes
n'est pas recouvrable, 109 → 70 µs), pas la largeur de tranche.

Régime ≥ 20 s au compteur (le chiffre qui compte), F = atomiques TN128 S2 :

| | B | F | Δ |
|---|---:|---:|---:|
| ms/pas (22 s) | 12,04 | 12,17 | **+1 %** |
| W | 399,4 | 399,9 | plafond des deux côtés |
| J/jeton | 0,401 | 0,406 | +1 % |
| jetons (12 séquences, 229 pas) | réf. | **4 divergences** (s4 @13, s7 @24, s8 @1 : 198/197, s11 @5) | atomiques : ordre des sommes non fixe |

Contre les seuils de poste7 (§ 3) : **≤ 75 µs/couche froid : TENU (69,7)** ;
**pas 22 s ≤ 11,3 ms : RÉFUTÉ (12,17, = B)** ; **J ≤ 0,38 : RÉFUTÉ (0,406 ;
sous le 0,41 de réfutation stricte, mais égal à B)** ; jetons identiques :
non (4/12, atomiques). Le banc court (−4,6 %) ne se retrouve pas à 22 s :
les 13 µs/couche gagnés sur les GEMM (83 → 70, 0,6 ms/pas) sont mangés par
les deux lancements ajoutés (zéros de y32, cast) et par le plafond de
400 W qui rend le pas insensible à 5 % de noyaux en moins — B et F sont au
même W, au même J.

Lecture : **le MoE n'est plus le poste** — 48 × 70 µs = 3,4 ms sur 12,2 ;
b12x à 63 µs ne ferait gagner que 0,3 ms de plus. Les 8,8 ms restants
sont les projections int8 (2,6), le lm_head (0,9), l'attention (0,7), les
normes/routage/glue (~2) et les trous de lancement — à 400 W, J/jeton suit
ms/pas. Le noyau fusionné reste disponible (coupé : atomiques non
reproductibles, pas de gain au régime 20 s) ; son gain apparaîtra si le
plafond cesse d'être atteint (mode éco, horloge bridée) ou pour GLM (I=1536,
tranche 128 = 12 CTA par expert-tuile). Données :
`revue/donnees-ncu-*fused*` (à copier) ; tests `tests/test_moe_fused.py`.
