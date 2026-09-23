# Pièce 120 — audit SASS des noyaux GEMM/GEMV chauds, à sec, sans carte

Instrument : `.so` de `~/.cache/acvram/kernels-26080a459664/acvram_kernels.so`, compilé aujourd'hui 21:39 par
poste2 (`poste2-w-21-09`) sur le commit source identique au nôtre (md5 `acvram_kernels.cu` égal entre son arbre et
`poste4-nvtx-21-09`, vérifié avant lecture). `cuobjdump -res-usage` et `-sass -fun=<symbole>`, CUDA 13.4
(`/usr/local/cuda-13.4/bin`). Cible : `acvram_kernels.sm_120.cubin` seul ELF présent dans le `.so`. Aucune
compilation neuve lancée par moi (aurait exigé `torch.cuda.is_available()`, donc un contexte carte — REGLES
« à sec »). `marlin_port` et le cubin Triton de `gemm_etroit` n'ont pas de `.so`/cubin séparé trouvé à sec —
`narrow_gemm_kernel` (le vrai nom du noyau « gemm_etroit ») vit dans `acvram_kernels.cu`/`.so`, couvert
ci-dessous ; le Triton JIT n'est pas mis en cache sous forme de cubin lisible hors exécution.

## Méthode

Pour chaque famille : `REG`/`STACK`/`SHARED` (`cuobjdump -res-usage`), instructions MMA (`HMMA`/`IMMA`/`OMMA`/
`QMMA`, grep sur le SASS), compte de `LDL`/`STL` (déversement registre→local et retour = la signature d'un
déversement, pas la présence de `STACK>0` seule qui inclut aussi la trame d'appel).

## Résultats, par noyau chaud

| Noyau (fichier:ligne source) | Variante | REG | STACK (o) | MMA émise | Spills (LDL+STL) |
|---|---|---:|---:|---|---:|
| `narrow_gemm_kernel` (`acvram_kernels.cu:5458`) — c'est **gemm_etroit** | tuile 16, NVFP4=1 | — | — | `HMMA.16816.F32.BF16` | 48 |
| idem | tuile 32, NVFP4=1 | — | — | `HMMA.16816.F32.BF16` | 48 |
| idem | tuile 128, NVFP4=1 | 52 | 32 | `HMMA.16816.F32.BF16` | 48 |
| `nvfp4_gemv_kernel` (`acvram_kernels.cu:264`) | ROWS=4, NV=1 | 114 | 0 | **aucune** | (non compté, réf.) |
| idem | ROWS=4, NV=2 | 166 | 0 | aucune | — |
| idem | ROWS=4, NV=3 | 219 | 0 | aucune | — |
| idem | ROWS=4, NV=4 | **255** (plafond) | 0 | aucune | — |
| idem | ROWS=4, NV=7 | **255** | 328 | aucune | — |
| idem | **ROWS=4, NV=8** | **255** | **448** | aucune | **456** (sur 8 715 lignes SASS) |
| `nvfp4_gemv_marlin_kernel` (dense, `Li=2`) | bf16 | 64 | 0 | aucune | 0 |
| `nvfp4_gemv_marlin_slots_kernel` (Marlin par lot, `Li=2,slots=8`) | bf16 | 248 | 32 | aucune | 10 |
| `nvfp4_gemm_grouped_kernel` (MoE, forme simple) | — | 56 | 0 | `HMMA.16816.F32.BF16` | 0 |
| `nvfp4_gemm_grouped_mma_kernel` (MoE, ancien) | tuile 32 | 70 | 128 | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 15 |
| idem | tuile 64 | 100 | 208 | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 21 |
| `nvfp4_gemm_grouped_mma2_kernel` (MoE, actuel) | tuile 128, K=4 | 169 | 0 | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 0 |
| `nvfp4_moe_fused_kernel` | D3=3, tuiles 128/128 | 72 | 0 | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 0 |
| `paged_attn_partial_kernel` (décodage attention) | tuile 128, bf16 | 40 | 0 | **aucune** | 0 |

## Lecture, noyau par noyau

**`narrow_gemm_kernel` (gemm_etroit) — DÉJÀ conforme à la leçon de la pièce 119.** Le source
(`acvram_kernels.cu:5460-5463`, `mma_bf16_16816`) écrit l'instruction `mma.sync.aligned.m16n8k16.row.col.
f32.bf16.bf16.f32` **en PTX inline**, exactement le niveau que `waynehacking8/blackwell-tensorcore-kernels`
(pièce 119) mesure à 106 % de cuBLAS-TC — pas une API WMMA qui plafonnerait à 45 %. **Mais** : la branche
`NVFP4=1` du même noyau **dé-quantifie le FP4 en BF16 avant la MMA** (le même `mma_bf16_16816` sert aux deux
branches int8 et nvfp4) — elle n'utilise donc **jamais** l'instruction FP4 native `OMMA.SF` que sm_120 expose
et que d'autres noyaux du même fichier (ci-dessous) savent émettre. C'est un choix qui peut être délibéré
(précision, simplicité) mais qui laisse sur la table le débit FP4 natif de la carte pour ce chemin précis —
**à nommer, pas à corriger sans mesure**. Le compte de déversement (48) est **identique aux trois tailles de
tuile testées (16/32/128)** : pas un effet de taille, plutôt une routine partagée (probablement l'épilogue ou
le chargement des échelles) qui déverse un nombre fixe de valeurs quel que soit ROWS.

**`nvfp4_gemv_kernel` — LE noyau qui déverse vraiment, à nommer en premier.** Le second paramètre de gabarit
(`NV`, commentaire source : « activations par lecture de poids », `acvram_kernels.cu:262-263`) pipeline `NV`
colonnes d'activation en registres pour ne relire le poids qu'une fois. Le nombre de registres croît avec `NV`
(114 → 166 → 219) puis **plafonne à 255 (le maximum absolu par thread sur cette architecture) dès NV=4**, et
au-delà (`NV=5..8`) l'excédent part en pile : **STACK passe de 0 à 448 octets et 456 instructions `LDL`/`STL`
apparaissent** pour `NV=8` — un déversement massif, pas un cas limite. À 255 registres, l'occupation par SM est
mécaniquement plafonnée (peu de blocs résidents), et chaque déversement ajoute un aller-retour vers la mémoire
locale (L1/L2) sur le chemin chaud. **Aucune MMA ici, ce qui est attendu** : GEMV pur (M=1), les tensor cores ne
s'appliquent pas à cette forme — le problème n'est pas l'absence de MMA, c'est le déversement aux `NV` élevés.
**Reste à vérifier (hors cette pièce, à sec) : quel `NV` le régime sert réellement en production** — si c'est
`NV≤3`, le déversement sévère de `NV≥4` peut ne jamais être exercé ; si `NV=8` est atteint (gros lots
d'activations empilées), c'est un levier de gain nommé, chiffrable par un banc `ncu` ciblé sur ce seul noyau.

**Marlin (dense et par lot) — pas de MMA, par conception, pas un défaut.** `nvfp4_gemv_marlin_kernel` (dense) et
`nvfp4_gemv_marlin_slots_kernel` (par lot/godets) sont des GEMV : Marlin gagne en étant limité par la mémoire
(déquantification + FFMA), pas par le calcul — l'absence de MMA est le design, pas un manque. Le seul point à
surveiller : `nvfp4_gemv_marlin_slots_kernel` à `slots=8` tient à **248 registres, à 7 du plafond**, avec 10
déversements déjà présents — une évolution vers plus de slots simultanés (godets plus larges) rapprocherait ce
noyau du même mur que `nvfp4_gemv_kernel`.

**MoE groupé — le chemin qui exploite VRAIMENT le FP4 natif de sm_120.** `nvfp4_gemm_grouped_mma_kernel` (ancien,
tuiles 32/64) et `nvfp4_gemm_grouped_mma2_kernel` (actuel, tuile 128×K4) émettent tous deux
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` — l'instruction native de bloc-échelle FP4 de Blackwell (E2M1 × E2M1,
échelle UE4M3, accumulation FP32), la même classe d'instruction que documente le microbenchmark Blackwell cité
en pièce 119 (`arXiv:2507.10789`, « OMMA, QMMA » nommées comme les nouvelles instructions Blackwell). C'est le
noyau le plus proche de l'esprit CUTLASS/CuTe DSL de la pièce 119, mais écrit à la main. `nvfp4_moe_fused_kernel`
(le chemin fusionné, probablement le défaut servi) est **propre : zéro déversement**. L'ANCIEN
`nvfp4_gemm_grouped_mma_kernel` déverse (15 à 21 selon la tuile) — cohérent avec son nom qui suggère qu'il a été
remplacé par `mma2` ; à confirmer qu'il n'est plus sur le chemin par défaut avant de le juger prioritaire.
`nvfp4_gemm_grouped_kernel` (la variante « plain », sans MMA dans le nom) émet en fait `HMMA` BF16 — nom
trompeur, code correct.

**`paged_attn_partial_kernel` — absence de MMA cohérente avec l'état de l'art du décodage.** Godet de décodage
(M=1 par séquence), FFMA pur, zéro déversement, 40 registres seulement. Aucun moteur connu (vLLM `unified_
attention`, pièce 93/Q18 ; llama.cpp) n'utilise de tensor cores pour l'attention au décodage à q_len=1 — la
forme ne le justifie pas. **Pas un noyau à corriger.**

## Bilan pour chef

* **Deux familles natives déjà conformes à la pièce 119** : `HMMA.16816` (BF16, `mma.sync` inline) dans
  `narrow_gemm_kernel`/`nvfp4_gemm_grouped_kernel`, et `OMMA.SF.16864` (FP4 bloc-échelle natif Blackwell) dans
  `nvfp4_gemm_grouped_mma2_kernel`/`nvfp4_moe_fused_kernel` — aucun de ces chemins n'a besoin d'un changement de
  langage ou d'outil, la leçon de la 119 est déjà appliquée où la forme (M assez grand) le permet.
* **Un vrai déversement, nommé et localisé** : `nvfp4_gemv_kernel` à `NV≥4` (255 registres, jusqu'à 456
  déversements à `NV=8`) — à vérifier si `NV=8` est sur le chemin servi avant de le classer prioritaire.
* **Une occasion FP4 non prise, nommée, pas corrigée** : `narrow_gemm_kernel` dé-quantifie NVFP4 en BF16 avant
  la MMA au lieu d'utiliser `OMMA.SF` comme le fait déjà le chemin MoE du même fichier — écart de cohérence
  interne, à mesurer avant tout changement (peut être un choix délibéré non documenté ici).
* **Un noyau à surveiller, pas à corriger** : `nvfp4_gemv_marlin_slots_kernel` à `slots=8`, 248/255 registres.
* Aucun défaut trouvé sur `paged_attn_partial_kernel` (pas de MMA, cohérent avec l'état de l'art).

Aucune mesure carte, aucune décision prise — trois pistes nommées avec leur localisation exacte (fichier:ligne,
symbole SASS), à chef/poste7 de prioriser.
