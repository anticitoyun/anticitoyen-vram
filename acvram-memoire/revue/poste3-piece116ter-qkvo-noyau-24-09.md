# Pièce 116 ter — lecture du noyau QKVO, rattachement des 3 grilles, prédiction

À sec, code lu, aucune modification, aucune carte. Fichier : `acvram/kernels/acvram_kernels.cu`
(noyau `int8_gemv_kernel`, ligne 569) et `acvram/engine/mla.py` (`fuse_projections`/`empiler`,
lignes 602-624 ; `_marquer_etroit`, 627-641) et `acvram/kernels/__init__.py` (routage narrow vs
gemv, 1020-1071 ; `_NARROW_MIN`, ligne 555).

## Correction avant tout : les 3 grilles ne sont PAS q/k/v/o

Seules 2 des 3 grilles de la trace 116 sont notre `int8_gemv_kernel` — la 3e (`gemvx::kernel
[32x1]`) est du cuBLASLt, un noyau différent, capturé dans `proj_etroites_int8` par le même
motif générique (`familles-noyaux.py:29`, `gemvx::kernel` dans la regex). Rattachement :

- **`int8_gemv_kernel<...> [1280x1]`** = **Q+K+V fusionnés en UNE GEMV** (`mla.py:623`,
  `self.q_kv = empiler([self.q_proj, self.kv_a_proj])`, `stack_int8_linears`). M = Q(4096) +
  K(512) + V(512) = 5120, K(entrée) = hidden = 2048. `ROWS_PER_BLOCK=4` (`acvram_kernels.cu:61`)
  → grille = ⌈5120/4⌉ = **1280**. Au bit avec la trace.
- **`int8_gemv_kernel<...> [512x1]`** = **O seul** (pas fusionné : `o_proj` n'entre pas dans
  `empiler`). M = hidden = 2048 (sortie), K(entrée) = num_heads×head_dim = 4096 (l'attention
  produit 4096, `o_proj` les ramène à 2048). Grille = ⌈2048/4⌉ = **512**. Au bit.
- **`gemvx::kernel [32x1]`** (cuBLASLt, PAS notre noyau) = très probablement le **routeur**
  (logits, non quantifié → bf16, M = 128 experts). 128/4 = 32 correspond par coïncidence à
  `ROWS_PER_BLOCK`, mais c'est le tuilage interne de cuBLASLt, pas notre macro — à confirmer si
  besoin (hors de portée « à sec »), pas Q/K/V/O dans tous les cas : cette part (0,10 ms/pas,
  5,2 Go/s) n'appartient pas à la question posée et doit sortir du calcul 116 bis.

## Pourquoi la GEMV et pas le GEMM étroit (tensor cores) à b=1

`_marquer_etroit()` (`mla.py:627`) pose `w.etroit = True` sur `q_kv`/`o_proj` int8 par défaut
(`ACVRAM_NARROW_MLA=1`) — ce qui devrait router vers `narrow_gemm` (tensor cores,
`kernels/__init__.py:1053`), PLUS RAPIDE en régime GEMM. Mais la condition exige
`_NARROW_MIN <= n` (`kernels/__init__.py:1057`), et `_NARROW_MIN = 2` par défaut
(`kernels/__init__.py:555`) : à **b=1, n=1 < 2**, la condition échoue et le chemin retombe sur
le GEMV pur (`kernels/__init__.py:1071`) — exactement ce que montre la trace. **C'est le régime
attendu, pas un défaut** : narrow_gemm sert à batcher plusieurs jetons sur les tensor cores,
sans bénéfice à n=1 seul. Le plafond bande passante (1,79 To/s) est donc la BONNE grille de
lecture pour QKV/O à b=1, contrairement à l'attention/normes/rope (116 bis).

## Forme, split-K, largeur des chargements, occupation

| | QKV fusionné | O |
|---|---|---|
| grille (x, y=splits) | 1280, 1 | 512, 1 |
| M (lignes de poids) | 5120 | 2048 |
| K (contraction) | 2048 | 4096 |
| threads/bloc (`threads_for`, `:1226`) | 128 | 256 |
| itérations K par thread | 128/128 = **1** (exact) | 256/256 = **1** (exact) |
| largeur de chargement | `uint4` = **128 bits** déjà (`:583`, une lecture par thread) |
| split-K (`splits_for`, `:1242`) | `row_blocks(1280) ≥ want(170×2=340)` → **1** | `row_blocks(512) ≥ 340` → **1** |
| octets lus (plancher, int8) | 5120×2048 = 10,49 MB/couche × 48 = **503,3 MB/pas** | 2048×4096 = 8,39 MB/couche × 48 = **402,7 MB/pas** |
| µs/pas mesurés (116) | 389,0 | 346,0 |
| Go/s atteints | 1 294 (72,3 % du plafond) | 1 164 (65,0 % du plafond) |

`SM=170` (RTX 5090, déjà posé dans `plancher-par-octets.py:66`), `want = SM×2 = 340`. Les DEUX
projections ont déjà plus de blocs que `want` : `splits_for` juge (par construction) qu'aucun
split-K n'est nécessaire, mais il ne regarde QUE le compte de blocs, jamais l'occupation réelle
par SM ni l'effet de vagues (voir plus bas).

## Les quatre leviers nommés par chef — tous DÉJÀ faits, aucun gain à en attendre

1. **Chargements 128 bits** : déjà `uint4` (`:583`), 1 seul par thread, aucune relecture — RIEN
   à gagner.
2. **Split-K** : déjà à 1, ET `row_blocks ≥ want` pour les deux — le critère existant dit
   explicitement qu'un split n'aiderait pas la SATURATION du nombre de blocs. RIEN à gagner
   par CE critère (le vrai goulot, s'il existe, est ailleurs — voir hypothèse).
3. **Fusion QKV** : déjà faite (`mla.py:623`, `empiler`) — RIEN à gagner, déjà la forme testée.
4. **Échelles** : `qkvo-i8c` = un `scale`/`zero` PAR LIGNE (`group_size = K`, cf.
   `gemm_etroit.py:141`), donc `ng=1`, une lecture triviale par bloc (`:614-616`, un `__half` et
   un `unsigned char` par ligne parmi ROWS=4) — poids négligeable dans le trafic total
   (quelques octets contre 8-16 Ko de poids par bloc). RIEN à gagner.

**Les quatre leviers proposés sont donc des impasses, déjà couvertes par le code actuel — dit
ici plutôt que réessayé.**

## Hypothèse retenue (non vérifiée, ncu manquant « à sec ») et prédiction

Chaque bloc ne fait qu'UNE seule itération sur K (128 ou 256 threads = exactement `nloads`,
`threads_for` cale pile dessus) : le bloc lit sa tranche de poids en UNE salve puis termine —
aucun flux soutenu à l'intérieur d'un bloc. Avec `row_blocks` 3,8× (QKV) et 1,5× (O) le nombre
`want` de SM×2, le lancement se fait en plusieurs VAGUES ; la dernière vague, partiellement
remplie, laisse des SM inactifs pendant qu'elle attend les autres — effet classique des noyaux
GEMV à bloc court (mémoire-bound mais à courte durée de vie, qui n'atteint jamais le régime
« flux soutenu » qui approche le plafond HBM). C'est une lecture du code, PAS une mesure
d'occupation (`ncu` non lancé, hors de portée « à sec, aucun code »).

**Prédiction chiffrée, écrite avant toute mesure** : si l'hypothèse « vagues » est la cause,
augmenter `ROWS_PER_BLOCK` (moins de blocs, plus gros, mieux remplis) rapprocherait les deux
projections de 85 % SANS toucher aux 4 leviers nommés. Gain visé pour ATTEINDRE 85 % (pas
100 %, cible de chef) :
- QKV : 389,0 µs → 503,3 MB / (0,85×1,79 To/s) = 330,8 µs → **gagne 58,2 µs/pas**
- O : 346,0 µs → 402,7 MB / (0,85×1,79 To/s) = 264,7 µs → **gagne 81,3 µs/pas**
- **Total prédit : ≈ 139,5 µs/pas gagnées (4,6 % du pas de 3,037 ms), SI et seulement si
  l'effet de vagues est la cause et qu'un `ROWS_PER_BLOCK` plus grand ne dégrade pas ailleurs
  (registres/occupation par bloc, ou perte de parallélisme utile à b=12 où le régime est
  différent — non vérifié).**

**Ce qui réfuterait cette prédiction** : un rapport `ncu` (occupation + débit DRAM) montrant que
les blocs sont DÉJÀ résidents au maximum permis par SM (aucune place pour plus de blocs
concurrents) ET qu'un micro-banc synthétique de MÊME forme de bloc (mêmes threads, mêmes
octets/bloc, aucune autre variable) plafonne lui aussi vers 65-72 % du débit HBM annoncé — alors
le plancher n'est pas un défaut de code mais une limite d'accès de CE motif de noyau (bloc
court, une seule salve), et 85 % ne serait atteignable qu'en changeant la FORME du noyau
(`ROWS_PER_BLOCK`, refonte de grille), pas par les 4 leviers nommés ni par un simple réglage.

## Reste, hors de cette pièce (aucun code demandé)

- Lire `gemvx::kernel [32x1]` pour confirmer qu'il s'agit bien du routeur (grep du site
  d'appel du GEMM des logits), pas laissé en supposition.
- Un vrai profil `ncu` (occupation, débit DRAM, vagues) trancherait l'hypothèse avant tout essai
  de `ROWS_PER_BLOCK` — sinon un changement de code viserait un mécanisme non prouvé.
