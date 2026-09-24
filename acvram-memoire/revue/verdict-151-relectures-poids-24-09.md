# Verdict — 151 (à sec) : nos GEMV à M = 2..8 lisent-ils chaque poids une fois ? — 24/09 (poste1)

Question du chef, tirée de la veille (sparkinfer PR #1081 : une boucle par ligne qui relisait et re-décodait les poids,
59 % du GPU à 16 flux). Lecture du code seulement, sans carte : les octets donnés sont ceux de l'algorithme (poids du
manifeste), pas des compteurs DRAM.

## Chemins servis à 2 ≤ M ≤ 8 : un poids lu et décodé UNE fois pour les M lignes
| chemin | où | pourquoi une fois |
|---|---|---|
| NVFP4 naturel, M = 2-3 | `kernels/__init__.py:647` → `nvfp4_gemv_kernel<ROWS, NV>` (`acvram_kernels.cu:264-376`) | le paquet `uint4` et ses échelles sont chargés une fois (`pre_p4[r]`, l. 295-346), décodés une fois (`e2m1_pair`, l. 365-366) et appliqués aux NV lignes (`for n < NV`, l. 368-371). NV = min(M, 8) par passe (`acvram_kernels.cu:1358-1373`) : une passe jusqu'à 8, deux au-delà (le GEMV ne sert plus au-delà de 3 lignes) |
| NVFP4 naturel, 4 ≤ M ≤ 32 | `kernels/__init__.py:629-637` → `_dense_etroit_kernel` (`gemm_dense_etroit.py:232-246`) | grille `(tuiles_n, tranches)` sans axe M, BM = 16 (M ≤ 16) ou 32 : chaque tuile de poids est décodée une fois (LUT) puis `tl.dot` avec toutes les lignes |
| Marlin unique, M = 1 | `_marlin_seul` (`kernels/__init__.py:995-1004`) → `nvfp4_gemv_marlin2` | une ligne, sans objet |
| Marlin unique, 2 ≤ M ≤ 32 | `_marlin_seul` → `MP.gemm_dense` (`kernels/__init__.py:1030`) | GEMM Marlin : un bloc M de 16 lignes (M ≤ 16) par tuile de poids, lue une fois |
| int8 (qkvo Coder, tête liée) M ≤ 16 | `int8_gemv_kernel<ROWS, NV>`, NV jusqu'à 16 par passe (`acvram_kernels.cu:569`, hôte `:1640`) ; étroit Triton `_etroit_reduit_kernel` BM = 16, grille `(tuiles_n, tranches)` (`gemm_etroit.py:32, 234`) | idem : une passe sur les poids pour ≤ 16 lignes |
| GDN, décodage b ≤ 8 sous graphes (le défaut) | `_la_decode` → `decode_static_batch` (`couches.py:246`, `gdn.py:269-276`) → `_lot_projete` → `_projections(x)` sur [b, hidden] | qkv, gate, α, β projetés une fois pour les b séquences ; `out_proj` une fois |

**Octets par pas à b = 8, Qwen3.8-27B (chemin servi)** : poids des couches 13,70 Go + tête 0,72 Go = **14,42 Go lus une
fois** ; plus l'état GDN (48 couches × 8 séquences × S fp32 [48, 128, 128] = 3,1 Mo, lu et écrit) ≈ 2,4 Go et le KV
(~0,3 Go) → plancher ≈ **17,1 Go ≈ 9,6 ms à 1 790 Go/s**. Mesuré (ABBA 129) : B 16,07 ms (60 % du plancher),
A 25,30 ms (bridé à 400 W). **L'écart au plancher ne vient pas d'une relecture des poids** ; il faut le chercher
ailleurs (occupation, lancements, état GDN en fp32).

## Relectures PAR LIGNE trouvées : trois replis GDN, dont un servi
1. **Vérification spéculative des hybrides (q_len > 1, b = 1)** : `_la_decode` DÉROULE les jetons un à un
   (`couches.py:271-273`, `for j in range(q_len): un(h[j:j+1], self.static)`). Chaque jeton refait `_projections`, donc
   les projections GDN (**3,13 Go** sur Qwen3.8 : qkv, gate, α, β, out) sont lues **q_len fois**, alors qu'elles ne
   dépendent pas de l'état (seules la convolution et la récurrence sont séquentielles). À k = 3 (q_len = 4) : **+9,4 Go
   par pas de vérification**, 23,8 Go au lieu de 14,4, soit ×1,65. C'est exactement la classe de sparkinfer, sur le
   chemin MTP de cette branche.
2. **Décodage eager des hybrides à b > 8** (les graphes ne couvrent que b ≤ 8 : « graphes=on(hybrides≤8) ») :
   `DecoderLayerGDN.forward` ne prend `forward_batch` que si aucun état n'est `_STATIC` (`couches.py:109-115`). Un état
   lié à un créneau de graphe l'est, d'où la boucle par séquence (`couches.py:126-133`) : les projections GDN sont lues
   **b fois**, 37,6 Go à b = 12 au lieu de 3,13. **À confirmer sur carte** : que les états y soient bien `_STATIC`.
3. Sans fla (`_voie_fla` faux, pas le défaut) : `torch.cat([un(h[i:i+1]) …])` (`couches.py:255`), soit b lectures.
   (MLA sans `_MLA_BATCH` : même forme, hors de la question.)

## Correctifs prédits (non codés)
1. q_len > 1 : `_projections` une fois sur les q_len jetons [q_len, hidden], puis convolution, récurrence et photographie
   de l'état jeton par jeton (le calcul séquentiel reste identique, au bit si les projections sont bit-identiques à M = 1
   contre M = q_len — à tester : le GEMV à NV lignes rend-il chaque ligne au bit de NV = 1 ?), puis `out_proj` une fois.
   Prédit : −9,4 Go par vérification à q_len = 4, soit environ −35 % sur le pas de vérification s'il est limité par la bande passante.
2. b > 8 eager : reprendre les états statiques en lot (`_reprendre` vectorisé) et passer par `forward_batch`, ou étendre les
   graphes hybrides au-delà de 8. Prédit : les projections GDN passent de b × 3,13 Go à 3,13 Go.
Condition commune : test d'équivalence (sortie au bit ou KL sous témoin) dans le même commit (REGLES : une optimisation
qui change la sortie est un bogue).

**CORRECTIF 24/09 12 h 25 (poste1, verdict 152)** : les relectures « par jeton » du point 1 sont servies par le L2, pas
par la DRAM. Dans le déroulé, la boucle des jetons tourne DANS la couche, et seuls ≈ 65 Mo (qkv, gate, α/β, out) passent
entre deux jetons, sous 96 Mo de L2. ABBA 152 : regrouper par poids = −0,5 %. Les « +9,4 Go » et le « ×1,65 » étaient des
octets de l'algorithme, pas des octets DRAM : ils ne comptent pas comme levier. Le point 2 (eager b > 8) relève
probablement du même cas. Leçon : des octets algorithmiques sans ensemble de travail rapporté au L2 ne donnent pas un levier.
