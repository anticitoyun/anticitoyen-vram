# Vis-à-vis décodage b=12 : acvram vs vLLM, par poste (14/09, poste4)

Même modèle (Qwen3-Coder-30B-A3B, NVFP4 ; 48 couches, d=2048, 128 experts
top-8, inter 768, 32 têtes q / 4 kv × 128, vocab 151 936), même lot b=12,
régime court (nous : invites 128 + 64 décodés, `outils/banc_decodage_moe.py` ;
vLLM : invites 256 + 200 décodés, profil de poste2,
`acvram-memoire/revue/audit-a2-duel-vllm-14-09.md` § « Profil du pas de
décodage », 8 pas → ÷ 8).

Borne = octets DRAM ÷ 1 050 Go/s (lecture seule mesurée sur la 5090).
Octets « calc » = poids qu'il faut lire au moins une fois par pas ; les
octets DRAM RÉELS (`dram__bytes_read`, ncu) restent à mesurer — colonne
vide = pas encore mesuré, pas un zéro.

| poste | nous, ms (eager, profil) | vLLM, ms (profil ÷ 8) | octets calc nous | borne nous | % borne (sur calc) | octets DRAM réels |
|---|---:|---:|---:|---:|---:|---|
| MoE gate·up (`nvfp4_gemv_grouped_gateup`) | 3,35 | 5,95 (GEMM groupée, gate·up + down) | ≤ 5,1 Go (≤ 68 experts distincts/couche × 48 × 1,57 Mo) | ≤ 4,9 ms | ≥ 68 % | à mesurer |
| MoE down (`nvfp4_gemv_grouped_warp`) | 2,63 | ″ | ≤ 2,6 Go | ≤ 2,5 ms | ≥ 95 % | à mesurer |
| routage + glue (topk, tri, élémentaires) | ~1,7 | 0,97 shuffle + 0,46 reduce + 0,75 cvt ≈ 2,2 | ~0 (L2) | — | — | à mesurer |
| projections d'attention (`int8_gemv`, ×192) | 2,62 | 1,15 + 0,68 = 1,83 (GEMM FP4 dense) | 0,91 Go (int8) — vLLM : 0,51 Go (FP4) | 0,86 ms | 33 % (vLLM : 27 % de 0,49) | à mesurer |
| attention paginée (`paged_attn_partial`) | 0,72 | 0,66 (Triton unified) | 0,23 Go KV (L≈192) | 0,22 ms | 30 % | à mesurer |
| lm_head (`int8_gemv`) | 0,88 | 0,44 (wmma bf16, ligne mêlée au routeur) | 0,31 Go | 0,30 ms | 34 % | à mesurer |
| norm + reste | (dans ~1,7) | ≈ 3,3 (reste des 14,4) | — | — | — | |
| **Σ noyaux** | **12,25** (eager) — **14,5 rejeu** (horloge) | **14,4** sous profil — **10** à l'horloge | | | | |

Le MoE (6,0 ms, 48 % du pas) : la borne dépend du nombre d'experts
distincts réellement touchés par couche (uniforme : 68 ; réel : moins) —
c'est la mesure ncu qui donne le dénominateur. Si ≥ 70 % de borne, il n'y
a pas de ×2 à y prendre ; vLLM y est d'ailleurs au même chiffre (5,95).

## Ce que le tableau montre déjà

1. **Sous le même instrument (somme des noyaux profilés) nous sommes à
   12,25 et eux à 14,4 ; à l'horloge, 14,5 (rejeu) contre 10.** L'écart
   ×1,45 du duel n'est pas dans la somme des noyaux eager. Deux lectures,
   une mesure tranche :
   (a) le profil eager ne voit pas ce que le rejeu exécute : godet 16 pour
   b=12 (`graphs.py:45`, `bucket_batch`) = 4 lignes fantômes sur 12, +33 %
   de trafic x dans tous les postes étroits (projections, lm_head, norm) ;
   (b) leurs noyaux sont courts et le profileur les surestime (leçon ncu
   +45 % sur les noyaux courts) ; à l'horloge ils se recouvrent.
2. **Le seul poste où vLLM fait moitié moins en ms est lm_head** (0,44 vs
   0,88) — 3 % du pas, trop petit seul. **En octets, ce sont les
   projections d'attention** : 0,51 Go FP4 chez eux contre 0,91 Go int8
   chez nous, et leur ms (1,83) suit les octets (27 % de borne, comme nos
   33 %). Même noyau chez nous pour les deux postes (`int8_gemv`), même
   levier : diviser les octets par 1,8.
3. 12 jetons / 14,5 ms = 828 t/s ; le duel mesure 569 : **6,6 ms par pas
   (31 %) ne sont pas dans le rejeu** (banc HTTP de poste3, ordonnanceur,
   échantillonnage). Hors de mon périmètre noyaux, mais c'est le plus gros
   poste du duel.

## Mesures à faire (carte exclusive, `ACVRAM_TYPE=mesure`, `outils/carte.sh`)

M1 — noyaux SOUS rejeu : `BANC_GRAPHES=1 outils/banc_decodage_moe.py profil 12`.
Prédiction scellée : Σ noyaux rejeu ≥ 14,0 ms ; ≥ 60 % de l'écart
(14,5 − 12,25 = 2,25 ms) dans les postes étroits (projections int8,
lm_head, norm/glue), < 20 % dans le MoE (fantômes masqués par poste1).
Réfutation : écart < 1 ms, ou réparti au prorata des postes.

M2 — octets DRAM réels par noyau : `outils/ncu_pas_decodage.sh 12 gemv`
(`dram__bytes_read.sum`, 1 200 lancements après 4 000 sautés).
Prédiction scellée : MoE ≥ 70 % de borne (experts distincts 45-60/couche,
4,5-6,5 Go/pas) ; projections int8 ≤ 40 % ; lm_head ≤ 40 % ; attention
paginée ≤ 40 %. Réfutation : MoE < 50 % (alors c'est LUI le bead).

## Prochain bead (sous réserve de M1/M2) : projections d'attention + lm_head en W4A4 MMA natif au décodage

À N=12, une tuile m16 de `nvfp4_gemm_grouped_mma2` contient tout le lot
(0 ligne perdue au godet 12 ; 4 au godet 16) ; une seule « experte » par
projection, poids NVFP4 lus une fois (0,51 + 0,17 Go au lieu de 0,91 +
0,31), activations quantifiées par `nvfp4_quant_act` (déjà bit-identique
à la référence). C'est exactement le chemin de vLLM (GEMM FP4 dense,
1,83 ms).

Prédiction scellée : projections 2,62 → 1,6 ms (± 0,2), lm_head 0,88 →
0,5 ; pas rejeu 14,5 → 13,0 (−10 %). Réfutation : < −0,6 ms sur le pas.
Condition de qualité : PPL (poste2) ≤ +0,5 % sur le total — l'« attention
en NVFP4 » a déjà perdu une fois à b=1 avec le GEMV
(`docs/FEUILLE-DE-ROUTE.md:833`, 767 Go/s contre 1 780) : à b=12 avec la
MMA le régime est autre, c'est le contrôle qui peut rendre faux.
Issue gênante nommée : à 192 lancements × ~13 µs, si c'est le plancher de
lancement (~10 µs) et non les octets qui domine, le W4A4 ne rend que
−0,3 ms et le vrai levier est le nombre de lancements (fusion norm +
projection, ou q/kv/o en un seul lancement groupé).
