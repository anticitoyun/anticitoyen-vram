# 194 — à sec (poste1, 25/09) : moins d'appels à l'étroit int8 ? Le levier au bit est α/β (≈ 0,6 ms/pas) ; NInfer ne fait PAS moins d'appels

* instrument : trace nsys de la 185 c prise 1 (`~/.cache/acvram/dumps-poste1/p185c_cuda_gpu_trace_nvtx=mesure.csv`, 50 pas,
  hors git) ; trace NInfer de la 148 bis (`travail/poste5/scratchpad/poste5-p148bis-24-09/trace-ninfer_cuda_gpu_trace.csv`, hors git) ;
  lecture du code (fichier:ligne ci-dessous). Aucune carte.
* commit : poste1-194 = origin/main f3197f0b
* régime : Qwen3.8-27B-unsloth-mixte-i8c, b = 8, défaut de main (`ab=auto(48)`), -lgc 2700 ; NInfer `qwen3_8_27b_nvfp4.ninfer`
* scellé : aucun (à sec) ; chaque chiffre de gain ci-dessous est une borne à prouver par un banc ≤ 5 min avant tout code
* mesuré : 145 appels étroits/pas (pas 193), 877 noyaux/pas ; NInfer 8 bits : mêmes 4 formes, mêmes comptes par couche
* verdict : prémisse corrigée (145 appels ; NInfer n'en a pas moins) ; classement ci-dessous
* durée : à sec, ≈ 25 min

## Correction de la prémisse : 145 appels, et NInfer en a autant
| forme (N × K) | appels/pas | nous µs/appel | NInfer µs/appel (géométrie) | écart × appels |
|---|---|---|---|---|
| GDN qkv‖gate 16384 × 5120 (`_etroit_segments`, 256 × 4) | 48 | 63,03 | 52,36 (1 024 blocs × 256 fils) | 0,51 ms |
| GDN out + attention o 5120 × 6144 (`_etroit_reduit`, 80 × 5) | 64 | ≈ 27,5 | 20,79 (320 blocs) | 0,43 ms |
| qkv attention 14336 × 5120 (224 × 2) | 16 | 53,95 | 45,95 (896 blocs) | 0,13 ms |
| down int8 5120 × 17408 (80 × 5) | 8 | ≈ 66 | 55,99 (320 blocs) | 0,08 ms |
| gate‖up int8 34816 × 5120 (544 × 1) | 8 | 157,2 | — (autre format chez NInfer) | — |
| tête 3880 × 1 | 1 | 841 | 788 | 0,05 ms |
Deux appels étroits par couche (64 couches) + 2 par couche MLP int8 (8) + la tête = 145. Les « 193 » de la 188 dataient d'avant la
pile qkv‖gate (176). NInfer : `fp8_gemv_kernel<Fp8Geometry<N, K>>`, 143,6 lancements/pas — **même compte**, rapport 6:8:2:1 identique.

## Structure d'un pas (trace, une couche GDN puis une couche attention)
GDN : rmsnorm → **segments** → `_gemv_bf16` α/β (14,3 µs, 3 programmes) → conv → 2 elementwise → récurrence → `_norme_gated`
→ **out** → rmsnorm → Marlin gate‖up → swiglu → Marlin down. Attention : rmsnorm → **qkv** → elementwise → rope → kv_write →
attention partielle + réduction → 2 elementwise → **o** → rmsnorm → MLP. Entre deux étroits d'une couche, toujours une opération
dépendante (récurrence, attention) ; entre deux couches, la MLP Marlin.

## (a) Noyau persistant qui enchaîne les projections int8 d'une couche
* Faisabilité : les deux étroits d'une couche sont séparés par la récurrence GDN ou l'attention, qui dépendent du premier et
  nourrissent le second ; les enchaîner = porter ces opérations dans le même noyau avec des barrières de grille (Triton n'a pas
  de lancement coopératif) → mégakernel de couche.
* Gain max : seule la part « lancement + montée + vidange » disparaît ; la chaîne de l'épilogue (store, atomique, relecture,
  store) reste. Trous mesurés 0,297 ms / 877 noyaux = 0,34 µs ; montée ≈ 1-2 µs (V4 3,3-4,0 µs contient aussi l'épilogue) →
  **0,2-0,35 ms/pas**. Au bit : oui (ordonnancement seul). Coût : très élevé. **Non recommandé.**
* Variante bon marché, même borne : **PDL** (lancement dépendant programmatique ; Triton 3.8 installé : `launch_pdl`,
  `driver.c:1004`). L'étroit lit ses poids (indépendants de x) avant d'attendre son prédécesseur. Au bit : oui. Coût : ≈ 40 lignes,
  mais deux inconnues à banc-er avant (5 min) : (1) la capture torch garde-t-elle l'arête programmatique ? (2) sans
  `launch_dependents` dans les prédécesseurs CUDA (rmsnorm, Marlin), le gain tombe au seul lancement (≈ 0,05 ms/pas).

## (b) Fusion avec l'opération voisine
* Norme AVANT (72 étroits précédés d'un rmsnorm, 2,2-2,4 µs) : borne 0,17 ms/pas, mais chaque programme devrait relire la ligne
  x entière pour sa norme (1 024 programmes × 80 Ko = 82 Mo de L2 pour les segments) ; au bit douteux (arbre de réduction CUDA
  1 024 × 5 contre Triton). **Gain net ≤ 0,05 ms, probablement négatif. Rejeté.**
* Résidu + norme APRÈS (o/out) : la norme demande la ligne entière → le dernier programme de TOUTE la grille la calcule seul,
  queue série ≥ 2 µs contre 2,2 µs de lancement séparé. **≤ 0. Rejeté.**
* **α/β avec les segments** (même entrée normée, `gdn.py:251-257`) : `_gemv_bf16_kernel` = 3 programmes qui bouclent 40 fois
  sur K (`gemv_bf16_etroit.py:43`), 14,3 µs × 48 = **0,688 ms/pas** sur le chemin critique, pendant que les segments (60 µs)
  occupent la carte. NInfer le fait (`gdn_projected_conv`, 148 bis).
  - (b1) 3 programmes α/β EN TÊTE de la grille des segments (même corps, BM 16, BN 32, BK 128, 4 warps — comme les segments) :
    **−48 lancements, gain 0,55-0,65 ms/pas**, au bit probable (même code, même forme de lancement ; à prouver octet pour
    octet), coût moyen (≈ 60 lignes Triton + test au bit + capture godets {1, 2, 8, 16}).
  - (b2) sans code noyau : α/β sur un second flux dans le graphe (fourche/jointure autour des segments) : au bit par
    construction, gain 0,4-0,65 ms/pas (les 3 programmes attendent une place libre sous les 1 024 des segments), coût faible
    (gdn.py + capture) ; risque : ordre de capture et flux dans les graphes existants.

## (c) Ce que fait NInfer (géométrie figée par forme)
Pas moins d'appels : chaque appel est plus rapide — grille 1D de N/16 blocs × 256 fils, **K entier par bloc, sans split-K,
sans partiels ni atomique** ; débit 1,51-1,60 To/s contre 1,14-1,33 chez nous. Cohérent avec la 185 c (a) : 320-340 programmes
pour 5 120 colonnes, ≤ 2 par SM. Gain si nous égalions leurs µs par forme : **≈ 1,15 ms/pas** (table). Au bit : **NON** — K entier
change l'ordre de la somme (nos 5 tranches), et leur arithmétique W8A8 fp8 diffère encore → opt-in « ± 1 ulp » seulement
(REGLES § 1 : variable, ligne de régime, PPL + KL). Coût : moyen-élevé (noyau K entier à géométrie constexpr ≈ 150 lignes,
tables par forme, protocole ± 1 ulp).

## Classement (gain / au bit / coût)
| levier | gain max ms/pas | au bit | coût | banc de preuve avant code |
|---|---|---|---|---|
| (b) α/β dans les segments (b1) ou second flux (b2) | 0,55-0,65 | oui (b2 par construction) | faible-moyen | (b2) graphe à 2 flux, 5 min |
| (c) géométrie NInfer, K entier | ≤ 1,15 | NON (opt-in ± 1 ulp) | moyen-élevé | noyau K entier sur o/out, 5 min |
| (a) PDL | 0,15-0,3 (0,05 sans déclencheurs) | oui | moyen | arête gardée sous capture, 5 min |
| (a) persistant de couche | 0,2-0,35 | oui | très élevé | — |
| (b) normes avant/après | ≤ 0 | — | — | rejeté |
**Proposition : (b2) d'abord (au bit sans noyau neuf), puis (b1) si la contention mange le gain ; (c) en opt-in sur décision.**
