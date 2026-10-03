# Sonde, carte : le produit cuBLAS bf16 hors des six sites est NOMMÉ — les couches int8 de Devstral sont déquantifiées en bf16 au préfill et confiées à cuBLAS par le repli de `int8_matmul` (`kernels/__init__.py:1770`, `:1775`), 180 GEMM sur 7 865 lignes hors de `linear_prefill` ; l'attention est flash partout, le décodage est hors cuBLAS

instrument : `scratchpad/poste6-bf16/carte-sonde.sh` (garde de chaîne `poste6-sonde-chaine`, deux processus `service`) — `sonde-cublas.py` : `torch.profiler` avec formes sur un préfill de 7 865 jetons (`sequence_de_chauffe`, `max_tokens=1`), compteur d'appels de `kernels.linear_prefill` par forme, `sdpa_kernel([FLASH_ATTENTION])` forcé, 8 pas de décodage sous profileur, top-10 du premier jeton `etroite` contre `exacte` dans le même processus ; journal `scratchpad/poste6-bf16/carte-sonde.log` (non suivi)
commit : poste6-gemma-anneau d20dc91bb, arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, aucun autre poste (journal du verrou) ; Devstral-24B `srcawq-nvfp4` (40 couches : NVFP4 et int8 mêlés — q int8 × 9, k × 11, v × 6, o × 5, gate / up × 11, down × 13 d'après les comptes ci-dessous), NOMINAL, `reduction_bf16=reduite` ; torch 2.14.0+cu130
scellé : `poste6-bf16-sonde-scelle-03-10.md` (S1-S4)
mesuré : 03/10 03:32:47 → 03:33:36 (49 s)
verdict : **nommé.** Au préfill, chaque couche int8 passe par `int8_matmul` → `gemm_i8c_cublas` refuse (`:1156`, poids non éligible) → **déquantification en bf16 et `F.linear` cuBLAS** (`:1770` par tranches de lignes de sortie quand la matrice dépasse `_DEQUANT_TRANCHE_MAX`, `:1775` sinon) — hors de `linear_prefill`, donc hors de `etroite`, mais sous le drapeau global. Formes vues (S2) : k / v [7 865 × 5 120] × [5 120 × 1 024] × 17 (la forme dont 31 % des éléments dépendent de M), down par tranches [7 865 × 32 768] × [32 768 × 1 344 / 1 088] × 52 (K = 32 768, 18,6 %), gate / up par tranches [32 768 → 8 704 / 6 656] × 44, q × 9, o × 5 : **180 GEMM bf16 cuBLAS par préfill hors des six sites**, contre 225 dedans. S1 : l'attention est flash (40 `_scaled_dot_product_flash_attention`, flash forcé réussi) ; S4 : le décodage ne lance aucun GEMM cuBLAS (GEMV maison nvfp4 et int8) ; S3 : le premier jeton diffère entre `etroite` et `exacte` (Δ 0,079) et pas entre `etroite` et `reduite` (Δ 0) — le produit agit au préfill, comme prédit.
durée : 49 s de carte (prévu ≈ 1 min)

## Prédit / mesuré

| # | prédit | mesuré | |
|---|---|---|---|
| S1 SDPA au préfill | flash × 40 (A), aucun `bmm` / math ; flash forcé réussit | `_scaled_dot_product_flash_attention` × 40, noyau `pytorch_flash::flash_fwd_kernel` × 40, aucun `bmm` ; flash forcé : RÉUSSI | tenu (A seulement, voir limite) |
| S2 GEMM bf16 cuBLAS hors `linear_prefill` | ≥ 1 par couche int8 ; candidat : repli bf16 des couches int8, formes k / v et down | **180 hors / 225 dedans** : `aten::linear` [7 865, 5 120] × [1 024, 5 120] 80 contre 63 (+17), [4 096, 5 120] 40 contre 31 (+9), [5 120, 4 096] 40 contre 35 (+5), tranches [1 344 / 1 088, 32 768] 52 et [8 704 / 6 656, 5 120] 44 (0 dans `linear_prefill`) ; noyaux : `cutlass_80_tensorop_bf16_s16816gemm_*` seulement, aucun noyau int8 au préfill | **tenu, candidat confirmé** |
| S3 premier jeton `etroite` contre `exacte` | différents | ids de tête égaux, **Δ 0,079** sur 10 candidats ; `etroite` contre `reduite` : Δ 0 | tenu |
| S4 décodage | `bmm` bf16 cuBLAS (attention math) présent | **absent** : `nvfp4_gemv_marlin2_kernel`, `int8_gemv_kernel`, `nvfp4_gemv_kernel` ; aucun `aten::bmm`, aucun cutlass de plus que le préfill | FAUX : le décodage est hors cuBLAS — tant mieux, un produit de moins |

Limite, dite : le processus B (« morceaux ») n'a pas découpé — `sequence_de_chauffe` + `generate` en direct ne passent pas par
le planificateur qui pose les morceaux (le serveur de la chaîne S1 le fait : `prefill_morceaux 4`). Les deux processus ont
donc relevé un seul tenant ; S1 pour les morceaux reste la lecture à sec (biais bas-droite → flash), non vue au profileur.

## Ce que la fenêtre apprend

1. **La cause complète est écrite** : sous le défaut de torch, les produits cuBLAS bf16 d'un préfill Devstral sont (a) les
   six sites de `linear_prefill` (NVFP4 déquantifié, Marlin dépaqueté, vues) et (b) le repli bf16 des couches int8
   (`int8_matmul`, `:1770` / `:1775`). Les formes dépendantes de M (k / v N = 1 024 ; down K = 32 768) existent dans les
   deux ; `etroite` ne tenait que (a) — d'où « B ≠ A1 » (C1 du 03/10) et « A1 `etroite` = A1 `reduite` » (C5) alors que le
   drapeau global, qui tient (a) + (b), rend B = A1 (C1 du 02/10) et change A1 (C3 du 02/10).
2. **Le chemin int8 au préfill n'est pas « cublas »** sur ce modèle : la ligne de régime dit `prefill_int8=cublas`, mais
   `gemm_i8c_cublas` rend None pour ces poids (`_i8c_eligible`, `:1156`) et le repli déquantifie en bf16 — 180 GEMM
   cuBLAS de plus, 44 par tranches de lignes de sortie. REGLES § 6 : la ligne de régime nomme le réglage, pas le chemin
   pris ; un compteur `CHEMINS_INT8["repli_bf16"]` dit sur la ligne corrigerait cela (à proposer).
3. **Correctif qui découle, non codé ici** : faire passer `:1770` et `:1775` par `linear_prefill` (deux lignes) ; alors
   `etroite` couvre tout cuBLAS bf16 du préfill et doit rendre B = A1 au bit comme le drapeau global, à peu près au coût
   du drapeau global (+2,40 % à M = 4 096) — les +2,32 % de `etroite` n'incluaient pas ces 180 GEMM ; prédiction pour le
   scellé suivant : +2,4 à +2,8 %. Le seuil de 2 % resterait franchi à 4 096 ; c'est le chef qui tranche entre le seuil
   et l'indépendance au découpage.
4. S4 faux dans le bon sens : rien à faire au décodage.

## Ce qui reste au chef

* Sceller puis coder : `:1770` / `:1775` → `linear_prefill`, test cassant (un poids int8 inéligible au chemin cublas, M > 32,
  sous `etroite` : le drapeau est posé pendant l'appel), chaîne S1 `etroite` (prédit B = A1 au bit) et débit C2 (prédit
  +2,3 à +2,6 % à 4 096) — une fenêtre ≈ 4 min.
* Dire le chemin int8 réellement pris sur la ligne de régime (`repli_bf16=N`), pour que « prefill_int8=cublas » ne masque
  plus 180 GEMM bf16.
* Le défaut reste `reduite` ; `etroite` opt-in ; la garde de qualité au modèle reste due avant toute bascule.
