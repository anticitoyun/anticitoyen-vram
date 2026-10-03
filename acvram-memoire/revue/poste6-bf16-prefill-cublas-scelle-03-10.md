# (a) à sec : le SDPA du préfill prend flash — ce n'est pas lui ; les produits restants sont les `F.linear` cuBLAS bf16 du chemin « Marlin dépaqueté au préfill » (q, o, gate, up, down), jamais couverts par `etroite` — scellé de (b) / (c) AVANT le code et la carte (poste6, 03/10 02 h 1x)

Ordre : chef 03/10 01 h 5x — « seulement (a), à sec, sans carte — établir le backend SDPA réellement choisi au préfill avec
masque explicite, et écrire le scellé de (b) / (c) ». Aucun code, aucune carte. Arbre lu : poste6-gemma-anneau 9e3f7cb6d ;
torch 2.14.0+cu130 (`sdp_utils.cpp` de l'étiquette v2.14.0, lu sur GitHub).

## (a) Le backend SDPA au préfill, établi par lecture

1. **Les morceaux n'envoient PAS de masque dense.** `layers.py:1163-1166` : `bas_droite` → `causal_lower_right(q_len,
   kv_len)` est un `CausalBias`, dont `__torch_function__` → `_dispatch` (`torch/nn/attention/bias.py:182`) : pour
   `LOWER_RIGHT` avec `q_len ≠ kv_len`, il construit `SDPAParams(q, k, v, attn_mask=None, …, enable_gqa)` et, si
   `can_use_flash_attention`, appelle `aten._scaled_dot_product_flash_attention(is_causal=True)` — sans masque. Si
   `q_len == kv_len` (premier morceau, seul tenant à `is_causal=True`, `layers.py:1174`) : SDPA ordinaire sans masque.
2. **Flash est admissible sur cette carte pour ces formes** (`sdp_utils.cpp` v2.14.0) : `check_flash_attention_hardware_
   support` accepte `sm80` … `sm121` (`:413-416`, la 5090 est sm_120) ; `can_use_flash_attention` (`:1047`) : pas de
   masque (`check_for_attn_mask`, vrai ici), head_dim 128 ≤ 256, bf16, dense, **GQA porté par flash**
   (`backend_supports_grouped_query_attention = true`, `:1082`), `check_flash_causal_non_square_seqlens` ne mord pas
   (`is_causal` est faux dans les paramètres du dispatch). Rien dans acvram ne désactive flash (aucun `sdp_kernel`,
   aucun `enable_flash_sdp(False)` : grep vide). cuDNN n'est préféré que sur sm_90 / sm_100 (`:95-105`).
3. Donc **le SDPA du préfill est flash, pas cuBLAS** : le drapeau bf16 ne le touche pas, et il est indépendant du découpage
   par construction (réduction par blocs de clés dans l'ordre, par ligne de requête). Mon candidat du verdict étroite tombe.
   Reste à le voir au moteur : `torch.backends.cuda.can_use_flash_attention(params, debug=True)` sur les formes exactes
   (1 s de carte, point B1 ci-dessous) — un `False` ici renverserait ce paragraphe.
4. **Les produits cuBLAS bf16 restants au préfill** (fichier:ligne, 9e3f7cb6d) : au-delà de 32 lignes et en régime
   `prefill=bf16`, le chemin « Marlin seule » **dépaquette les poids en bf16 et les confie à cuBLAS** —
   `kernels/__init__.py:1368-1393` (`marlin_depaquete_prefill`, `F.linear` aux lignes `:1390` et `:1393`), et la vue d'une
   pile Marlin fait de même (`:842-853`). Ce sont q, o, gate, up, down en NVFP4 (Devstral : q 31 couches, o 35, gate 35,
   down 27 ; les autres couches de ces rôles sont int8 → `_int_mm` entier, exact, hors sujet). `etroite` ne couvre que
   `_linear_naturel` (`:924`, `:971`, `:975`) : k / v NVFP4 et rien d'autre. H4 (02/10) n'avait mesuré sur la couche 0 que
   q, gate, up (0 dépendance à M, à K = 5 120) ; **o_proj (N 5 120, K 4 096) et down_proj (N 5 120, K 32 768) n'ont jamais
   été mesurés** — la couche 0 les a en int8. Un K de 32 768 est la forme où cuBLAS a le plus de raisons de couper la
   réduction (split-K) : c'est le suspect principal, et le plus gros produit du préfill (30 % des opérations) — ce qui
   expliquerait aussi que les +2,40 % du drapeau global ne soient pas dans k / v.

## (b) Mesures nues sur carte (une prise, ≈ 2 min, sans modèle chargé sauf B1)

| # | grandeur | prédit | faux si |
|---|---|---|---|
| B1 | `can_use_flash_attention(debug=True)` pour (1, 32, 7 865, 128) × (1, 8, 7 865, 128) bf16 GQA sans masque, et pour q 3 769 / kv 7 865 | **True, True** | un False : le SDPA du préfill n'est pas flash, (a) est faux et il redevient candidat |
| B2 | part des éléments qui changent entre 7 865 lignes et 4 096 + 3 769, `F.linear` nu bf16, drapeau retiré, par forme de Devstral : q (4 096 × 5 120), o (5 120 × 4 096), gate / up (32 768 × 5 120), **down (5 120 × 32 768)**, k / v (1 024 × 5 120, témoin) | q, gate / up : 0 (H4) · o : 0-5 % · **down : > 0, 10-40 %** · k / v : 15-31 % | down = 0 ET o = 0 : il ne reste aucun produit cuBLAS dépendant de M — la cause de E4 est ailleurs (cache, transitoires) et je le dirai |
| B3 | même chose sous le drapeau posé | 0 partout | > 0 : le drapeau global n'explique pas C1 |
| B4 | chrono exacte / réduite par forme, M = 512, 1 024, 2 048, 4 096 (médiane de 20) | down × 1,2-2,0 à 4 096 · gate / up × 1,00-1,10 · q, o × 1,00-1,15 · k / v × 1,13 (E7) ; somme pondérée ≈ +2,4 % à 4 096 | somme < +1 % : le surcoût global vient encore d'ailleurs |
| B5 | tranches de 1 024 lignes SOUS LE DRAPEAU RETIRÉ (chaque bloc à la réduction que cuBLAS choisit pour 1 024 lignes) : part des éléments qui changent entre 7 865 d'un coup en blocs et 4 096 + 3 769 en blocs, par forme | **0 partout** (les blocs sont les mêmes : 4 096 est un multiple de 1 024, le dernier bloc de 697 lignes est le même des deux côtés) | > 0 : cuBLAS change de noyau selon le nombre total de blocs ou la pile — tranches abandonnées |
| B6 | chrono tranches 1 024 / réduite d'un coup, par forme, M = 2 048 et 4 096 | × 1,00-1,06 (gate / up, down : × 1,00-1,03 ; q, o, k / v : jusqu'à × 1,10) | > × 1,10 sur gate / up ou down |

## (c) Correctif à sceller — deux bras, un seul retenu

Un seul point d'entrée `kernels.linear_prefill(x, w)` remplace les `F.linear` de `:853`, `:924`, `:971`, `:975`, `:1390`,
`:1393` (tout produit cuBLAS bf16 d'un poids quantifié au préfill). `ACVRAM_BF16_REDUCTION` :
* `etroite` (redéfini : **toutes** ces projections, plus seulement k / v) — réduction exacte le temps de l'appel ;
* `tranches` — drapeau laissé tel quel, M découpé par blocs de 1 024 lignes alignés sur le début de la séquence ;
* `exacte` (global) et `reduite` (défaut) inchangés. `etroite-tranches` disparaît (E9 : sans objet).

| # | grandeur | prédit | seuil / faux si |
|---|---|---|---|
| C1 | chaîne dense S1, `etroite` redéfinie, B / A1 (comparateur même candidat) | **Δ 0, ids égaux** (= C1 du 02/10 sous le global) | > 0 : un produit hors de ces six lignes dépend encore du découpage |
| C2 | débit moteur `etroite` redéfinie contre `reduite`, M = 512 … 4 096 | 4 096 : **+1,8 à +2,6 %** (≈ le global, le coût est dans ces GEMM) ; 2 048 : +0,4 à +1,0 | **> 2 % à un M → non retenue** (prédit : non retenue à 4 096) |
| C3 | chaîne dense S1, `tranches`, B / A1 | **Δ 0, ids égaux** (si B5 tient) | > 0 |
| C4 | débit moteur `tranches` contre `reduite` | 512, 1 024 : 0 ± 0,3 % (aucune tranche) · 2 048 : +0,0 à +0,8 · **4 096 : +0,2 à +1,5 %** | **> 2 % à un M** |
| C5 | seul tenant A1 `tranches` contre A1 `reduite`, 7 865 jetons | ids **DIFFÉRENTS** possibles : les blocs de 1 024 ne sont pas le noyau de 7 865 lignes d'un coup (C6 : 7 865 est une forme « exacte », 1 024 aussi — alors Δ 0 ; sinon bascule) | — (dit, non décisif) ; la section « ce qui change pour qui » du scellé étroite vaut ici aussi |
| C6 | morceaux NON alignés (cache de préfixe : reprise de 7 865 avec 2 000 jetons en cache, morceau de 5 865) sous `tranches` | B ≠ A1 attendu (blocs décalés) : `tranches` ne rend l'indépendance qu'aux découpages multiples de 1 024 — **limite nommée avant**, à écrire dans le régime | — |

Décision, fixée ici : bras retenu = celui qui tient C1 (ou C3) **et** ≤ 2 % à chaque M ; prédiction : `tranches` retenue,
`etroite` redéfinie non retenue (C2). Si les deux échouent : tout reste opt-in, défaut `reduite`, et la conclusion est que
l'indépendance au découpage coûte plus de 2 % avec cuBLAS — le levier suivant serait un GEMM maison à réduction fixe, hors
de cette pièce. La garde de qualité au modèle reste due avant toute bascule du défaut (C5 : les M réduits changent).
Issues qui me gêneraient : B1 False (j'ai lu la source mais pas la carte) ; B5 > 0 (tranches mortes) ; B2 à 0 partout
(la cause de E4 n'est pas un GEMM) ; C4 > 2 % à 4 096 (tranches payantes, pièce close sans bras).

Fenêtre prévue : une prise, garde de chaîne : B1-B6 (script nu, ≈ 2 min) → test E0-E3 étendu aux six lignes → débit C2 / C4
→ chaînes S1 `etroite`, `tranches`, `reduite` (3 × 1,5 min) → C5 / C6 ; ≈ 9 min. Code et tests cassants commités avant.
