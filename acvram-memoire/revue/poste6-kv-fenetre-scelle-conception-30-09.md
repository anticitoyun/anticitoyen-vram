# kv31b, levier 2 — scellé de conception : cache KV borné à la fenêtre glissante (Gemma 3/4, Mistral SWA) — poste6, 30/09, à sec, mise en œuvre à l'ordre de chef

État : CONCEPTION, rien de codé. Chiffres du dépôt (fichier et ligne) et de la note duck.ai Q3 de poste4 (relayée par chef 30/09 :
vLLM et llama.cpp ne gardent QUE la fenêtre pour les couches à fenêtre glissante, sauf `--swa-full` ; Gemma 4 : 0,82 Mio/jeton bf16,
0,41 int8 « sourcé par la config HF » — notre manifeste donne 60 couches × 16 têtes KV × 256 × 2 × 2 o = 0,94 Mio bf16, 0,47 int8 hors
échelles : écart de 13 % à trancher sur la config HF avant tout chiffre publié ; les rapports ci-dessous n'en dépendent pas).

## 1. Ce que fait le moteur aujourd'hui (inventaire)
* Un `PagedKVCache` PAR COUCHE, même `KVCacheConfig` (kvcache.py:438 « le cache d'une seule couche », `num_blocks` identique) : `total_bytes` =
  bloc × num_blocks × num_layers (kvcache.py:371). Un seul `BlockAllocator` (runner.py:705), UNE liste `seq.blocks` par séquence, la même
  table de blocs pour toutes les couches (runner.py:1634, 2134 ; attention.py:505, 578 `batch.block_tables[i]`).
* La fenêtre glissante n'est qu'un MASQUE : `Attention.window` (attention.py:143), masque de préfill (layers.py:1108), noyaux de décodage
  `lo = max(0, slen − window)` (acvram_kernels.cu:994, 1224) — ils lisent `tables[b][pos // 16]` pour pos ∈ [lo, slen) : ils ne lisent
  déjà QUE la fenêtre. La mémoire, elle, garde tout.
* Budget : `kv_bytes_per_token` × `max_model_len` sur toutes les couches à KV (config.py:315, loader `_kv_plancher`) — gemma-4-31B à
  32 768 : 15,12 Gio int8 (verdict kv31b). Préfill de l'attention d'un seul tenant (q_len = T) ; MLP par tranches (a5v).
* Préfixe partagé par hachage de bloc (kvcache.py:53), déversoir hôte `spill_cb` (runner.py:735), recul spéculatif (`rollback_hybrid`,
  runner.py:2095), graphes CUDA capturés sur les tables de blocs.

## 2. Conception : deux groupes de KV, tables en anneau pour les couches à fenêtre
* **Groupes** : `pleine` (full_attention : 10/60 sur gemma-4-31B) et `fenetre` (sliding_attention : 50/60, `sliding_window` = 1 024).
  Un allocateur par groupe, une liste `seq.blocs_par_groupe`, une table par groupe passée à la couche selon son type.
* **Anneau** : une couche à fenêtre garde R blocs par séquence, R = ⌈(fenêtre + morceau_de_préfill) / 16⌉ + 1 ; la table logique
  (longueur ⌈slen/16⌉) pointe sur le slot physique `(bloc_logique mod R)` : **les noyaux ne changent pas** (ils indexent `tables[b][pos//16]`
  pour pos ≥ lo, positions toutes dans les R derniers blocs par construction). Le +1 protège le bloc en cours d'écriture.
* **Précondition** : préfill de l'ATTENTION par morceaux (≤ morceau, 1 024-4 096) pour ces couches — sinon q_len = T impose R ≈ T/16 et rien
  n'est gagné au préfill. Deux voies : (a) tranches d'attention (le morceau écrit ses K/V dans l'anneau puis lit fenêtre + morceau) ; (b) SDPA
  sur K/V frais du morceau + fenêtre lue dans l'anneau (HF). (a) réutilise `forward_tranches` (284 b, model.py:271) : à préférer.
* **Budget** (config.py, nouvelle `kv_bytes_pour_sequence(n, kv_bits, fmt)`) = Σ_pleine par_couche × n + Σ_fenetre par_couche × R × 16 ;
  `_kv_plancher` et `kv_max_tokens` du planificateur (tiering.py:476-487) passent par elle ; le groupe `fenetre` se budgète PAR SÉQUENCE
  (max_seqs × R blocs), le groupe `pleine` reste paginé partagé.
* **Hors périmètre du groupe fenêtre** : cache de préfixe (un anneau n'est pas adressable par hachage — désactivé pour ce groupe, gardé pour
  `pleine`) ; déversoir hôte (rien à déverser : la fenêtre est toujours chaude) ; puits k8v4 (positions 0-15 hors fenêtre après 1 024 jetons :
  HF Gemma n'en garde pas sur les couches locales — à vérifier au bit).
* **Recul spéculatif** : n_rejetés < fenêtre, l'anneau revient en arrière comme une table pleine (positions, pas de recyclage).
* **Graphes CUDA** : tables de forme fixe par groupe (`gather_fixed`, kvcache.py:716) — R constant, capturable.

## 3. Chiffres (gemma-4-31B, int8 8 256 o/couche/jeton, 32 768 jetons, une séquence)
| terme | aujourd'hui | conception (morceau 1 024) | (morceau 4 096) |
|---|---|---|---|
| couches pleines (10) | 2,52 Gio | 2,52 | 2,52 |
| couches à fenêtre (50) | 12,60 Gio | 50 × 129 blocs × 16 × 8 256 = 0,79 | 50 × 321 × 16 × 8 256 = 1,97 |
| **KV d'une séquence** | **15,12** | **3,31** (÷4,6) | **4,49** (÷3,4) |
| à 33,6 Gio libres : poids 19 + KV + réserve 5,0 + marge 1,7 | 40,8 → 33/60 MLP exilés | 29,0 → **0 exilé** | 30,2 → 0 exilé |
| à 28,1 (edz) | refus | 29,0 → 1-5 MLP exilés, sert | 30,2 → 11 exilés, sert |
| kimi (34 816) à 33,6 | refus | sert | sert |
b = 8 séquences : groupe fenêtre 8 × 0,79 = 6,3 Gio (morceau 1 024), groupe pleine paginé partagé.

## 4. Prédiction et seuils pour la prise sur carte (à écrire AVANT, ici les gabarits)
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| S1 | logits d'un préfill + 64 pas de décodage, prompt de 6 000 jetons (> 5 fenêtres), anneau contre stockage plein | **identiques au bit** (même masque, mêmes positions lues) | un seul logit diffère → défaut de rotation ou de R |
| S2 | KV alloué (`nvidia-smi` après chauffe) gemma-4-31B, 32 768, b=1 | 3,3-4,5 Gio contre 15,1 | > 6 Gio |
| S3 | MLP exilés à carte seule | 0/60 (33 aujourd'hui) | > 5 |
| S4 | décodage b=1 | ≥ 5× le débit actuel exilé (la falaise disparaît) ; ≈ débit d'un dense 31B nvfp4 résident | < 3× |
| S5 | PPL wiki-gptq 2048/2048 | égale à la valeur pleine à 1e-4 près (S1 au bit ⇒ PPL identique) | écart > 1e-3 |
| S6 | préfill 32 768 par morceaux | temps ≤ 1,3 × le préfill d'un seul tenant (perte : morceaux) | > 1,5× |
Issues nommées : (a) le préfill par morceaux de l'attention n'existe pas encore → c'est le gros du travail, pas l'anneau ; (b) puits k8v4 sur
couches locales : sortie change au bit → à documenter, pas à cacher ; (c) recul spéculatif > fenêtre (impossible avec ngram : n ≤ 8) ; (d) images
(masque `ouvert` bidirectionnel, attention.py:140) : le `or_mask` peut regarder AVANT la fenêtre → groupe fenêtre refusé aux lots à images ou
R élargi à l'image ; (e) Mistral/Devstral SWA = même mécanique, gain différent (fenêtre 4 096-32 768) : à chiffrer par modèle avant d'annoncer.

## 5. Ordre de mise en œuvre proposé (chaque étape avec son test au bit)
1. `ModelSpec.kv_bytes_pour_sequence` + `_kv_plancher`/tiering par groupes (à sec, chiffres de la table 3 en test).
2. Groupes d'allocation et tables par groupe dans le runner, R fixe, noyaux inchangés (à sec : tiny modèle, S1 au bit avec fenêtre 4, bloc 2).
3. Préfill de l'attention par morceaux pour le groupe fenêtre (S1 au bit sur un prompt > 5 fenêtres).
4. Carte : S2-S6 sur gemma-4-31B à 32 768, puis kimi 34 816. Défaut : OFF (`ACVRAM_KV_FENETRE=1`) tant que S1-S6 ne sont pas tenus.

## 6. Versé après coup (chef, duck.ai 30/09 soir, accord des 3 modèles)
* Q1 : le préremplissage par morceaux de vLLM N'EST PAS au bit (ordre de réduction du softmax en ligne sur les morceaux de KV), et l'écart
  peut changer un jeton échantillonné. Chez nous la règle reste le bit : découpe par LIGNES DE REQUÊTE, chaque ligne réduisant sur le même
  KV, dans le même ordre, avec le même noyau (ce qui a rendu le masque par blocs de lignes d'poste1 au bit, 72/72). L'étape 1 suit cette
  règle (`poste6-prefill-morceaux-verdict-30-09.md`) ; le jouet ajoute : mêmes chemins à petit M et même côté du seuil de fusion gate/up.
* Q2 : llama.cpp garde un anneau strict et retombe en préremplissage complet quand l'état n'est pas reconstructible (recul au-delà de la
  fenêtre) ; vLLM ne réutilise pas le préfixe hors fenêtre pour les couches à fenêtre glissante — comme prévu en § 2 (préfixe partagé hors
  du groupe fenêtre ; recul spéculatif < fenêtre).
* Écart 0,82 (duck.ai) contre 0,94 Mio/jeton (manifeste) : le manifeste (le code) fait foi, l'écart est noté.
