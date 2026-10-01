# 9hm — VMM (cuMemCreate/cuMemMap) pour un « outil de dépassement de VRAM » : 5v7, KV extensible à la vAttention, et ce que ACVRAM_ALLOC_EXTENSIBLE fait déjà (poste6, 01/10, à sec, ordre chef)

instrument : lecture du code (fichier:ligne), des verdicts (5v7, profils a8 et coder-2), de la note duck.ai de poste4 (`poste4-duckai-01-10-uvm.md`) et de vAttention (arXiv 2405.04437, HTML) ; aucune mesure
commit : 9bbfa3c9e (poste6-g9m = main 0f7a54062 + revues)
régime : à sec, aucune carte (poste2 275 en cours) ; PyTorch 2.14.0+cu130, pilote nvidia-open 595.91.07 (modules ouverts, MIT/GPL)
scellé : pas de mesure dans cette pièce ; une prédiction scellée pour la seule mesure proposée (§ 3)
mesuré : rien
verdict : **(1) 5v7 : non applicable — déjà résolu autrement** (la compaction d'poste1 rend les 7,4 Gio ; VMM garde la même page de 2 Mio, il ne changerait rien) ; **(2) KV extensible à la vAttention : non applicable** (notre KV est alloué d'un bloc au plan, sans fragmentation ni allocation à l'exécution ; le gain de vAttention est celui d'un noyau non paginé contre le noyau paginé lent de vLLM — notre noyau paginé pèse 10,6 % du pas à b = 1 et 5 % à b = 12 : plafond ×1,12 / ×1,05, et l'article ne tranche pas 1,23 contre 1,36 parce que ce sont deux témoins différents) ; **(3) VMM à la main n'apporte rien de plus que `expandable_segments`** pour le dépassement : PyTorch le fait DÉJÀ en VMM (`cuMemCreate`/`cuMemMap`, pages de 2 Mio) ; ce qui reste, c'est l'épreuve de la capture de graphes sous ce réglage — mesure de 10 min proposée, prédiction scellée
durée : 0 min de carte

## Ce qu'est « VMM », et ce que nous avons déjà
* API pilote `cuMemCreate` (mémoire physique) / `cuMemAddressReserve` / `cuMemMap` : virtuel et physique découplés. Granularité : **multiples de
  2 Mio** sur les API CUDA (vAttention, table 3 : ses pages de 64-256 Kio viennent d'une **extension du pilote** que les auteurs ont écrite —
  possible chez nous avec nvidia-open 595, mais c'est un module noyau à patcher, hors de portée d'un outil utilisateur).
* PyTorch `expandable_segments:True` = exactement ces API (`torch/include/c10/cuda/driver_api.h` : `cuMemCreate`, `cuMemMap` ;
  `CUDACachingAllocator.h:70,289`). Chez nous : `ACVRAM_ALLOC_EXTENSIBLE=1` → `acvram/__init__.py:89-90` pose `PYTORCH_CUDA_ALLOC_CONF`,
  déclaré `regime.py:159`, liste CLI `cli.py:56`. Hors défaut pour la raison écrite `acvram/__init__.py:30-48` : tension avec la capture de
  graphes CUDA (adresses figées) — et `graphs.py:834-842` nomme l'allocateur quand une capture est perdue sous ce réglage.
* UVM/HMM (poste4, Q1-Q3) : pris en charge sur sm_120 avec nvidia-open, mais aucun moteur ne l'emploie pour décharger — tous déchargent
  explicitement, comme nous (`loader.py` exil par couche/expert, `layers.py:335` `ExpertPool`, RAM épinglée). Rien à reprendre.

## 1. Les 7,4 Gio de segments de 2 Mio du 5v7
`poste1-5v7-scelle-30-09` : les experts de Coder-30B (0,75 Mio) vivent dans le **bassin des petits blocs** (segments de 2 Mio) avec des petits
tenseurs qui survivent à la pile (`global_scale`…) ; un segment n'est rendu que vide → 7,42 Gio réservés pour 0,14 alloués. Correctif
`moe.py` `_compacter_survivants` (regrouper les survivants d'une couche dans UN tampon, par vues, avant de rendre les originaux) : petits blocs
**0,31 / 0,02 Gio**, 13,15 Gio libres au lieu de 6,04, Coder-30B sert à 29 096, **au bit** (témoin sha256 égal). Résolu.
Et VMM ? Sous `expandable_segments`, le bassin des petits blocs est un segment extensible fait de **pages de 2 Mio** (la granularité de
`cuMemCreate`) : un survivant de 96 Kio épingle sa page de 2 Mio exactement comme il épinglait son segment. VMM ne libère pas un morceau de
page ; seule une page plus petite (extension pilote de vAttention) ou la compaction (faite) le peut. **Non applicable, et plus nécessaire.**

## 2. Un KV extensible à la vAttention contre notre KV paginé
* Chez nous : `memory/kvcache.py:37` `BLOCK_SIZE = 16` ; `kvcache.py:444-447` : le cache K et V de chaque couche est **UN `torch.zeros`**
  dimensionné par le plan (`cfg.num_blocks`, `loader.py` budget KV borné par la VRAM libre) ; `kvcache.py:53-110` `BlockAllocator` distribue des
  indices de blocs dans ce tampon fixe. Aucune allocation CUDA à l'exécution, donc **aucune fragmentation à l'exécution** : le problème que
  vAttention résout (PagedAttention de vLLM allouant des blocs physiques à la demande, fragmentés) n'existe pas ici — notre « fragmentation »
  est celle du PLAN (plancher d'une séquence, `fenêtre qui tient`, kv31b), pas de l'allocateur.
* Le gain de vAttention est un gain de NOYAU : « vLLM's PagedAttention kernel is up to 2.8× slower than FlashAttention-2 », « PagedAttention-based
  prefill kernel up to 37 % slower (FA2), 42 % (FlashInfer) », décode 12 % ; d'où « up to 1.23× » (abstract, contre FA2/FlashInfer paginés) et
  « up to 1.99× » (contre vLLM) ; le 1,36 est une des colonnes contre un autre témoin — ce n'est pas « non tranché », ce sont des témoins
  différents. Chez nous : préfill déjà **non paginé** (attention.py:502-507 : à `offset == 0`, K/V frais contigus → flash/SDPA) ; décodage par
  notre noyau paginé Triton (`kernels/attn_paginee.py:1-12`, une lecture par page pour les n_rep têtes GQA, poste E), mesuré :
  `verdict-a8-nvfp4-14-09:30-36` **9,6 % + 1,0 % du pas à b = 1** (`paged_attn_partial` + `reduce`), `verdict-profil-coder-2-17-09:8`
  **5 % à b = 12**. Un noyau non paginé parfait rendrait au plus ×1,12 (b = 1) / ×1,05 (b = 12), et un KV virtuellement contigu coûte en plus
  la réservation d'adresses par séquence (max_model_len × séquences) : **non applicable**, gain plafonné sous le bruit de nos cellules (2 σ ≈ 3 %).

## 3. Ce que VMM à la main apporterait de plus que `expandable_segments`
Rien pour le dépassement de VRAM : le « dépassement » vers l'hôte, c'est l'exil (poids) et l'étage hôte du KV (`kvcache.py:379` `HostKVPool`),
tous deux explicites, comme chez tous les moteurs (poste4 Q3). VMM ne mappe que de la mémoire physique du GPU ; la mémoire hôte se mappe
déjà en zéro-copie (`cudaHostAlloc` mapped, pds (1) « froids en RAM épinglée zéro-copie »). Ce que `expandable_segments` donne et que nous
n'exploitons pas, c'est la fin de la fragmentation ENTRE segments de l'allocateur (A8 du 14/09 : OOM deux fois, allocateur par défaut puis
extensible — `echec-a8-modele-ne-tient-plus-14-09` : la capacité réelle manquait, pas la fragmentation). Le seul point ouvert est celui de
`acvram/__init__.py:30-48` : la capture de graphes sous segments extensibles n'a jamais été éprouvée sur nos deux cartes.
**Mesure proposée (10 min, à l'ordre de chef)** : Coder-30B qkvo-i8c à 29 096 (le cas 5v7), `ACVRAM_ALLOC_EXTENSIBLE=0` puis `=1`, serveur
neuf, chauffe, 8 jetons ; lire le régime et les libres après piles.
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| V1 | graphes capturés (`graphes=on`, `captures` > 0) sous `=1` | **on** — la capture alloue avant de figer, pas pendant | `graphes=off` avec `expandable_segments` nommé : l'opt-out de `__init__.py` est confirmé, on ferme |
| V2 | libres après piles, `=1` contre `=0` | ±0,3 Gio (la compaction a déjà vidé les segments) | > +1 Gio : il restait de la fragmentation inter-segments à prendre |
| V3 | chauffe tenue | 29 096 dans les deux bras | une seule tenue |
| V4 | sortie 8 jetons | sha256 égal dans les deux bras (allocateur ≠ numérique) | différent : à ouvrir, ce serait un bogue |
Issues : (a) V1 faux → VMM/extensible clos pour nous tant que les graphes sont le défaut ; (b) V1 et V2 tenus → `expandable_segments` peut
devenir défaut après une suite GPU verte (gain : robustesse aux OOM de fragmentation, pas de VRAM en plus) ; (c) V2 > +1 Gio → une pièce
« fragmentation inter-segments » s'ouvre, chiffrée.

## Reste
* Pour l'utilisatrice : l'outil de dépassement de VRAM existe, c'est l'exil (couches, experts, KV hôte) : UVM/HMM et VMM n'y ajoutent rien
  que les moteurs n'aient refusé ; la seule question ouverte est `expandable_segments` par défaut (V1-V4).
* 5v7 : clos par la compaction ; vAttention : non applicable (noyau paginé déjà à 5-11 % du pas).
