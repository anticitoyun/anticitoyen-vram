# aym — piles MoE 30B : cause et correctif des 18 préchargements morts de l'edz définitif (poste1, 30/09, à sec)

* instrument : journaux du harnais de l'edz (`~/.cache/acvram/menus-reels-definitif/journaux/`, lignes du moteur au démarrage seulement, § 6) ; lecture du code (fichier:ligne ci-dessous) ; tests `tests/test_piles_au_chargement_aym.py`
* commit : poste1-aym (depuis main 0040b3195)
* régime : à sec (CUDA_VISIBLE_DEVICES="") ; le bras carte du test (`-k carte`) et le rejeu servi attendent la fenêtre accordée après la 0.7.16
* scellé : chaque test de défaut casse sur main (vérifié : 2/2 rouges, sources de main remises par stash) ; témoin `ACVRAM_PILES_AU_CHARGEMENT=0` = construction d'avant
* mesuré : 5 tests aym (4 à sec verts, 1 carte sauté) ; ya1 et pile-rend-le-cache verts ; `-k "moe or marlin or pile or loader or chauffe or contexte or expert or kv or plan"` : 2 590 passed, 1 failed (`test_server.py::test_kv_blocks_are_returned_after_traffic`, rouge AUSSI sur main : préexistant, hors pièce)
* verdict : cause = **piles d'experts et repack Marlin construits APRÈS le KV**, qui a déjà rempli la carte ; correctif = construction au chargement, avant le KV (même sortie, seul le moment change) + OOM du repack rattrapé et nommé
* durée : 0 (à sec)

## Cause
1. Le chargeur alloue le KV pour remplir la carte jusqu'à la marge (`acvram/engine/loader.py`, boucle `caches[i] = PagedKVCache(cfg)`).
   Le plan compte la pile en régime établi, Σ × 1 : la pile Marlin remplace la naturelle (`loader.py`, `_reserve_prefill`).
   Il ne compte pas le transitoire de construction : pile empilée, copie Marlin, concaténation w13 (≈ 2 × 324 Mio par couche sur Coder-30B, ya1).
2. Les piles se construisaient ensuite, paresseusement :
   * dans `GraphRunner._eligible`, avec graphes (`graphs.py:473-477`, appelé depuis `runner.py:743`) ;
   * ou avant la chauffe, sans graphes (ya1, `contexte.py`, `_construire_piles_avant_chauffe`).
3. `_try_build_stacks` rattrapait l'OOM de la pile naturelle (`moe.py:313-330`, boucle par expert). L'appel `_construire_marlin` restait HORS de cette garde (`moe.py:346` sur main).
4. Au journal de l'edz, avec graphes, 15 alias Coder-30B : `OutOfMemoryError` dans `_construire_marlin` → `MP.preparer_pile`. Le processus tient 31,29 Gio, il reste 47,94 Mio libres ; serveur mort.
5. Sans passer par ce chemin (3 alias : `qkv-22-09`, qwen3-vl-30b-a3b ×2) : 11 couches sur 48 tombent en « boucle par expert » (OOM de la pile naturelle rattrapé). Il ne reste rien pour la chauffe : 0 jeton tenu à 32 768.
6. Les passes rapides d'avant les tenaient, mais à 4 096 et sans graphes : peu de KV, donc de la marge. Ce n'est pas une régression prouvée, c'est un autre régime.

## Correctif
* `loader.py` (avant le `empty_cache` qui précède l'allocation du KV, après les fusions) : `construire_piles_sur_carte(layers)`,
  fonction partagée désormais avec la chauffe (`contexte.py`). Mêmes blocs (couche sur la carte, état « ? »), même
  `_try_build_stacks`, mêmes piles : la chauffe et `_eligible` trouvent les blocs décidés et ne refont rien. Journal :
  « piles d experts construites au chargement, avant le KV : N couches ». Témoin : `ACVRAM_PILES_AU_CHARGEMENT=0`.
* `moe.py:346` : `_construire_marlin` sous `try/except torch.OutOfMemoryError` → `_raison_marlin = "mémoire GPU insuffisante
  pendant le repack Marlin"`, pile naturelle gardée (le chemin des couches refusées pour sous-normales), une ligne au
  journal. Pas d'`empty_cache` dans l'`except` : la trace de l'exception tient encore les copies partielles ; le rendu
  se fait en sortie (`_rendre_le_cache_apres_la_pile`).

## Prédiction scellée pour la fenêtre (écrite AVANT la mesure)
Arbre poste1-aym, serveur sous carte.sh, graphes :
* alias : acvram-qwen3-coder-30b-a3b-qkvo-i8c-nvfp4 à 29 096, puis qwen3-vl-30b-a3b-abl-nvfp4-vision à 32 768 (§ 6 : codes et longueurs) ;
* prédit : les deux démarrent ; le journal porte « construites au chargement » ; il ne contient ni « boucle par expert » ni « repack Marlin » ;
* seuil : chauffe ≥ contexte demandé.

Issues nommées :
* (a) tenu ;
* (b) démarre mais la chauffe < contexte → le KV planifié est trop gros pour ce que les piles laissent : `_reserve_prefill` à revoir ;
* (c) OOM à l'allocation du KV → même cause, vue plus tôt : refus nommé à ajouter ;
* (d) « repack Marlin » refusé → sert, mais sur le chemin d'avant : débit à mesurer.

Équivalence : `pytest tests/test_piles_au_chargement_aym.py -k carte`, au bit.

## Recoupement (duck.ai, `revue/poste4-duckai-30-09.md` Q4, 3 modèles d'accord)
vLLM reconditionne Marlin AVANT le profilage mémoire et l'allocation du KV : l'ordre que ce correctif installe.
