# P3 (4) duel Qwen3-VL-30B (TTFT, J) — 21/09

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-4.sh` (fichier suivi, fusion 73a67474)
* commit : 3ed4e807 (worktree poste2)
* régime : NOMINAL, graphes=on(hybrides≤4), kv=int8, pipeline=1, eco=2700(2692), vision=bf16(eager), mrope=[24,20,20](interleaved), deepstack=3, masque_images=causal, ctx_tenu=non-chauffe (jamais atteint la chauffe pleine — cassé avant)
* scellé : TTFT ≤ 0,094 s (incertain), J ≤ 40 (tenu, référence antérieure)
* mesuré : rien — acvram meurt au chargement, après la chauffe de contexte (4096/4096 tenus, 4,9 s) et juste avant/pendant la capture de graphe CUDA
* verdict : **ÉCHEC, acvram mort au chargement** — `serveur-duel-30b.log` : `RuntimeError: The size of tensor a (256) must match the size of tensor b (257) at non-singleton dimension 0`, aucun traceback capturé (le serveur n'affiche que `str(e)`). Site le plus probable par recherche du code : capture/`warm_graphs` (`acvram/engine/runner.py:2162`), à vérifier par qui corrige — je ne l'affirme pas sans traceback. Décalage de 1 (256 vs 257) évoque un token image/padding en trop ou en moins pour ce modèle vision (deepstack=3, mrope). `-lgc 2700` posé puis `-rgc` relâché proprement (225→1140 MHz), verrou pris 61 s et rendu.
* durée : 61 s, prévu ≤ 1800 s

## Suite (hors mon domaine)
Pas de traceback exploitable pour localiser le site exact — à qui corrige : faire échouer la même prise avec `ACVRAM_TRACEBACK=1` pour obtenir la pile complète avant de chercher plus loin.
