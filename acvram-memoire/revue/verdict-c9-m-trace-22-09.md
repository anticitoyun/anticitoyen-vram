# C9 M-TRACE — TENU, mieux que prédit — 22/09 (Manon)

* instrument : `outils/gpu/mesure/c9-m-trace.py` (proxy **Qwen3-Coder-30B-A3B-nvfp4**, 128 experts top-8, 48 couches — le 119B réel n'a pas de manifeste acvram, jamais converti au format natif, donc jamais chargeable directement par cet instrument ; le protocole d'Océane prévoit explicitement ce proxy « tant que le 119B ne charge pas »), sous `carte.sh`, régime dégradé demandé (graphes off, `ACVRAM_DISABLE_CUDA_GRAPHS=1`, sampler=lent, nécessaire pour synchroniser la trace)
* commit : main à jour, worktree manon-w-21-09
* régime : capacité C = 52/128 (41 %), chargement 37 s, 20 requêtes, 7 707 jetons de trace au total
* scellé (Océane, § 3.2) : politique `pin` (appris moitié 1, jugé moitié 2) contre témoin `lru` ; prédit h_pin(52) global 0,55 ± 0,10 ; réfuté sous 0,45 ; > 0,65 = bornes S1/S3 à recalculer
* mesuré : **h_pin = 0,733** (couches min 0,56, max 0,80) ; h_lru = 0,869 (témoin, au-dessus, attendu) ; uniforme = 0,41 (plancher de référence)
* verdict : **TENU, nettement au-dessus de la prédiction** (0,733 contre 0,55 ± 0,10, hors de la marge haute 0,65) — les bornes S1/S3 de la conception `oceane-c9-conception-21-09` doivent être recalculées à cette valeur, plus favorable qu'anticipé pour un cache d'experts épinglé à C=52. M-HÔTE (`verdict-c9-m-hote-22-09`) a déjà réfuté/arrêté le chantier C9 sur le poste (a) (l'hôte scalaire trop lent) — cette mesure reste informative pour une future itération avec un noyau AVX-512.
* durée : ~2 min de carte (chargement 37 s + 7 707 jetons sans graphes)

## Suite
C9 clos pour aujourd'hui (arrêt (a) sur M-HÔTE, indépendant de ce bon résultat de cache). TRT-LLM ensuite.
