# Levier 2 (rapatriement épinglé) : DÉFAUT À BASCULER — synthèse des quatre pièces, chiffre publiable — 22/09 (poste4)

instrument : lecture de `verdict-levier2-1-ids-bit-22-09` (équivalence), `-2bis-frontiere-corrige` (isolé, 300 pas ×4), `-3-abba`+`-4-abba-confirmation` (servi, 2×4+4 entrelacé) ; recalcul indépendant des médianes (0 min de carte)
commit : `main` à jour (correctif `0c17a2bf` inclus, tampon `[2,n]` contigu), `92657459` (2e passe)
régime : Coder nvfp4-qkvo-i8c, b=12, `sampler=graphe` (défaut du levier 1) + `ACVRAM_RAPATRIEMENT_EPINGLE=1` = B, défaut `rapatriement=flux` = A
scellé (poste1, confirmation chef) : ids au bit ; `trou_gpu` 10-30 µs (réfuté >100) ; ABBA B/A ≥ 1,000, 2e passe attendue [1,02;1,08] réfutée <1,01
mesuré : ids 3/3 TENU ; isolé `trou_gpu` 165,3→24,5 µs (dans la fourchette) ; **passe 1 : A méd 1536,4 · B méd 1641,5 · ratio 1,0684 ; passe 2 : A méd 1557,1 · B méd 1607,2 · ratio 1,0322** — **médiane des deux passes : A 1546,7 t/s · B 1624,3 t/s · ratio B/A = 1,050 (+5,0 %)** (recoupe le 8+8 combiné : 1,045) ; dispersion intra-bras (63/63 pour A, 74/32 pour B) < écart A/B dans les deux passes (jamais le défaut du 6f-v2) ; horloges : écart 1,33 % (passe 1) et 0,16 % (passe 2), sous le seuil 3 % les deux fois
verdict : **DÉFAUT BASCULE à `rapatriement=epingle`** — équivalence tenue, mécanisme isolé confirmé dans sa fourchette, gain servi positif et significatif sur deux passes indépendantes à horloges maîtrisées (+3,2 % à +6,8 %, médiane publiable +5,0 %), aucune régression. L'écart entre les deux passes ABBA (dispersion connue ~50 t/s du banc, notée par poste2) n'entame pas le signe ni l'ordre de grandeur — à publier avec sa fourchette, pas un point unique.
durée : 0 min de carte
suite : poste1 bascule `rapatriement=epingle` en défaut ; cellule publiée = **1 624 t/s ± (1 590-1 660)** ; contre vLLM 1 596 : **acvram devant de +1,8 %** (à confirmer par la cellule certifiée/nsys déjà prévue par poste2).
