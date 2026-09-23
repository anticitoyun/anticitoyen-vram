# Levier 1 (sampler=graphe) : DÉFAUT À BASCULER — synthèse des trois pièces de Manon, 22/09 (Laurine)

instrument : lecture de `verdict-frontiere-1-ids-bit-graphe-22-09` (équivalence), `verdict-frontiere-2-graphe-22-09` (isolé, 300 pas ×4), `verdict-abba-graphe-22-09` (servi, 4+4 entrelacé) ; 0 min de carte
commit : b0c25910 (frontières), 3a925976 (ABBA)
régime : Coder nvfp4-qkvo-i8c, b=12, `ACVRAM_SAMPLER_GRAPHE=1` = B, `sampler=lent` (défaut actuel) = A ; les trois pièces sur le même levier, deux instruments indépendants (isolé/servi)
mesuré : équivalence 2/2 (b=1, b=12, ids+logprobs au bit) ; isolé 300 pas ×4 : −53,95 µs, A0≈A1 et B0≈B1 (pas de dérive) ; servi 4+4 entrelacé : B/A=1,0124, B ≥ A sur les 4 paires
verdict : **DÉFAUT BASCULE à `sampler=graphe`** — les trois pièces convergent (ids intacts, gain isolé dans la fourchette prédite, gain servi positif sans coût mesuré nulle part) ; aucune régression trouvée. Réserve nommée, pas bloquante : l'instrument `horloge_med` de `chaine-sampler-abba.sh` rend une valeur non plausible (210 MHz constant, artefact de tri déjà signalé par Manon) — n'invalide pas ce verdict (identique des deux côtés) mais à corriger avant de s'y fier pour un futur rejet sur ce même script.
durée : 0 min de carte
suite : Océane bascule `ACVRAM_SAMPLER_GRAPHE=1` en défaut (nom de variable à retourner en négatif, ex. `ACVRAM_SAMPLER_LENT` opt-out, pour ne pas laisser un ancien défaut orphelin) ; corriger `horloge()` dans `chaine-sampler-abba.sh` avant sa prochaine cellule A/B ; scellé E ensuite (déjà annoncé par Manon).
