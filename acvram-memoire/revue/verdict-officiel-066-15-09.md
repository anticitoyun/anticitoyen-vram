# Re-tampon de l'officiel 0.6.6 (narrow + route+pack + MMA godet 12), durée corrigée

poste3, 15/09/2026, 21:34-21:42 (unité officiel066-poste3), une session ABAB.
Ordre : poste7 § 8 / chef. Arbre : **travail/poste3 figé à 66c7532**
(= main a385d9b, v0.6.5 + `energie.py` b11b8a3 corrigé), chemin d'import au
JSON ; carte vide au départ (42 °C).

## En-tête (REGLES §3)

    instrument   energie.py CORRIGÉ (durée relevée avant le join) ; garde
                 « durées Energie/hôte divergent > 0,05 s » active, jamais
                 déclenchée (l'affichage du chrono hôte manquait dans le JSON,
                 corrigé après coup ; les fenêtres à x,94 s montrent la fin
                 de l'arrondi au tic entier) ; cartes [0] ; -pl 400 ; horloge
                 libre ; T 37-40 °C avant, 51-53 pendant
    protocole    c6377d5 : rondes ctx 2048, invite 256, ≥ 20 s, repos 30 s ;
                 A = ACVRAM_NARROW_GEMM=0, B = 1 ; route+pack et MMA godet
                 12 des deux côtés (défauts v0.6.5)
    données      scratchpad/officiel-066-15-09/rondes-{A1,B1,A2,B2}.json

## Résultat

    bras      pas ms          t/s      J/jeton          W      fenêtre s
    A (0)     13,956 / 13,982 784,3    0,5079 / 0,5101  398-399 24,94 / 24,99
    B (1)     11,718 / 11,716 934,2    0,4262 / 0,4267  398     20,94 / 20,94
    B/A       −16,1 %         +19,1 %  −16,2 %

**Chiffre officiel 0.6.6 (ce protocole, durée corrigée) : 11,72 ms /
934 t/s / 0,426 J/jeton / 398 W ; témoin A 13,97 ms / 784 t/s / 0,509 J.**
Correction du biais sur les valeurs d'hier : A 14,01 → 13,97 (excès 0,05-
0,10 s), B 11,77 → 11,72 (excès ≈ 0,09 s) — petites ici, par chance de
fraction ; c6377d5 (16,25 / 17,37) reste annoté « ≤ +4 % ».

## Colonne « décodage pur 22 s » : ABSENTE

Trois tentatives (ctx 2304 / 2560 à 22 s, ctx 2048 à 16 s) ont échoué sur
ma garde « lot constant » : une séquence sur 12 quitte `engine.running`
avant la fin de fenêtre, dès ~1 300 pas, indépendamment du `max_tokens`
(2 044-2 300) — ce n'est donc pas la longueur ; cause non identifiée ce
soir (limite de blocs KV planifiée sur 2 048 ? séquence exilée ?). Pas de
chiffre pur reconstruit. Le 9,52 pur de poste4 (banc court) reste seul,
annoté « durée biaisée ≤ +4 % ». Première commande de reprise dans le
carnet.
