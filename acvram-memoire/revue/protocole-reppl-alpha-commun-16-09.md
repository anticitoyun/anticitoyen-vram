# Protocole — bloc 2 : re-PPL du converti alpha-commun après le correctif quant_act (k_x=4, k_act=8), compteurs

poste3, 16/09/2026, avant mesure. Ordre : chef (ETAT.md l. 2), scellés poste7
`poste7-glm-pile-correctif-16-09` § 1.5 (iii) et § 6.3. Arbre : travail/poste3
figé au commit de ce protocole (code `acvram/` = main **e84a43d**, correctif
poste4 01c48ef fusionné, t-qa 210/210). Extension compilée hors verrou
(ninja, 85 s, cache `kernels-84a817c8c4ce`, empreinte du source contrôlée au
chargement `kernels/__init__.py:376-386`).

## Montage
`acvram eval` (ligne de poste2, `verdict-glm-mma0-confondant-16-09`) : corpus
`wiki-gptq.txt`, `--window 2048 --stride 2048 --min-context 256`, 4 fenêtres,
NOMINAL MMA=1 (pas d'`ACVRAM_MOE_MMA=0`), `ACVRAM_QA_COMPTE=1` (compteurs
imprimés à la sortie, `model.py:1530`). Référence bf16 8,1427
(`verdict-glm-ppl-finale-15-09`). Deux bras : **B** = alpha-commun
`GLM-4.7-Flash-srcbf16-nvfp4` (27 661 `act_scale`) ; **A** = témoin experts
sans AWQ `-avant-alpha-experts-fix` (9 617). Régime lu dans la sortie
(NOMINAL, piles_ok) sinon la mesure ne dit rien du correctif.

## Scellé (poste7)
B : ratio **≤ 1,010** (avant correctif 1,03086 en MMA=1 ; 1,008879 en W4A16) ;
compteurs **mis à zéro = 0 et saturés = 0** ; un compteur > 0 → k de ce site
bouge d'un cran avant toute autre mesure. A : ≤ 1,022, attendu 1,015-1,020
(avant : 1,02025 MMA=1, 1,022388 W4A16).

## Ma prédiction (scellée)
B = **1,009 ± 0,002** (le W4A4 corrigé retrouve le W4A16 à ≤ 0,002 près ; au
mieux 1,007, réfuté > 1,011 → autre chose reste sur le chemin pile, sonde § 3
de poste7). Compteurs B : mis à zéro **0**, saturés **0** ; si mis à zéro > 0 il
est < 0,01 % des blocs (queue basse au-delà des 16 jetons d'poste1) et c'est
k_act qui monte, pas k_x. A = **1,020 ± 0,002** (inchangé : sans table, `x`
n'est pas divisé, l'amplitude est déjà au-dessus du plancher) ; si A baisse
de > 0,004, le correctif agissait aussi sur `act/s_d` sans AWQ — bon signe,
à dire. Durée : 2 × (chargement 20 Go + 4 fenêtres) ≈ 6-8 min, unité
`reppl-alpha-poste3`.
