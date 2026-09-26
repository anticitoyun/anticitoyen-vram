# Protocole — bloc 2 bis : re-PPL alpha-commun avec le noyau par ligne (poste7 § 7), contrôle des 25 s, PPL du reconverti -k48

poste3, 16/09/2026, avant mesure. Ordre : chef (poste7 § 7, main d1b103a). Arbre :
**travail/poste3-qa @ 877169c** (branche poste4 : échelle globale par ligne
`g_r = amax_r/2688`, blocs `(amax_blk/amax_r)×448`, `__fmul_rn`, compteurs non
nuls/flushés/saturés ; t-qa 202/206, 4 rouges hors chemin : attente `0xFE`
= −448 dans `test_quant_act_echelle.py:75`, référence float64 du fused non
mise à jour, fused coupé par défaut). Instrument : `scratchpad/reppl-eval-16-09.py`
= `acvram.evaluate.perplexity` avec la ligne de poste2 (wiki-gptq, 2048/2048,
min-context 256, 4 fenêtres) + régime lu après chargement (piles par état,
paramètres par appareil → exil, chemin MoE). Référence bf16 8,1427.

## Scellés (poste7 § 7, relayés)
B alpha-commun (`GLM-4.7-Flash-srcbf16-nvfp4`) : ratio **≤ 1,010**, **0 saturé**,
**flush ≤ 0,01 %**, **durée ≤ 7,2 s** (4 fenêtres) ; réfuté > 1,015 → sonde par
étape avec poste1. K reconverti `-k48` (poste2) : **≤ 1,005**, NOMINAL, 0 exilé,
piles_ok. Contrôle des 25 s : même bras deux fois de suite.

## Mes prédictions (scellées)
1. **B = 1,012 ± 0,004** : k_x=4 seul rendait 1,019 avec 34 k blocs saturés ; la
   ligne ôte la saturation et donne au site down son échelle → sous 1,015, mais
   je ne crois pas ≤ 1,010 : le W4A16 est à 1,0089 et le W4A4 ajoute son bruit
   E2M1 sur trois sites. Réfuté si ≤ 1,010 (tant mieux) ou > 1,016.
2. Compteurs B : saturés **0 exactement** (borne par construction) ; flushés
   **0,001-0,01 %** (blocs < 2⁻⁹/448 du max de ligne). Réfuté si > 0,01 %.
3. **Les 25 s sont le cache de pages, pas le noyau** : B1 6-8 s (fichiers lus il
   y a 30 min), B2 6-7 s ; **K1 20-27 s** (`-k48` jamais lu), **K2 6-7 s**. Réfuté
   si K1 ≈ K2 ≈ 6 s (alors les 25 s venaient de k=(4,8)) ou si B1 ≈ 25 s.
4. **K = 1,003-1,008** : recette AWQ nvfp4 seul (0,998 en W4A16 avec les tables
   int8), + bruit W4A4 → au bord du scellé ; réfuté < 1,003 ou > 1,010.
Durée ≈ 4 × (chargement + fenêtres) ≈ 6-8 min, unité `reppl-sage7-poste3`.
