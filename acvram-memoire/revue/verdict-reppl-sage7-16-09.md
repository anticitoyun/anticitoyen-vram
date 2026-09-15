# Verdict — noyau par ligne (877169c) : compteurs 0/0 tenus, PPL alpha-commun 1,0277 RÉFUTÉE (> 1,015), reconverti -k48 1,0183 RÉFUTÉ (> 1,010) ; les 25 s étaient le cache de pages

- **instrument** : `scratchpad/reppl-eval-16-09.py` = `acvram.evaluate.perplexity` (ligne de poste2 : wiki-gptq, 2048/2048, min-context 256, 4 fenêtres) + régime lu après chargement ; `ACVRAM_QA_COMPTE=1` ; `seconds` = chargement + 4 fenêtres (`evaluate.py:175/364`) ; sorties `scratchpad/reppl-sage7-16-09/`
- **commit** : arbre **travail/poste3-qa @ 877169c** (poste4, échelle globale par ligne, poste7 § 7) ; t-qa 202/206 (4 rouges hors chemin : attente `0xFE` = −448, référence float64 du fused) ; protocole 6c91a5b
- **régime** : les 4 passes NOMINAL — piles `oui` 46/46, 0 paramètre hors `cuda:0`, chemin MoE mma ; blocs quantifiés 337 641 472 (alpha-commun) / 530 579 456 (-k48, denses en nvfp4 aussi)
- **scellé** (poste7 § 7) : B alpha-commun ≤ 1,010, 0 saturé, flush ≤ 0,01 %, durée ≤ 7,2 s, réfuté > 1,015 → sonde par étape avec poste1 ; K `-k48` ≤ 1,005, réfuté > 1,010 → métrique W4A4 cb2784b. Moi : B 1,012 ± 0,004 ; flush 0,001-0,01 % ; 25 s = cache (K1 lent, K2 rapide) ; K 1,003-1,008
- **mesuré** : B **8,368142 = 1,02769** (×2 identique), flushés **0**, saturés **0**, 25,5 s puis **6,3 s** ; K **8,291296 = 1,01825** (×2 identique), 0/0, 5,9 / 5,9 s
- **verdict** : compteurs et durée **tenus** (0/0, 6,3 s ≤ 7,2) ; PPL **RÉFUTÉE deux fois** : alpha-commun 1,0277 > 1,015 → sonde par étape (poste1) ; -k48 1,0183 > 1,010 → cb2784b entre. Mes prédictions 1 et 4 réfutées, 2 et 3 tenues (K1 rapide parce que `-k48` venait d'être écrit à 08:08 : déjà en cache)

## 1. Ce que les quatre chiffres disent ensemble

```
converti        noyau                      PPL      ratio    flush        sat      source
alpha-commun    k=0 (ancien)               8,3941   1,0309   0,066 %      0        ce matin
alpha-commun    k_x=4 seul                 8,3004   1,0194   0,066 % (act) 34 286  ce matin
alpha-commun    par ligne (877169c)        8,3681   1,0277   0            0        ici
alpha-commun    W4A16 (MMA=0)              8,215    1,0089   —            —        poste2, mma0-confondant
-k48            par ligne                  8,2913   1,0183   0            0        ici
livrable AWQ    W4A16, tables int8         8,1275   0,9981   —            —        ppl-finale
```
- **Les blocs à zéro n'étaient pas la cause** : à flush 0 et saturation 0 la PPL ne gagne que 0,003 sur l'ancien noyau (1,031 → 1,028), alors que le W4A16 du même converti est à 1,009. L'écart W4A4/W4A16 ≈ **+0,019** survit à une quantification d'activation sans aucun bloc perdu : il vient de l'E2M1 lui-même sur `x/s` et `act/s_d` (3 sites, blocs de 16, échelle E4M3 à 3 bits de mantisse), pas d'un défaut de plancher.
- **k_x=4 seul faisait mieux (1,019) avec 34 k blocs saturés** : la saturation écrêtait des valeurs aberrantes de `x/s` et la PPL y gagnait — un indice que la queue haute de `x/s` (après division AWQ) coûte plus que la queue basse ; à regarder dans la sonde par étape (distribution de `x/s` par expert, pas seulement amax).
- **-k48 à 1,018 avec le noyau propre** : la recette (AWQ nvfp4 seul, denses en nvfp4) est meilleure que l'alpha-commun de 0,009 mais loin du 0,998 du livrable W4A16 : le coût est le W4A4, pas la recette — la métrique W4A4 (cb2784b) devient nécessaire pour juger la reconversion, comme scellé.
- **Durée** : 25 s = premier chargement d'un modèle hors cache de pages (17 Go depuis le SSD), 6 s ensuite ; le noyau n'y est pour rien. Le scellé « ≤ 7,2 s » se lit sur une passe chaude.

## 2. Conséquence pour la file
Rien ne bouge sur la carte : sonde par étape (poste1 + poste4) sur l'alpha-commun, et la question « W4A4 au décodage à ce prix de PPL » revient à poste7 avant le pas b=12 et le duel (le duel W4A16 reste possible : livrable 0,998, régime à vérifier).
