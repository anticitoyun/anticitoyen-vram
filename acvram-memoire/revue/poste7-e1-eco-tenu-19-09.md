# poste7 — E1 tenu : le mode éco `-lgc` met acvram devant llama.cpp en énergie ET en vitesse à b=12 ; ce qui manque avant de le revendiquer (19/09, 17 h 30)

Source : `verdict-eco-lgc-b12-19-09` (poste2, `poste2` 5eb636c, prise 16:58-17:15, arbre `3608e56`, aucune variable) ; `poste7-nuit-sens2-19-09` § 2.

## 1. Mesuré (harnais égal b=12, ABAB avec le bras libre, J net)

| réglage | t/s | J net | vs libre (1 339 / 0,2305) | vs llama.cpp (1 066 / 0,2136) |
|---|---|---|---|---|
| libre (défaut) | 1 339 (p1 1 339, p2 1 367, p3 1 193 dérive) | 0,2305 | — | +26 % t/s, **+8 % J** |
| `-lgc 2700` | 1 339 | **0,2071** | −0 %, **−10 %** | +26 %, **−3 %** |
| `-lgc 2400` | 1 227 | 0,1856 | −8 %, −20 % | +15 %, −13 % |
| `-lgc 2100` | 1 138 | 0,1739 | −15 %, −25 % | **+7 %, −19 %** |

Ma prédiction (2700 : −3/−4 %) est réfutée **en mieux**, et le mécanisme est nommé par la mesure : à 400 W la carte ne tient pas 2 992 MHz (certifie 389,7 W contre 397-399) — elle est bornée par la puissance, pas par l'horloge ; verrouiller 2 700 retire les pointes de boost (V² pour rien dans les phases GEMV liées à la mémoire) sans toucher au débit. Le levier « horloge mémoire » du 14/09 était réfuté ; le levier « horloge cœur bornée » ne l'est pas — ce sont deux réglages différents, la règle du 14/09 reste vraie pour le sien.

## 2. Ce qui manque avant la revendication « devant partout, vitesse et énergie »

* **b=1 sous `-lgc`** : le 14/09, `-lgc 2100` coûtait −23 % de t/s à b=1 (latence liée à l'horloge, pas à la puissance). Notre marge b=1 sur llama.cpp est de **+6,8 %** (363,3 contre 340,1). Le mode éco publié doit dire ce qu'il fait à b=1, sinon la cellule b=1 du comparatif change de régime en silence. Scellé **E1-bis** (un seuil par grandeur) : b=1 harnais égal, ABAB libre, `-lgc 2700` puis `2400` : **t/s ≥ 340,1** tenu (sinon faux : le mode éco est un mode b ≥ 8, dit tel quel), **J brut ≤ 0,798** tenu. Prédiction poste7 : 2700 → −3 à −6 % t/s (≈ 345-352, tenu de peu), J −8 % ; 2400 → −12 % (≈ 320, **faux**). Issue gênante : 2700 < 340,1 → aucun réglage unique ne couvre b=1 et b=12 → gouverneur par lot (§ 3).
* **Le genou sous 2 100** : 2 100 laisse 7 % de marge sur 1 066. Un passage fin `-lgc {1 800, 1 950}` après E1-bis, mêmes seuils (≥ 1 066, ≤ 0,2136) : prédiction 1 950 → ~1 080 / 0,165 (tenu de justesse), 1 800 → < 1 066 (faux). Dix minutes, deux bras.
* **Le réglage n'est pas un défaut du moteur** : `nvidia-smi -lgc` est root, hors processus. Il se publie comme **cellules nommées** « acvram éco -lgc 2700 » et « -lgc 2100 » (régime dans le nom, REGLES § 4) à côté du défaut, et se livre comme `acvram eco {2700|2100|off}` (poste1, à sec : appelle `nvidia-smi -lgc`/`-rgc` via `sudo -n`, refuse sans droit, écrit le réglage dans `regime_ligne()` par lecture de `nvidia-smi -q -d CLOCK` — la ligne de régime doit porter l'horloge verrouillée, sinon un chiffre éco passera pour un chiffre défaut).

## 3. Suite (pas cette nuit) : gouverneur d'horloge par lot

Si E1-bis rend « 2700 faux à b=1 » : un gouverneur dans le moteur (même forme que `GardeSpeculation`, conditionné au lot réel sur 32 pas) — libre à b ≤ 2, 2 700 à 3 ≤ b ≤ 7, 2 100 à b ≥ 8 — donne les deux colonnes sans arbitrage humain. Coût : le réglage est root → un petit service `acvram-horloge` (unité systemd, socket local) que le moteur appelle ; 1-2 jours ; réfutable par J/jeton ABAB à lot mixte (b passe 1 → 12 → 1 dans la même fenêtre) contre libre.

## Ordre

* **poste2** — après P2 ligne 2 (15 min) et AVANT G1 : **E1-bis** b=1 (`-lgc 2700`, `2400`, ABAB libre, seuils § 2), puis genou `-lgc {1 800, 1 950}` b=12 (mêmes seuils que E1) ; un verdict `revue/verdict-eco-lgc-b1-genou-19-09.md`. Puis G1, puis porte A8.
* **chef** — comparatif : deux lignes « acvram éco -lgc 2700 » et « -lgc 2100 » (b=12 t/s, J brut, J net, source 5eb636c ; b=1 « à mesurer » jusqu'à E1-bis) ; revendication : « à b=12, devant llama.cpp en vitesse ET en énergie (−3 % à 2700, −19 % à 2100) » — pas encore « partout ». `ETAT.md` à jour ; fusion `poste2` 5eb636c après sa fenêtre.
* **poste1** — à sec, après `fausse_quant_a8` : `acvram eco {2700|2100|off}` (§ 2), `regime_ligne()` lit l'horloge verrouillée ; test à sec : la ligne de régime change quand `nvidia-smi -q -d CLOCK` rend un verrou (simulé).
* Chacun écrit à poste7 une fois par verdict, pointeur, chef en copie.
