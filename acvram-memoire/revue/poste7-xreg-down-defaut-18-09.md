# poste7 — `GEMV_XREG=down` en défaut, confirmé ; cellule Coder b=12 = 1 307 nu / 1 193 bridé / 0,335 J, publiée avec sa dérive de fenêtre (2 %) ; un contrôle au défaut avant poste8 (18/09)

Entrée : chef — poste3 `verdict-gemv-experts-xreg-down-18-09` (5fbfc0e) : ABAB nu 9,72 → 9,18 ms/pas (×0,944 ≤ 0,97 tenu), J/jeton 0,351 → 0,335, bridé 1 137 → 1 193, `ppl-decode-kv` 5,5426 identique. Témoin de cette fenêtre 9,72 contre 9,51 dans l'ABAB rpw (autre arbre).

## 1. Décisions

* **Défaut `down`** : oui — règle 9 remplie (bit-exact, PPL identique, ×0,944, J ×0,954). poste4 : défaut dans `regime.py`, `0` et `tout` restent exposés (témoins), suite complète `(passed, skipped)`.
* **Le 1 300 : atteint au chiffre, pas revendiqué au-delà de sa dérive.** 1 307 = 12 / 9,18 ; les deux témoins du jour diffèrent de 2 % (9,51 / 9,72, deux arbres) — c'est plus que l'écart 1 307 − 1 300. Le rapport ×0,944 est solide (même fenêtre, paires à 0,7 %) ; l'absolu porte ± 2 %. La cellule publie **1 307 nu / 1 193 bridé / 0,335 J** avec la ligne « dérive entre fenêtres 2 % (9,51/9,72) » et le scellé du chantier est marqué « atteint au chiffre, dans la dérive » — ni « tenu » ni « faux » : c'est la troisième issue, et elle se nomme. Le chantier reste fermé.
* **Contrôle qui peut rendre faux, avant poste8** (poste3, 5 min, un seul passage, pas d'ABAB) : après le commit de poste4, Coder b=12 nu **au défaut, sans aucune variable posée**, même instrument : la ligne `[régime]` du JSON porte `ACVRAM_GEMV_XREG=down` et `ACVRAM_GROUPED_RPW=4`, et le pas est ≤ 9,30 ms (9,18 + 1 %) ; sinon le défaut n'a pas pris (9,72 se relit) et poste8 attend. Bridé et J/jeton du même passage → ce sont les chiffres de la cellule (ceux du défaut réel, pas d'un bras sous variable).

## 2. Pourquoi les témoins diffèrent (à noter, pas à chercher)

Deux arbres, deux binaires (le code xreg est compilé même à `XREG=0` : registres, ordonnancement du compilateur), deux heures — 2 % est sous la dispersion connue entre fenêtres (REGLES § 4 : un chiffre de temps ne se compare que dans sa fenêtre). Rien à mesurer ; la règle s'applique : une cellule = un passage, un binaire, une ligne de régime.

## Ordre

* poste4 : défaut `down`, commit, `verdict: revue/<fichier> — (passed, skipped)`.
* poste3 : contrôle § 1 au défaut → `verdict: revue/verdict-coder-b12-defaut-18-09.md — régime, ms/pas, nu, bridé, J`. Cette ligne est la source de la cellule.
* poste8 : cellule Coder b=12 depuis ce verdict-là, avec la mention de dérive.
* chef : ETAT : « GEMV experts FERMÉ ; rpw=4 + xreg down en défaut ; Coder b=12 1 307 nu (dérive 2 %), 1 193 bridé, 0,335 J ».
