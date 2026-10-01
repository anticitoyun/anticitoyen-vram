# 61w — état KDA en [K, V] (poste5) sur Kimi-Linear-35B : ids gloutons IDENTIQUES sur 2 048 jetons à b=1 et b=12 ; b=12 −19,0 % du mur, b=1 −7,4 % (poste1, mesureuse, 01/10)

* instrument : `outils/gpu/mesure/kda-61w-abba.{sh,py}` (5458f4560 : étape 0 + ABBA A1 B1 B2 A2 en une prise, témoins A1=A2/B1=B2, refus sans .so précompilé), sorties `scratchpad/poste1-61w-01-10/`
* commit : A = 27b7c5451 (main, merge-base de poste5-kda) ; B = a7be59ef7 (poste5-kda, final) ; arbres figés, acvram de chaque bras contrôlé (ACVRAM_ARBRE + garde) ; .so précompilés, aucun JIT : A e0912aeb4fbde2c7 (compilé à sec), B 98af5877cb43721c (poste5, sha256 093921d9e795c5dd)
* régime : Kimi-Linear-35B-kda-nvfp4, régime NOMINAL (graphes on, hybrides ≤ 12, 0 exilée), en processus, glouton, eos coupé, invites synthétiques de 256 ; contexte 2 368 ; 5090 seule, avant/après : seul 8081 (3080 Ti) ; pause e50.2 tenue de 10:37 à 10:48:15
* scellé : prédiction de `poste1-p81-cake-kda-01-10.md` § « Levier » (b=12 : −14 à −16 % de noyaux, mur 11,07 → 9,3-9,6 ms, FAUX si gain < 1,0 ms ; b=1 : −9 à −11 % du mur, noyau ≤ 0,25 ms) ; exigence du chef : au bit, 2 048 jetons à b=1 et b=12
* mesuré :
  - étape 0 : `tests/test_kda_etat_kv.py` sur carte, arbre B : **8 passed** (au bit strict en pas chaînés, D=64 et D=128, 64 et 512 pas — l'itération j=0 contractée du témoin reproduite)
  - équivalence : témoins A1=A2 et B1=B2 au jeton près ; **A et B IDENTIQUES** sur 2 048 jetons, b=1 (1 séquence) et b=12 (12 séquences)
  - mur par pas (médiane de 6 répétitions de 128 pas par chemin, ABBA) :
    | | A (ancien) | B (61w) | Δ | prédit | lecture |
    |---|---|---|---|---|---|
    | b=1 | 3,649 ms | 3,381 ms | **−7,4 %** (−0,268 ms) | −9 à −11 % | **en deçà de la bande** (gain réel, plus faible que prédit) |
    | b=12 | 10,685 ms | 8,650 ms | **−19,0 %** (−2,035 ms) | −13 à −16 % (mur) | **au-delà de la bande** (mieux) ; FAUX (< 1,0 ms) non atteint |
    Étendue du témoin A : 0,4 % (b=1), 2,8 % (b=12, 3e répétition plus lente dans les 4 bras, 10,92 contre 10,63-10,69).
* verdict : **61w équivalente au jeton près et plus rapide aux deux lots ; défaut servable.** b=12 gagne plus que prédit : la prédiction comptait que le noyau fla relirait S en HBM une fois les transpositions retirées (+0,5-0,7 ms) ; fla écrit désormais l'état EN PLACE, et cette relecture coûte moins que prévu (à confirmer par nsys si l'on veut le détail). b=1 gagne moins que prédit : 0,268 ms au mur, contre 0,34-0,41 ; si tout le gain vient du noyau, celui-ci passe de 0,487 à ≈ 0,22 ms, sous le falsificateur de 0,25 ms (inférence, pas mesure : pas de trace dans cette prise)
* durée : prévue 8-9 min, tenue 248 s (étape 0 5 s, 4 bras ≈ 47-96 s) ; deux prises perdues avant : 1 s à 10:34:14 (contrôle d'origine sous ACVRAM_ARBRE hérité de carte.sh, corrigé 5458f4560) et 5 s à 10:34:55 (étape 0 rouge sur a7be59ef7 non encore écrit : pas 41, sortie bf16 déplacée ; poste5 a depuis reproduit l'itération contractée)

## Écarts
* Les 45 tests processeur de chef (10:34-10:35) précèdent cette prise (10:43:55) : sans effet.
* Invites synthétiques (ids arithmétiques), pas un texte : l'équivalence est celle du calcul, pas une qualité.
