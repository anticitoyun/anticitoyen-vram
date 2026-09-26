# Prédiction scellée — contrôle décisif en fp32

poste1, 14/09/2026, avant mesure.

## Ce que la couche 0 isolée a montré (revue/prediction-couche0-isolee-14-09.md, mesuré)

Prédiction RÉFUTÉE au sens strict (seuil 0,999999 dépassé aux 16 positions),
mais le motif est parlant : delta absolu ≈ 0,0002-0,001, cosinus
0,999993-0,999997 **partout**, y compris aux positions saines (2,4,6-15) —
pas de concentration sur 0,1,3,5. C'est la signature d'un bruit d'arrondi
bf16 diffus entre deux implémentations indépendantes (acvram et HF,
toutes deux en bf16, ordre des opérations différent), pas d'une MLA
fautive : une MLA fautive donnerait une erreur plus grande et/ou
concentrée sur les positions déjà rouges.

## Hypothèse

Le routage GLM (sigmoid+biais, déjà réfuté comme cause arithmétique
propre — revue/prediction-routeur-fp32-14-09.md) est correct, mais un
sous-ensemble des 16 jetons synthétiques a deux experts quasi ex-aequo au
rang 4/5 ; le bruit bf16 ambiant (~1e-3, présent PARTOUT, y compris couche
0) suffit à en faire basculer la sélection uniquement là où l'écart de
score est du même ordre. Ce n'est pas un bogue acvram : c'est une
instabilité intrinsèque du régime bf16 pour un routage top-k avec experts
proches, indépendante de l'implémentation.

## Prédiction

Rejouer l'équivalence entière (2 couches, 16 jetons) en **fp32** des deux
côtés (acvram `dtype=torch.float32`, HF `dtype=torch.float32`) : le topi de
la couche 1 est identique à HF sur les 16/16 positions, et pire cosinus
logits ≥ 0,999999.

**Seuil de réfutation** : si une seule position diverge encore en topi une
fois les deux bras en fp32, l'hypothèse « bruit bf16 diffus » est fausse —
il reste un vrai bogue algorithmique (poids gathered, expert partagé, ou
un défaut de la MLA qui ne se voit qu'en fp32 par une compensation
fortuite en bf16).
