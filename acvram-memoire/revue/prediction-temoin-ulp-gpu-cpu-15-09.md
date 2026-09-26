# Témoin GPU vs CPU (poste7 §3) — mesuré, nuance à trancher

poste1, 15/09/2026. Deuxième témoin référence-contre-elle-même (chef,
après le témoin CPU eager/sdpa) : HF bf16, `cuda:0` vs `cpu`, mêmes 16
jetons.

## Mesuré

| position | delta | ulp |
|---|---|---|
| 0 | 1,8359 | **14,69** |
| 1 | 0,1250 | 1,00 |
| 2 | 0,1250 | 1,00 |
| 3 | 0,1250 | 2,00 |
| 4 | 0,1875 | 3,00 |
| 5 | 0,1250 | 1,00 |
| 6 | 0,1250 | 1,00 |
| 7 | 0,1562 | 1,25 |
| 8 | 0,2188 | 1,75 |
| 9 | 0,1562 | 1,25 |
| 10 | 0,1250 | 1,00 |
| 11 | 0,1250 | 1,00 |
| 12 | 0,1250 | 1,00 |
| 13 | 0,2500 | 1,00 |
| 14 | 0,1250 | 1,00 |
| 15 | 0,2500 | **4,00** |

Top-1 identique aux 16 positions dans ce témoin (contrairement à
acvram vs HF, où top-1 diffère à la position 0).

## Nuance — pas tranchée ici

**Plancher brut (max sur les 16) = 14,69 ulp**, porté ENTIÈREMENT par la
position 0. Appliquer « seuil = max(4,00 ; 14,69) + 1 = 15,69 » comme
multiplicateur GÉNÉRAL (delta ≤ N ulp pour TOUTES les positions)
rendrait le critère quasi inopérant : `mini-acvram4` vs HF passerait
16/16 trivialement, y compris sur des positions où un vrai bogue futur
ne serait plus détecté.

**Ce que montre la position 0 dans CE témoin** : son écart top-1/top-2
propre (référence GPU seule) = 0,75 = **6 ulp** — plus grand qu'au témoin
CPU (0,375 = 3 ulp) mais du même ordre : position 0 est un ex-aequo
CONFIRMÉ, INDÉPENDANT du bras (CPU/GPU/eager/sdpa) — une propriété de
CE jeton précis pour CE modèle, pas un artefact d'implémentation d'un
bras en particulier. Elle diverge fort (14-15 ulp de delta plein-
vecteur) PARCE QUE c'est un ex-aequo, pas parce que le plancher général
de bruit serait à 15 ulp.

**Les 15 AUTRES positions, dans les DEUX témoins, plafonnent à 4 ulp**
(CPU eager/sdpa : 4,00 à la position 3 ; GPU/CPU : 4,00 à la position
15). Deux témoins indépendants s'accordent sur ce chiffre pour tout ce
qui n'est pas un ex-aequo.

## Recommandation, pas décidée seule

Calculer le plancher SANS la ou les positions déjà identifiées comme
ex-aequo (exclues de la même façon qu'elles le seraient dans le critère
lui-même), pas sur le maximum brut des 16. Avec cette lecture : plancher
= 4 ulp (les deux témoins s'accordent), seuil général = **5 ulp** —
inchangé par ce second témoin. La position 0 reste couverte par la voie
ex-aequo, dont le seuil (le même multiplicateur, par construction du
critère) la couvre déjà à 5 ulp (gap 3-6 ulp ≤ 5).

Si poste7/chef préfèrent appliquer le plancher BRUT (15,69) comme seuil
général plutôt que d'exclure l'ex-aequo du calcul du plancher, le
critère devient beaucoup plus permissif — à eux de trancher, ce n'est
pas une décision technique pure.
