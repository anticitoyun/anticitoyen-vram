# Compte égal à 149 — résultat : ma prédiction est réfutée, le confondant a migré vers les octets

11/09/2026. Prédiction scellée (`compte-egal-149-prediction.md`) : *B reste le
meilleur à compte égal → l'ordre compte.* **Réfutée.**

## Les quatre bras à 149 promus

```
bras                    promus   dépense_Gio   PPL       écart/5,4141
inverse (B) @budget       149      5,9979     5,4482     +0,63 %
base_croissant @149       149      5,8640     5,4463     +0,60 %   ← meilleur
erreur @149               149      5,0851     5,5344     +2,22 %
snr @149                  149      5,0365     5,5379     +2,29 %   ← pire
```

**B n'est pas distinctement le meilleur : base_croissant@149 l'égale** (5,4463
vs 5,4482, écart 0,0019, à peine au-dessus du déterminisme). L'ordre, à compte
égal, ne sépare pas B des autres comme prédit.

## Pourquoi — fixer le COMPTE n'a pas fixé les OCTETS

Chaque ordre a un coût différent pour ses 149 premiers : `snr` promeut les
tenseurs les moins chers par gain d'abord (5,04 Gio), `base_croissant` ignore le
coût et en prend des chers (5,86 Gio). **À compte égal, la PPL suit les octets
dépensés, pas l'ordre :**

```
Pearson(PPL, dépense_Gio) sur les 4 bras à 149 = -0,992
```

Les deux bras qui dépensent ~5,9 Gio (B, base_croissant@149) donnent ~5,446 ;
les deux qui dépensent ~5,05 Gio (snr, erreur@149) donnent ~5,535. Le critère de
tri n'agit sur la PPL, à compte égal, que par le prix de ses 149 tenseurs — un
proxy des octets.

## Le confondant en cascade

```
budget fixe (6 Gio)   →  le COMPTE varie (149–198)   →  confond « quels » et « combien »
compte fixe (149)     →  les OCTETS varient (5,0–6,0) →  confond « quels » et « combien de bits »
```

Chaque contrainte en libère une autre. La grandeur qui prédit la PPL n'est ni
l'ordre, ni le compte : c'est **le nombre de bits alloués à la promotion**.

## L'effet résiduel, plus petit, qui subsiste

À octets égaux (~6 Gio, budget-fill), B (149 promus) bat A (198 promus) :
5,4482 < 5,4918. Même budget, mais B concentre ses bits sur moins de tenseurs
(plus de bits chacun). **Concentrer les bits sur peu de tenseurs aide** — c'est
un second effet, plus faible que celui des octets, et il survit au confondant.

## Ce que cela retire et ce que ça ouvre

**Retiré** : « l'ordre du sac à dos est le levier de la PPL ». Faux ou marginal.
Le levier est le budget d'octets, puis la concentration. Le débat snr/inverse/
erreur mesurait surtout combien coûtent les premiers de chaque liste.

**Ouvre** : le seul test propre restant fixe les **octets** (pas le compte) et
fait varier l'ordre — et il faut y ajouter un contrôle de concentration
(bits/tenseur). Sans carte pour la corrélation octets↔PPL, elle est déjà là :
−0,992.

## Note d'honnêteté

Ma prédiction était fausse, et je l'avais scellée pour ne pas pouvoir m'ajuster
après. Le geste a fonctionné : le résultat me contredit noir sur blanc, et c'est
lui qui a fait apparaître le confondant des octets — que ni la prédiction ni le
protocole ne prévoyaient.
