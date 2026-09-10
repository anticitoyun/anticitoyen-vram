# RESULTAT `PA_WARPS` — 10/09 10:42, six points, trois empreintes distinctes

    PA_WARPS   ctx    pas (ms)   p (ms)   p / pas    contre PA_WARPS=4
           4   128      28,687    0,612    2,13 %          —
           8   128      28,279    0,367    1,30 %      -40,0 %
          16   128      28,134    0,185    0,66 %      -69,7 %
           4  1024      33,176    5,198   15,67 %          —
           8  1024      30,620    2,678    8,74 %      -48,5 %
          16  1024      29,355    1,409    4,80 %      -72,9 %

Contrôles passés aux six points : bras inconnu qui lève, trois sorties distinctes,
chaque bras reproductible (0,005 à 0,025 ms entre occurrences), graphes actifs,
0 couche exilée, **trois empreintes de `.so` distinctes**. Reproductibilité entre
campagnes : le pas à `PA_WARPS=4` ctx 1024 vaut 33,176 ms contre 33,286 ce matin,
soit 0,3 %.

## 1. Le gain, et il est moteur, pas seulement de noyau

    ctx  128 : 28,687 -> 28,134 ms   -1,9 % de temps par jeton, +2,0 % de debit
    ctx 1024 : 33,176 -> 29,355 ms  -11,5 % de temps par jeton, +13,0 % de debit

**Un paramètre de compilation, une ligne, +13 % de débit à ctx 1024.**

## 2. Mon modèle d'occupation tient — et ma prédiction est fausse deux fois

La clause était écrite : *si ctx 1024 ne montre aucun gain quand ctx 128 en montre,
mon modèle d'occupation tombe une seconde fois.* Les deux montrent du gain, monotone,
dans le sens prédit. **Le modèle tient.** Mais les deux chiffres que j'avais avancés
sont faux :

- **l'amplitude** : j'annonçais −20 à −45 % entre 4 et 16, c'est **−72,9 %**. Sous-
  estimé d'un facteur 1,6 à 3,6.
- **le classement** : j'annonçais le maximum du gain relatif à ctx 128 « puisque `C`
  n'y peut rien ». Le maximum est à **ctx 1024** (−72,9 contre −69,7 %). Mon
  raisonnement confondait *« `C` ne peut rien »* avec *« `PA_WARPS` peut plus »* :
  l'effet de `PA_WARPS` ne dépend pas de `C`, il est donc du même ordre partout, et
  c'est là où l'attention **pèse** le plus qu'il rapporte le plus.

## 3. Le fait le plus important : le seuil est franchi dans l'autre sens

À `PA_WARPS = 16` et ctx 1024, `p` tombe à **4,80 % du pas mesuré, sous le seuil de
5,24 %**. L'attention cesse donc d'être le poste qui contredit la conclusion
d'`ETABLI.md:2476`. Autrement dit : **cette conclusion n'était pas fausse en soi, elle
était fausse pour le noyau d'alors**, et un correctif de compilation la rend vraie.
C'est le premier cas de la journée où un seuil est franchi par une optimisation et non
par une correction de mesure.

## 4. Ce que ça change pour `PA_WARPS = 32`, contre ma propre prédiction

J'avais écrit que 32 n'apporterait rien, voire régresserait, l'occupation par SM y
retombant de 48 à 32 warps. **La décroissance observée est monotone et loin de
s'aplatir** : −40 %, puis −69,7 % ; rien n'annonce un retournement à 16. Le point 32
redevient donc intéressant, et il est bloqué par une instanciation `D = 512` **que
personne n'utilise** (0 modèle sur 120 recensés). **Le plafonnement des warps par
instanciation passe de « à faire si utile » à prioritaire.**

## 5. Une coïncidence à ne pas prendre pour une validation

À `PA_WARPS = 16`, mon pas vaut 29,355 ms contre 28,748 attendus au banc — écart
+2,2 %, dans les « quelques pourcents » que mon contrôle exigeait. **Il ne faut pas en
conclure que mon montage est devenu conforme.** Le banc de référence tournait à
`PA_WARPS = 4`, réglage où mon montage donne 33,176 ms, soit +15,4 %. Mes conditions
n'ont pas changé entre les deux réglages : l'accord numérique à 16 est **fortuit**, et
le montage reste 15 % plus lent que le banc à réglage égal. Le conclure serait la même
faute que le dénominateur emprunté — prendre un chiffre qui s'approche pour un chiffre
qui correspond.

## 6. Réserve sur le transport

Tout ceci porte sur `paged_attn_partial`, donc sur les modèles à attention paginée.
`mla.py` ne contient **aucun** appel à ce noyau : ces gains ne se transportent pas aux
modèles MLA du parc, GLM-4.7 compris.
