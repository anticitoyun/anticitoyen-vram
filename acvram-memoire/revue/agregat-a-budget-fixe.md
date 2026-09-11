# Aucun agrégat par tenseur ne suit la PPL à budget fixe — et pourquoi

11/09/2026, sans carte. La courbe du quota fait varier le budget : tout y
décroît ensemble, une monotonie commune gonfle toute corrélation. Ici les
**quatre ordres** ont le même budget (6,00 Gio) et ne diffèrent que par
l'ensemble promu. Le confondant de budget disparaît.

## Les quatre points

```
ordre              PPL     promus   somme_rel   somme_abs   quadrature
inverse (B)        5,4482    149        7,59      588,74      58,25
normal  (A)        5,4918    198        3,37      391,00      56,42
erreur             5,4927    196        3,25      371,47      53,40
base_croissant     5,4981    173        5,96      577,34      68,79
```

Ordre PPL (meilleur→pire) : **B < A < erreur < base_croissant**.

## Résultat — négatif, et net

Accord de rang (τ de Kendall) de chaque agrégat avec la PPL :

```
somme_rel    -0,333   (à l'envers)
somme_abs    -0,333   (à l'envers)
quadrature    0,000   (aucun lien)
promus        0,000
```

**Aucun agrégat ne classe les quatre ordres comme la PPL.** Pire : `somme_rel`
et `somme_abs` les classent légèrement **à l'envers** — B, la meilleure PPL,
porte la **plus grande** somme d'erreurs (7,59 rel, 588 abs). L'erreur totale
par tenseur ne prédit donc pas la perplexité ; elle la prédit à rebours.

Cela prolonge le résultat de `base_croissant` : `gain_db` ne prédit pas la PPL,
et maintenant ni l'erreur relative, ni l'erreur absolue, ni la quadrature (la
forme « additive dans l'unité de la perte ») ne la prédisent non plus, à
budget fixe.

## Le confondant à nommer — les quatre ne promeuvent pas le même NOMBRE

B promeut **149** tenseurs, base_croissant **173**, erreur **196**, A **198** :
inverser le signe ou ignorer le coût change combien de tenseurs tiennent dans le
budget. `somme_rel`/`somme_abs` est alors dominée par ce nombre — moins on
promeut, plus il reste de tenseurs à l'erreur de base, plus la somme est grande.
Ces agrégats mesurent donc surtout le **compte de promus**, pas la qualité de
l'ordre, et le compte lui-même est à τ=0 avec la PPL.

**Conséquence : ce test à budget fixe ne peut pas séparer « quels tenseurs » de
« combien ».** Le résultat propre qui survit au confondant est plus simple et
déjà fort :

> À budget 6,00 Gio, **promouvoir moins mais mieux** (B, 149) bat **promouvoir
> plus** (A 198, erreur 196). « Plus de promotions » n'est pas « meilleur ».

## Réserve statistique, écrite d'avance

**n = 4.** Un τ sur quatre points est une piste, pas une preuve ; un seul point
le fait basculer, et l'écart normal↔erreur (5,4918 vs 5,4927) est de 0,0009 —
réel car la PPL est déterministe, mais minuscule. Le successeur qui trancherait
est un bras à **compte de promus égal** et ordres différents — la seule façon
d'isoler l'ordre du nombre.
