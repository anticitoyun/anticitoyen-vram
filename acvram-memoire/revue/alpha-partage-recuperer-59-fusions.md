# Les 59 fusions refusées : un scalaire par tenseur, et une issue que ni l'une ni l'autre des deux hypothèses ne prévoyait

## La lecture, `calibrate.py:222` `search_channel_scales`

```python
act = stats.mean_abs...clamp(min=1e-6)      # statistiques de l'ENTREE
for i in range(n_grid + 1):                 # n_grid = 20, donc 21 valeurs
    alpha = i / n_grid
    s = act.pow(alpha)
    s = s / s.mean()
    wq = _quant_dequant(w * s.unsqueeze(0), fmt, group_size)
    err = ((x / s) @ wq.t() - y_ref).norm() / ref_norm
    ...garder le meilleur
if torch.allclose(best_scale, ones, atol=1e-3):
    best_scale = None
```

**Les deux hypothèses de chef sont vraies à la fois, et c'est ce qui donne
l'issue.** La **direction** de l'échelle vient uniquement de l'activation :
`act.pow(alpha)`, et `act` est partagé par `gate` et `up` puisqu'ils lisent la
même entrée. Seul l'**exposant** dépend du poids, parce que `alpha` est choisi en
minimisant l'erreur de quantification **de ce tenseur-là**.

Donc `gate` et `up` reçoivent deux échelles **colinéaires en exposant** sur le
même vecteur `act`, et `_scaler_commun` les compare par `torch.equal` : deux
`alpha` différents, refus. Un seul scalaire, pris sur une grille de 21 valeurs,
décide de l'empilement d'une paire.

## Ce que ça explique exactement

Les **5 empilements réussis sur 64** de `Llama-2-7b-int8` : la dernière ligne
de la recherche remet `best_scale = None` quand l'échelle est l'identité à
1e-3 près. Deux tenseurs qui tombent tous deux sur `alpha ≈ 0` reçoivent donc
`None` tous les deux, et `_scaler_commun` répond « tous identité : rien à
porter ». **Les cinq paires fusionnées sont celles où ni `gate` ni `up`
n'avaient besoin d'échelle** — pas un hasard, une conséquence.

## L'issue : un `alpha` commun par groupe empilable

`gate` et `up` lisent la même entrée. Rien n'oblige à chercher leur `alpha`
séparément : on peut chercher **un seul `alpha` qui minimise l'erreur conjointe
de la paire**. Les deux échelles deviennent alors identiques par construction,
`_scaler_commun` accepte, et les 32 paires d'un Llama-2 fusionnent au lieu de 5.

Ce que cela coûte est **borné et calculable sans mesurer** : chaque tenseur
s'écarte au plus de quelques pas de grille de son propre optimum, et la
recherche calcule **déjà** `err` pour les 21 valeurs. Il suffit de conserver
les 21 erreurs par tenseur au lieu du seul minimum, puis de choisir l'`alpha`
qui minimise la somme sur le groupe. **Le prix en qualité se lit dans les
tableaux d'erreurs, à la conversion, sans un seul chargement ni un seul banc.**

Et il se lit *avant* de décider : si l'erreur conjointe au meilleur `alpha`
commun dépasse de trop l'erreur au meilleur `alpha` individuel, on ne fusionne
pas ce groupe — le choix redevient par paire, mais **informé** au lieu d'être
subi.

## Ce qu'il ne faut PAS en conclure

Le gain de débit de cette récupération est **inconnu**. Le seul chiffre que
nous ayons, `+2,60 %`, est mesuré en bf16 pur, où tous les empilements
aboutissent déjà — il dit ce que vaut la fusion complète contre l'absence de
fusion, pas ce que vaut passer de 7,8 % à 100 % sur un int8. Il est plausible
que ce soit du même ordre ; ce n'est pas mesuré, et l'écrire autrement serait
la faute de la journée une fois de plus.

**Ordre juste :** d'abord conserver les 21 erreurs et publier le prix en
qualité de l'`alpha` commun (conversion seule, pas de carte) ; ensuite, et
seulement si ce prix est faible, mesurer le débit gagné.

## Portée

Tout modèle int8 ou int4_awq **calibré**. Les nvfp4 ont le même mécanisme de
scaler et sont donc concernés aussi, malgré leurs vues déjà en place : la vue
règle la mémoire, pas le refus d'empiler. Les bf16 ne sont pas concernés : ils
n'ont pas d'échelle, et c'est précisément pourquoi le `+2,60 %` y a été mesuré.
