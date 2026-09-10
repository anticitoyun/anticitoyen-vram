# Le confondant genre / coût ne se casse pas dans le régime dense

chef demande un second modèle pour valider les clés `genre` et
`cout_decroissant` : si « promouvoir `down_proj` puis `lm_head` » gagne aussi sur
un modèle que nous n'avons pas regardé, ce n'est plus une description, c'est une
loi. J'ai cherché lequel. **Résultat négatif, et il est net.**

Le confondant est structurel : sur Llama-2-7B toute projection d'attention coûte
7,38 Mio et toute projection de MLP au moins 19,82, parce que l'attention est
4096×4096 et le MLP 11008×4096. Le rapport qui décide est
`intermediate_size / hidden_size` — 2,6875 chez Llama-2. Il se casse quand ce
rapport approche ou passe sous 1.

`outils/ou-genre-et-cout-divergent.py` lit les 127 `config.json` du parc. Aucune
carte, aucun poids.

```
127 modeles lus, dont 35 MoE
DENSES dont le confondant est casse : 0

casser le confondant et etre MoE : 35 et 35, ensembles identiques = True
```

**Les 35 modèles qui cassent le confondant sont exactement les 35 MoE.** Ce
n'est pas « 35 et 35 » : c'est le même ensemble, vérifié par comparaison des
noms, pas déduit des comptes.

La raison est claire une fois vue : ce qui casse le confondant, c'est un MLP plus
étroit que l'attention, et **le seul mécanisme qui rétrécit un MLP est de le
découper en experts**. `moe_intermediate_size` vaut 0,25 à 0,375 fois
`hidden_size` sur le parc, contre 2,69 pour un dense. L'inversion y est même
totale : attention 3,69 Mio contre expert 0,46 Mio, facteur 8 dans l'autre sens.

## Pourquoi ce n'est pas la validation demandée

**Un gain ne se transporte pas hors de son régime** — c'est notre propre règle,
et un MoE change trois choses à la fois : les experts ne sont activés que pour
une part des jetons, le nombre de tenseurs promouvables explose, et la
perplexité ne paie plus au prorata du poids mais au prorata du poids **fois sa
fréquence de routage**. Mesurer là-bas ne validerait pas la clé sur les denses ;
ça mesurerait une autre grandeur.

## Et le confondant ne se casse pas non plus par regroupement

J'ai cherché s'il existait, dans Llama-2 seul, deux groupes de **même coût** et
de **genre différent**. Il n'en existe pas d'utile :

- par tenseur, les deux familles ne se recouvrent pas (7,38 contre ≥ 19,82) ;
- par total, X et Y sont déjà appariés (560,5 contre 572,9 Mio) — c'est
  précisément la comparaison en cours, et le coût par tenseur y reste le
  confondant ;
- 64 tenseurs d'attention pèsent autant que 24 `down_proj` (472 contre
  475,7 Mio), mais les deux hypothèses y prédisent la **même** issue, puisque
  `down_proj` est à la fois gros et MLP.

**Dans une architecture où le genre détermine la taille, les deux ne sont pas
séparables, à aucun niveau de regroupement.** Ce n'est pas un manque d'ingéniosité :
c'est une propriété du modèle.

## Ce qui reste faisable

1. **Le bras `o_proj` contre `v_proj`** — 29 contre 29, coût identique à
   l'octet, mêmes couches. Il ne sépare pas genre et coût, mais il teste le
   mécanisme résiduel de chef, qui est l'explication *physique* candidate.
2. **Un dense à MLP étroit qu'il faudrait acquérir.** Le parc n'en a aucun ;
   c'est le chiffre à retenir avant de proposer une conversion.
3. **La paire `genre` / `cout_decroissant` sur Llama-2 reste utile comme
   contrôle de cohérence** : elles doivent rendre le même ensemble. Si elles
   divergent, c'est une faute de code, pas un résultat.
