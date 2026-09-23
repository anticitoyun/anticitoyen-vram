# Le confondant genre / coût ne se casse pas dans le régime dense

Jérôme demande un second modèle pour valider les clés `genre` et
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

## La généralisation proposée, et pourquoi je ne l'écris pas telle quelle

Jérôme propose d'écrire *« le confondant est indissociable dans toute
architecture dense »* plutôt que *« le parc n'en contient pas »*, au motif qu'un
`intermediate_size` inférieur à `hidden_size` n'existe pas. **Sa conclusion tient
pour le parc, son critère est faux, et sa marge est plus courte qu'il ne le
pense.**

**La condition exacte n'est pas `intermediate_size < hidden_size`** — elle est :

```
num_attention_heads x head_dim  >=  intermediate_size
```

La reformulation par `hidden_size` suppose `nh × head_dim == hidden_size`.
**C'est faux pour 42 des 92 denses du parc**, jusqu'à un facteur 2,0 :

```
Qwen3-0.6B    hidden 1024   nh x hd 2048   ratio 2,000   condition 0,6667
Agents-4B     hidden 2560   nh x hd 4096   ratio 1,600   condition 0,4444
```

Il y a donc **deux routes** pour casser le confondant, pas une : rétrécir le
MLP, ou **élargir l'attention au-delà de `hidden_size`**. La seconde existe et
elle est courante — elle ne suffit simplement pas.

**Et la marge compte.** Le meilleur dense du parc atteint `att/mlp = 0,6667`,
soit **un facteur 1,50 du seuil**, contre 2,69 pour Llama-2. Une architecture
dense à `nh × hd = 2 × hidden` et `intermediate = 2 × hidden` casserait le
confondant, et elle n'a rien d'absurde : c'est Qwen3-0.6B avec un MLP à ×2 au
lieu de ×3.

**Donc j'écris « aucun dans ce parc de 92, le plus proche à un facteur 1,50 » et
non « aucune architecture dense au monde ».** Un échantillon de 92 avec un
quasi-succès ne porte pas une impossibilité universelle, et l'argument « il n'y
aurait aucune raison de l'écrire » est un argument, pas une mesure. La
différence est opérationnelle : sous sa formulation il n'y a rien à chercher,
sous celle-ci il y a une condition précise à passer au crible d'un catalogue.

## Ce qui reste faisable

1. **Le bras `o_proj` contre `v_proj`** — 29 contre 29, coût identique à
   l'octet, mêmes couches. Il ne sépare pas genre et coût, mais il teste le
   mécanisme résiduel, qui est l'explication *physique* candidate. **Et il est
   unique** : aucun autre appariement du modèle n'isole un mécanisme du prix et
   du genre à la fois. S'il ne tranche pas, rien ne tranchera, et ce sera un
   résultat à écrire tel quel.
2. **Un dense vérifiant `nh × head_dim ≥ intermediate_size`.** Le parc n'en a
   pas ; la condition est écrite, testable sur n'importe quel `config.json`, et
   l'outil la calcule. Ce n'est pas une acquisition à proposer — c'est un
   critère à garder sous la main.
3. **La paire `genre` / `cout_decroissant` sur Llama-2 reste un contrôle de
   cohérence** : elles doivent rendre le même ensemble. Si elles divergent ici,
   c'est une faute de code, pas un résultat.
