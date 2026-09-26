# La queue de variation, et ce qu'elle a fait apparaître

Réserve posée par chef le 10/09 : ma réfutation du mécanisme de l'hypothèse A
reposait sur des **médianes**, et une médiane ne voit pas une queue. Le mécanisme
ne prédit pas que *tous* les tenseurs de X souffrent — seulement ceux à forte
variation intra-groupe. Cinq sur soixante-seize ne bougeraient pas la médiane, et
la perplexité les paierait quand même : elle somme, elle ne prend pas la médiane.

Réserve juste. Trois lectures, aucune carte, `outils/queue-variation-x-y.py`.

## 1. La queue existe

```
   seuil      X (76)      Y (27)    X en %    Y en %
    1.38          14           3     18.4%     11.1%
    1.40           4           0      5.3%      0.0%
    1.42           2           0      2.6%      0.0%
    1.45           2           0      2.6%      0.0%
    1.50           2           0      2.6%      0.0%
```

Deux tenseurs de X dépassent 1,45 ; **aucun** de Y ne dépasse 1,40. Ce sont
`model.layers.1.self_attn.q_proj` (1,5353, p99 = 3,201) et
`model.layers.1.self_attn.k_proj` (1,5085, p99 = 2,922) — la couche 1.

## 2. Mais elle ne pèse rien, et la pondération va contre le mécanisme

```
groupe                        octets Mio  moy ponderee  moy simple   mediane
X = A seul (76)                   1244.5        1.3727      1.3727    1.3652
Y = B seul (27)                   1272.1        1.3719      1.3701    1.3677
ecart des moyennes ponderees : +0.06 %   (medianes : -0.18 %)
```

Les deux tenseurs de la queue portent **2,6 % des octets de X**. Et le test que
chef désignait comme décisif — pondérer par les octets, puisque la perplexité
paie au prorata du poids — **resserre l'écart au lieu de l'ouvrir** : +0,06 %
contre −0,18 % en médiane. La raison est structurelle : les tenseurs à forte
variation de X sont les **petits** (q/k à 16,4 Mio), ceux de Y sont les **gros**
(`lm_head` 127,9 Mio, `down_proj` 44,0 Mio).

Le mécanisme reste donc infirmé sur les trois lectures. **Mon premier verdict
disait le contraire, sur un seuil vide de contenu** : « X en a deux, Y aucun » est
satisfait par n'importe quelle queue non vide. Le seuil qui tranche porte sur les
**octets** — il est posé à 10 % dans l'outil, et la queue en porte 2,6.

## 3. Ce qui n'était pas cherché : le genre de tenseur sépare totalement

```
-- X = A seul (76)   1244.5 Mio            -- Y = B seul (27)   1272.1 Mio
    30 tenseurs  491.2 Mio  39.5 %  v_proj     24 tenseurs 1056.2 Mio 83.0 % down_proj
    29 tenseurs  474.9 Mio  38.2 %  o_proj      1 tenseur   127.9 Mio 10.1 % lm_head
     9 tenseurs  147.4 Mio  11.8 %  q_proj      1 tenseur    44.0 Mio  3.5 % gate_proj
     8 tenseurs  131.0 Mio  10.5 %  k_proj      1 tenseur    44.0 Mio  3.5 % up_proj
```

**X est 100 % attention. Y est 96,5 % `down_proj` + `lm_head`. Aucun
recouvrement de genre.** Après trois candidats éliminés (taille, paramètres,
variation), c'est le premier discriminant parfait.

Il est en partie **tautologique** avec les clés, et il faut le dire : la clé A
trie par gain de décibels **par octet**, donc elle préfère les tenseurs peu
coûteux — l'attention est petite ; la clé B a le signe inverse et prend les gros.
Le fait non trivial n'est pas la séparation, c'est que **B gagne** : ce que la
perplexité veut voir promu, ce sont `down_proj` et `lm_head`, pas l'attention.

Corollaire testable, et plus fort que « B » : une clé qui promeut simplement
`down_proj` puis `lm_head` devrait approcher B sans aucun décibel.

## Ce que ça ne dit pas

Rien n'est mesuré ici sur la perplexité. Les prédictions de X et Y restent
écrites : **B**, par chef et par moi.
