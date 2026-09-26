# Le genre du tenseur est indissociable de son coût sur Llama-2-7B

Il y a une heure j'ai annoncé un discriminant parfait : X (les 76 promus par la
clé A seule) est 100 % attention, Y (les 27 promus par B seule) est 96,5 %
`down_proj` + `lm_head`. **C'est vrai, et ce n'est pas un discriminant
identifié.** Mesuré depuis :

```
genre        n   cout de promotion par tenseur
lm_head      1   57,62 Mio
down_proj   24   19,82 Mio
gate/up      2   19,82 Mio
q k v o     76    7,38 Mio      <- les quatre au meme cout, a la deuxieme decimale

max de X = 7,38 Mio    min de Y = 19,82 Mio    facteur 2,69    AUCUN recouvrement
```

Sur ce modèle, « le genre du tenseur » et « le tenseur le plus gros » désignent
**exactement le même ensemble** : les projections d'attention sont 4096×4096, le
MLP est 11008×4096, la tête est 32000×4096. Les deux explications sont
inséparables ici.

## Ce que ça retire à ma note précédente

**« La taille est éliminée comme discriminant » était vrai de la taille AGRÉGÉE
et faux de la taille par tenseur.** Les totaux se ressemblent (560,5 contre
572,9 Mio) parce que le budget est le même par construction — 76 petits d'un
côté, 27 gros de l'autre. Par tenseur, la taille sépare parfaitement. C'est le
dénominateur emprunté, encore : un chiffre juste pour un total, faux pour un
individu.

Le compte des candidats éliminés retombe donc de trois à deux : paramètres et
variation intra-groupe. **Taille et genre sont un seul candidat, non tranché.**

## Ce que ça fait à la prédiction de chef sur `o_proj`

Son mécanisme : `lm_head` sort dans les logits, `down_proj` écrit dans le
résiduel, leur erreur ne traverse aucune normalisation qui l'absorbe. Il en
déduit que `o_proj`, qui écrit aussi dans le résiduel, devrait suivre
`down_proj` — or `o_proj` est dans X, du côté qui perd, donc la prédiction
devrait tomber.

**Elle ne tombe pas, parce que ce test n'en est pas un.** L'appartenance à X est
décidée par le **coût** : la clé A trie par décibels par octet, `o_proj` coûte
7,38 Mio comme toute l'attention, donc A le prend, quelle que soit sa position
dans le résiduel. X ne mesure pas un rendement, il mesure une préférence de
prix. Le mécanisme reste ni confirmé ni réfuté.

**Le bras qui le trancherait**, et il est constructible aujourd'hui avec le mode
liste :

```
o_proj seul   29 tenseurs   7,38 Mio chacun   ECRIT dans le residuel
v_proj seul   30 tenseurs   7,38 Mio chacun   n ecrit PAS dans le residuel
```

Même coût, même famille, même nombre à un près, **position résiduelle opposée**.
C'est le seul appariement du modèle qui isole le mécanisme du prix et du genre.
Si `o_proj` rend nettement plus que `v_proj`, le mécanisme tient ; s'ils rendent
pareil, il tombe.

## Les deux clés, et pourquoi elles sont deux

`genre` et `cout_decroissant` sont **dérivées de l'observation** : bâties pour
reproduire B, elles ne peuvent pas le confirmer. Écrit dans le code avant de les
construire, à la demande de chef.

Elles existent en paire précisément à cause du confondant : **elles doivent
rendre le même ensemble de promus sur Llama-2**, et divergeront sur un modèle où
une projection d'attention est aussi grosse qu'une projection de MLP (GQA à
`num_key_value_heads` élevé, MLA, tête non liée). **Leur écart, quand il
apparaîtra, sera la mesure du confondant** — et c'est aussi la validation que
chef demande : si « promouvoir `down_proj` puis `lm_head` » gagne sur un modèle
que nous n'avons pas regardé, ce n'est plus une description, c'est une loi.
