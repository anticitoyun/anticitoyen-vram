# Pièce 44 — compteurs de graphes exposés, et le signe de ma table était à l'envers — 22/09 (Océane, à sec)

* instrument : `pytest tests/test_compteurs_graphes.py tests/test_regime_graphes_vivants.py tests/test_stats_temps_cumules.py` (CPU, `CUDA_VISIBLE_DEVICES=""`, 2,5 s), lecture du client `banc-llamacpp-16-09.py` et de `acvram/engine/graphs.py:100-118, 608-620`
* commit : branche `oceane-compteurs-graphes` sur `9ed90223`
* régime : à sec, 0 min de carte, régime RÉSERVE (exception ≤ 15 min)
* scellé : les trois compteurs doivent sortir par `/metrics` ET par la ligne de régime, 0 sortie de génération changée, le test doit rendre faux sans le correctif
* mesuré : 14 verts avec le correctif ; **5 rouges sur 5 sans lui** (`git stash` de `runner.py` seul) — le contrôle peut rendre faux
* verdict : **exposé** — `graphes_nombre` / `graphes_captures` / `graphes_replays` dans `EngineStats.to_dict()` (donc `/metrics`) et `graphes_n=<n> captures=<c> replays=<r>` sur la ligne de régime ; **et la lecture de ma table du 22/09 (`d76cbb14`) a le signe inversé : elle attribuait « décroissante-puis-plateau » à H1, c'est H2**
* durée : ~12 min, prévu 15

## 1. Ce qui est livré

* `EngineStats.source_graphes` (champ non comparé) + `compteurs_graphes()` : les
  trois compteurs sont **lus en direct** sur le `GraphRunner`, jamais recopiés —
  `/metrics` est interrogé ENTRE deux pas, et un entier recopié aurait été juste
  ou faux selon l'endroit du rafraîchissement.
* `Engine.__init__` pose `self.stats.source_graphes = lambda: self.graphs`
  (lecture paresseuse de l'attribut : `self.graphs` change encore après ce
  point — repli en eager, faux runner d'un test).
* `regime()` porte les mêmes trois clés, de la même source ; `regime_ligne()`
  les imprime sous **`graphes_n=`** et non `graphes=` : la ligne porte déjà
  `graphes=on|off`, que trois lecteurs cherchent tel quel
  (`outils/gpu/mesure/capture-godets.py:41`,
  `tests/test_regime_graphes_vivants.py:44,61`) — deux clés du même nom
  auraient fait dépendre leur verdict de l'ordre de la recherche.
* Aucune sortie de génération touchée : rien n'est lu ni écrit dans le chemin
  du pas, les compteurs existaient déjà (`graphs.py:273, 292-293`).

## 2. L'hypothèse demandée : le contexte des sondes s'allonge-t-il ? — NON, réfutée par le client

`banc-llamacpp-16-09.py:211-216` : chaque passe courte construit des invites
**neuves** — `invite(2000 + p*31 + k, INVITE)`, `INVITE = 256`, 128 jetons
décodés. La graine change à chaque passe, donc les invites **diffèrent dès le
jeton 0** (`invite()` = `(k*104729 + i*7919) % …`, ligne 56-57) : pas de
préfixe commun, pas de cache de préfixe qui grandirait, aucun état de
conversation entre passes. La longueur d'attention va de 256 à 384 jetons à
**chaque** passe, identique de la première à la septième. **H6 close sans
carte.**

## 3. Ce qui gêne : ma table lisait le signe à l'envers

Une capture **coûte** du temps à la passe qui la paie (+30-60 %, H1) : H1
prédit donc une passe **lente d'abord, rapides ensuite** — croissante-puis-
plateau. La série de Manon est l'inverse (353,9 → 272, `b3e3fbd5`) : trois
passes rapides, puis un plateau **durablement 23 % plus bas**. Un coût ponctuel
ne fait pas un plateau bas ; il faut un changement de régime **durable**.

`graphs.py:103-118` le donne : **aucune éviction LRU**. Une fois les
`MAX_GRAPHS = 16` places prises, toute forme nouvelle est refusée
**définitivement** et repasse en eager pour la vie du serveur. La clé étant
`(b, ql, nblk, lb)`, les 20 s de fenêtre chargée qui précèdent les sondes
consomment les places ; les premières sondes rejouent encore leurs graphes,
puis une forme nouvelle ne trouve plus de place et le plateau eager s'installe.
Le commentaire du dépôt chiffre déjà ce mécanisme sur la dispersion de 26-37 %
du 10/09 ; ici l'écart est de 23 %.

**Table corrigée** (remplace la lecture de `d76cbb14`) :

| forme de la série | lecture |
|---|---|
| **croissante-puis-plateau** (lent d'abord) | **H1** — captures payées par les premières sondes, godets ensuite chauds |
| **décroissante-puis-plateau** (rapide d'abord) | **H2** — `MAX_GRAPHS` atteint, eager **durable** sans éviction ; ou H7 thermique |
| décroissante **sans** plateau | H7 (horloge/température) ou H5 (fragmentation) |
| sans tendance, dispersée | H4 (charge hôte) |
| périodique | H1 avec alternance de godets |

**H7 ajoutée** (thermique) : la fenêtre chargée de 20 s précède les sondes ;
horloge SM et température décroissantes donnent la même forme. Elle se sépare
de H2 sans ambiguïté : H2 ⇒ `graphes_n` plafonne à 16, `captures` cesse de
croître et `repli_eager` monte pendant les sondes ; H7 ⇒ les trois compteurs
sont **fixes** et ce sont `clocks.sm` / `temperature.gpu` qui bougent.

## 4. Ce que Manon relève au prochain trou (2 min, aucun code)

Les sept passes, en relevant **avant chaque passe** `/metrics`
(`graphes_nombre`, `graphes_captures`, `graphes_replays`, `repli_eager`) et
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu`. Prédiction écrite avant la
mesure : **H2** — `graphes_nombre` atteint 16 avant la 4ᵉ passe et n'en bouge
plus, `captures` se fige, `repli_eager` croît à chaque passe du plateau. Si les
trois compteurs sont fixes dès la 1ʳᵉ passe et que l'horloge SM baisse de plus
de 5 %, **H2 est fausse et H7 tient**.
