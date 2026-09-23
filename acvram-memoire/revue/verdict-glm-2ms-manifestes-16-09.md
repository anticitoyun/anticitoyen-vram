# Verdict — d'où viennent les +2,06 ms/pas entre les deux convertis GLM (à sec)

Manon, 16/09. Ordre Jérôme, relayant le pas b=12 GLM de Laurine (main
`97e3538`) : `-avant-noawq-experts` coûte +2,06 ms/pas contre `-sansawq`,
régime NOMINAL des deux côtés (0 exilé). Comparaison à sec des deux
manifestes, sans carte. Sage (main `86b152b`) a nommé l'hypothèse attendue
avant que je publie : les tables AWQ int8 de l'attention — confirmée
ci-dessous, en plus de celles des experts.

## Ce qui diffère : `opts.awq` global, pas un réglage MoE isolé

```
                              -avant-noawq-experts   -sansawq
tenseurs experts (9024)      has_act_scale=True      has_act_scale=False
                              9024/9024               0/9024
attention + down_proj dense  has_act_scale=True       has_act_scale=False
  int8, 124 tenseurs          124/124                 0/124
```

`-sansawq` n'écrit **aucune** clé `act_scale`, ni côté attention ni côté
experts (`has_act_scale: False` partout) — converti avec `opts.awq=False`
sur toute la passe. `-avant-noawq-experts` en écrit sur les DEUX groupes,
et ce sont de vraies échelles AWQ, pas des identités laissées par le
mécanisme d'identité explicite (`2205709`) :

* **Attention + `mlp.down_proj` dense (124 tenseurs int8)** : **124/124
  échelles réelles**, aucune identité. Échantillon :
  `q_b_proj` couche 0 (0,375–1,594), `q_a_proj` couche 1 (0,824–2,00),
  `down_proj` couche 0 (0,786–2,72).
* **Experts (`gate_proj`, échantillon de 401 tenseurs)** : **390/401
  réelles** (ex. couche 1 expert 0 : 0,544–1,871), 10 retombées à
  l'identité.

`-avant-noawq-experts` est un converti d'AVANT que `opts.awq` soit coupée
— nom pris au pied de la lettre : la recherche AWQ a tourné partout, pas
seulement sur les experts.

## Où ça coûte, dans le code

**Dense (attention, down_proj)** : `layers.py:479-481`
(`QuantLinear.forward`) —

```python
if self.scaler is not None and not self.scaler.is_identity:
    x = self.scaler.apply(x)
```

`scaler.is_identity` est un test STRUCTUREL (`scale is None and
hadamard_block == 0`, session du 15/09) — une échelle réelle non nulle
n'est jamais identité au sens de ce test, `apply()` s'exécute à chaque
appel, sur les 124 tenseurs, à CHAQUE jeton (pas de routage qui filtre
l'attention : les 124 sont sur le chemin de tous les jetons, contrairement
aux experts). C'est le facteur dominant attendu par Sage.

**Experts (MoE, gated par le routage)** : `loader.py:142-150`
(`_build_scaler`) construit un `ChannelScaler` réel pour `has_act_scale=
True` ; `model.py:731-746` (`_try_build_stacks`) empile la table `[E,K]`
par projection et applique la garde d'unité (`unite = torch.all(table ==
1)`, Sage §8, coût 0 scellé) — mais avec 390/401 réelles par échantillon,
quasiment aucune table de 64 experts n'est uniformément 1, `awq[nom]` reste
la table réelle. À chaque pas (`model.py:1114-1117`, même schéma au
prefill `:982-983`) :

```python
if awq.get("gate_proj") is not None:
    xs = xs / awq["gate_proj"][e_sorted.long()]
```

Division indexée par expert, exécutée pour les jetons routés vers un
expert à table non-unité (gate + up, deux fois si `up_distinct`), sautée
entièrement pour `-sansawq` (`awq.get(...)` vaut `None`, la garde `is not
None` court-circuite). Les deux convertis restent en régime NOMINAL, 0
exilé des deux côtés — ce n'est pas un repli de pile.

## Réponse

**+2,06 ms/pas = la somme du coût de l'AWQ appliquée au décodage sur DEUX
chemins** : le chemin dense (attention + down_proj, 124 tenseurs, exécuté
à chaque jeton, sans routage — le facteur dominant nommé par Sage) et le
chemin MoE groupé (experts, gated par le top-k). `-avant-noawq-experts` a
`opts.awq=True` partout (pré-coupure) ; `-sansawq` a `opts.awq=False`
partout. Pas un artefact de conversion, pas un repli de pile — un coût
structurel de présence de table, indépendant de sa valeur numérique.

Le diff tranche seul, pas besoin du bras B∅∅ (10 min carte) que Sage
proposait en repli si le diff ne suffisait pas.

Rendu à Jérôme/Sage : n'entre pas dans le sort du scellé de la
reconversion (`sage-glm-mma0-verdict-16-09.md` § 4 — noyau, pas précision,
`cb2784b` reste hors main) — c'est un fait de coût, pas de qualité, et il
ne discrimine pas (1a)/(1b) : `-avant-noawq-experts` et `-sansawq` ne
diffèrent que par la présence des tables AWQ, pas par le chemin
d'application.
