# Le transport d'experts ne change pas un seul chiffre

Mesuré le 8 septembre 2026. Protocole écrit et accordé avant le lancement,
prédictions figées avant de voir le résultat.

## La question

La table des témoins de qualité distinguait deux cases : « MoE résident » et
« MoE exilé ». Elle les distinguait parce que nous ne savions pas si faire
traverser le PCIe aux poids d'un expert changeait le résultat du calcul. Tant
que la réponse manquait, tout écart mesuré sur un modèle partiellement exilé
était ambigu : le défaut pouvait venir du modèle ou du transport.

La question ne demande pas un second modèle. Elle demande le **même** modèle
mesuré deux fois avec deux placements.

## Le protocole

`lfm-8b-a1b-bf16` — 16 Go, 32 experts, le seul MoE du parc converti qui tienne
**entièrement** en VRAM, donc le seul qui parte de zéro couche exilée.
Corpus `wiki.test.raw`, fenêtre 512, stride 512, `min_context` 256,
146 717 positions notées.

| passe | placement |
|---|---|
| A₁ | 0 couche exilée |
| A₂ | 0 couche exilée, rien changé |
| B | `ACVRAM_EXIL_COUCHES=12` |

`_forcer_exil` ne touche que `mlp_storage` et `mlp_exec`. `embed_device`,
`lm_head_device`, `kv_budget` et `kv_bytes_per_token` sont lus du manifeste et
restent intacts. Une seule variable change.

A₂ existe pour donner le **plancher de bruit** : sans elle, un écart faible
entre A et B serait indistinguable d'une variabilité d'exécution.

## Les prédictions, écrites avant

1. **A₁ = A₂ au dernier chiffre.** Sinon la perplexité n'est pas reproductible,
   et c'est un défaut plus grave que celui qu'on cherche.
2. **A = B exactement**, pas « proche ». Un transport correct ne dégrade pas :
   il déplace des octets.
3. Troisième issue nommée d'avance : si A₁ = A₂ mais que B en diffère de
   l'ordre de 5e-3 relatif, la conclusion n'est pas « transport fautif » mais
   « ordre d'accumulation modifié » — l'exil change l'ordre d'arrivée des
   experts, et deux ordres de somme en bf16 diffèrent de 5,4e-3. Le départage
   se ferait en fp32.

## Le résultat

    A₁   ppl 58.815   sur 146 717 positions      82 s
    A₂   ppl 58.815   sur 146 717 positions      61 s
    B    ppl 58.815   sur 146 717 positions     257 s

**Les trois chiffres sont identiques.** Prédictions 1 et 2 vérifiées, la
troisième issue ne s'est pas présentée.

## Les contrôles, sans lesquels le résultat ne vaudrait rien

Le journal de B porte :

    [acvram] mesure : 12 couches à perceptron exilé (minimum imposé par la capacité : 0)

Ce contrôle est le plus important des trois. `_forcer_exil` **refuse
silencieusement** de descendre sous le minimum imposé par la capacité, et le
dit alors par `ACVRAM_EXIL_COUCHES=N ignoré`. Sans cette ligne au journal, B
aurait pu être une troisième exécution de A : on aurait obtenu l'égalité
prédite et conclu « transport innocent » en ayant mesuré deux fois la même
chose. Un résultat plausible, conforme à l'attente, et faux.

Aucune des trois passes n'a émis `plan réajusté : N MLP de plus en RAM hôte` :
le planificateur n'a pas ajouté d'exil de son côté, donc B est bien à douze
couches et A à zéro.

La VRAM libre relevée avant chacun des trois chargements est identique
(32 096 Mio et 6 762 Mio) : le placement ne dépend pas de l'instant, ce qui
n'allait pas de soi — le même modèle chargé deux fois de suite a rendu 15,8
puis 152,2 jetons par seconde le même jour, parce qu'un serveur avait rendu son
port sans rendre sa mémoire.

## Ce qui est établi, et ce qui ne l'est pas

**Établi** : sur ce modèle, exiler douze couches en mémoire hôte ne change pas
un seul chiffre de la perplexité. Le calcul reste en bf16 sur le GPU
(`mlp_exec` vaut `gpu` des deux côtés) ; seul le domicile des octets change, et
il ne se voit pas dans le résultat.

**Conséquence pour la table des témoins** : les cases « MoE résident » et
« MoE exilé » cessent d'être deux cases. Un modèle partiellement exilé peut
être mesuré et interprété sans réserve sur le transport.

**Non établi** : que ce soit vrai de toute architecture. LFM2 est hybride
(convolution courte) ; l'invariance mesurée ici porte sur son chemin d'experts,
pas sur ceux d'un GDN ou d'une attention latente. Le résultat se transporte à
un modèle dont le chemin d'experts est de même nature, pas à n'importe lequel.

## Ce que l'exil coûte en temps

Douze couches exilées font passer l'évaluation de 61-82 s à **257 s**, soit un
facteur 3 à 4. C'est le prix du PCIe, et il est entièrement dans le temps : le
chiffre de qualité, lui, ne bouge pas d'un dix-millième. Qualité et performance
se paient sur deux comptes séparés — c'est mesuré ici, et non supposé.
