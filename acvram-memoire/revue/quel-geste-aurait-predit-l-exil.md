# Quel geste de ggrun aurait prédit la falaise de l'exil ? (poste1, 12/09/2026)

Évalue les gestes 1-3 de [`ggrun-lanceur-placement-moe.md`](ggrun-lanceur-placement-moe.md)
contre `_reajuster_plan` (`acvram/engine/loader.py:1082-1171`) et le fait mesuré :
douze couches exilées font passer l'évaluation de 61-82 s à **257 s** (facteur
3 à 4), **entièrement dans le temps**, la qualité inchangée au dix-millième
(`docs/TEST-EXIL.md:97-102`).

## Sur quel axe `_reajuster_plan` décide

Sur l'axe des **octets**. La boucle descend un MLP tant que
`utilise() > capacite - marge` (loader.py:1141-1146) ; `utilise()` somme des
octets (1098-1105), `capacite` est une capacité (1113). Aucune durée n'entre
dans la décision. Elle exile la dernière couche candidate jusqu'à ce que les
octets tiennent, sans jamais chiffrer ce que cette couche coûtera par jeton.

Or le coût de l'exil est « entièrement dans le temps » (TEST-EXIL:100). Décider
sur les octets, c'est décider sur le mauvais axe : c'est la définition de la
falaise — un gain de quelques Gio payé d'un facteur 3-4 en débit.

## Verdict par geste

| geste | prédit la falaise ? | pourquoi |
|---|---|---|
| **1. `T_transfer` chiffré** (bande mesurée) | **OUI — le seul** | seul geste sur l'axe du temps. `T_transfer(couche) = mlp_bytes / bande_PCIe` par jeton ; pour douze couches c'est exactement le +196 s (257−61) mesuré. Chiffré avant de descendre, il refuse l'exil dès que `T_transfer` dépasse le calcul GPU épargné. |
| **2. Preuve d'allocation à niveaux** | non | orthogonale. Elle attrape une allocation qui **échoue en silence** (`verifier_place` aveugle à la mémoire épinglée). Or la couche exilée **s'alloue parfaitement** en RAM hôte : c'est le placement qui RÉUSSIT qui est la falaise. Un contrôle d'allocation ne rend jamais « lent ». |
| **3. Inventaire mesuré VRAM/DRAM/PCIe** | non, seul — mais entrée de (1) | une bande PCIe mesurée ne prédit rien sans modèle qui la consomme ; c'est l'entrée de `T_transfer`. **Preuve que l'octet ne suffit pas** : `_reajuster_plan` mesure DÉJÀ la moitié VRAM de cet inventaire (`torch.cuda.mem_get_info`, loader.py:1117) et exile quand même. Mesurer l'axe des octets n'était pas ce qui manquait. |

## Ce qui manquait, en une ligne

Pas une mesure de capacité — le loader en fait une (1117). **Un axe de temps.**
Geste 1 est le seul à en poser un ; geste 3 lui fournit la bande PCIe ; geste 2
répond à une autre question (le placement est-il valide, pas combien il coûte).
Ordre d'adoption : 3 (mesurer la bande) → 1 (la facturer), 2 séparément.
