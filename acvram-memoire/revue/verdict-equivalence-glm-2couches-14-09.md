# Verdict — équivalence CPU 2 couches GLM-4.7-Flash vs HF (item 2 de poste7)

poste2, 14/09/2026 soir. Seuil scellé par poste7
(`revue/poste7-lancement-14-09.md` §2) : max |Δlogit| ≤ 5·10⁻² **et**
cosinus ≥ 0,999 sur les logits des 16 positions d'une invite commune,
forward CPU bf16 des couches 0-1 (dense + première MoE), acvram contre HF
transformers. Réfuté → pas de conversion ce soir.

## Résultat

RÉFUTÉ. Sur les 16 positions :

| position | \|Δlogit\| max | cosinus |
|---|---|---|
| 0 | 3,4495 | 0,998698 |
| 1 | 1,6774 | 0,994951 |
| 2 | 1,8831 | 0,997807 |
| 3 | 1,9267 | 0,992638 |
| 4 | 0,1681 | 0,999774 |
| 5 | 5,0669 | 0,952137 |
| 6 | 4,2859 | 0,999004 |
| 7 | 2,6502 | 0,986674 |
| 8 | 1,2240 | 0,998870 |
| 9 | 2,3774 | 0,995889 |
| 10 | 0,6064 | 0,999924 |
| 11 | 0,8287 | 0,998951 |
| 12 | 3,1211 | 0,992331 |
| 13 | 0,2088 | 0,999970 |
| 14 | 1,4689 | 0,999202 |
| 15 | 2,9739 | 0,972043 |

Pire |Δlogit| : 5,0669 (seuil 0,05, ×101). Pire cosinus : 0,952137 (seuil
0,999). **Aucune des 16 positions ne passe le critère Δlogit** ; 8/16
passent le critère cosinus seul. Les deux sont exigés ensemble.

## Écarté avant de conclure

**Tenseurs manquants dans l'extraction** — écarté par mesure directe :
les 192 tenseurs des 64 experts (gate/up/down_proj) de la couche 1 (MoE)
sont présents un-à-un dans le mini-répertoire extrait, comparés à la
source (`model.safetensors.index.json`). Pas de perte de données côté
extraction.

## Non tranché, à ne pas confondre avec la cause

Le journal de conversion affiche, sur une conversion **bf16 pure, sans
AWQ, sans quantification** : `régime DÉGRADÉ — piles_ok=False`, cause
`« gate_proj : formats de quantification mélangés entre experts (ni
tout NVFP4, ni tout INT4) »` (message ajouté par moi le 14/09,
`acvram/engine/model.py::_try_build_stacks`). Ce contrôle vérifie
l'homogénéité de scaler de quantification entre experts — il ne devrait
normalement jamais se déclencher quand aucun expert n'est quantifié.
Deux lectures possibles, **aucune tranchée ici** :
1. faux déclenchement du contrôle sur du bf16 (les experts bf16 n'ont
   pas de scaler du tout ; peut-être lu comme « mélangé » par erreur) ;
2. signal réel d'un problème dans la construction de la pile MLA/MoE qui
   contaminerait aussi les résultats numériques.

Le régime DÉGRADÉ ne fait que forcer un chemin lent (boucle par expert
au lieu d'un noyau groupé) — **numériquement équivalent en théorie**,
donc ne devrait PAS à lui seul expliquer un écart de logit aussi grand.
Mais un doute existe : le chemin lent est moins testé que le chemin
rapide.

## Ce qui n'a pas été fait

- Pas de décomposition couche par couche (attention seule vs MLP seul)
  pour isoler où l'écart apparaît en premier.
- Pas de vérification indépendante du message DÉGRADÉ lui-même (cause
  1 vs 2 ci-dessus).
- Pas de conversion, comme l'exige le protocole sur un verdict réfuté.

## Suite

Pas de conversion GLM-4.7-Flash ce soir. Rendu à chef/poste7 pour
décider : creuser la cause (probablement poste1, qui connaît le mieux
son propre correctif MLA fraîchement fusionné, `16bac2f`) ou reporter au
lendemain.

## Script et données

`outils/equivalence-glm-2couches.py`. Logits bruts :
`/tmp/glm-equivalence-2couches/logits-acvram.json`,
`/tmp/glm-equivalence-2couches/logits-hf.json` (non versionnés, scratch).

**Incident de procédure, disjoint du verdict** : l'étape `acvram` de ce
script a tourné sur `cuda:0` (planificateur `auto_plan`, choix
automatique) sans passer par `carte.sh` — `--quant-device cpu` ne
force QUE le device de la recherche AWQ/quantification à la
conversion, pas le device d'exécution du moteur au forward. A
chevauché une mesure `ncu` de poste4 (prévenue). Le script doit être
corrigé pour forcer le CPU au forward, ou passer par `carte.sh`, avant
toute réutilisation.
