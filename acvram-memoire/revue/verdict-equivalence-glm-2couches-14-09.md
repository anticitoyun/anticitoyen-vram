# Verdict — équivalence CPU 2 couches GLM-4.7-Flash vs HF (item 2 de Sage)

Manon, 14/09/2026 soir. Seuil scellé par Sage
(`revue/sage-lancement-14-09.md` §2) : max |Δlogit| ≤ 5·10⁻² **et**
cosinus ≥ 0,999 sur les logits des 16 positions d'une invite commune,
forward CPU bf16 des couches 0-1 (dense + première MoE), acvram contre HF
transformers. Réfuté → pas de conversion ce soir.

## Résultat

RÉFUTÉ, **reproduit sur deux passages indépendants** — un premier par
mégarde sur GPU (incident de procédure, voir plus bas), un second corrigé
et vérifié entièrement CPU (`device_override="cpu"` + `--host-exec cpu`,
aucun PID sur `nvidia-smi` pendant le passage). Les deux passages
donnent le même ordre de grandeur :

| passage | pire \|Δlogit\| | pire cosinus | régime |
|---|---|---|---|
| 1 (GPU, incident) | 5,0669 | 0,952137 | DÉGRADÉ (piles_ok=False) |
| 2 (CPU, propre) | 5,0124 | 0,954475 | NOMINAL (piles_ok=None) |

Détail du passage 2 (CPU, retenu comme résultat de référence — celui
versionné dans `donnees-equivalence-glm-14-09/`) :

| position | \|Δlogit\| max | cosinus |
|---|---|---|
| 0 | 3,5411 | 0,998692 |
| 1 | 2,0248 | 0,997028 |
| 2 | 1,9315 | 0,997851 |
| 3 | 0,3509 | 0,998970 |
| 4 | 0,1925 | 0,999777 |
| 5 | 5,0124 | 0,954475 |
| 6 | 1,4751 | 0,999855 |
| 7 | 1,5771 | 0,994733 |
| 8 | 1,2026 | 0,999014 |
| 9 | 2,5615 | 0,995985 |
| 10 | 0,8620 | 0,999905 |
| 11 | 0,8429 | 0,998694 |
| 12 | 1,9596 | 0,996858 |
| 13 | 4,8674 | 0,976307 |
| 14 | 1,4890 | 0,999249 |
| 15 | 3,1620 | 0,970630 |

Pire |Δlogit| : 5,0124 (seuil 0,05, ×100). Pire cosinus : 0,954475 (seuil
0,999). **Aucune des 16 positions ne passe le critère Δlogit** sur aucun
des deux passages. Les deux critères sont exigés ensemble.

**Ce que la reproduction CPU tranche** : le régime passe de DÉGRADÉ
(GPU) à NOMINAL (CPU, `piles_ok=None` — pas de couche MoE vérifiée par
pile car le chemin CPU n'utilise jamais la construction par pile),
**et l'écart reste le même ordre de grandeur**. La cause « formats
mélangés » du passage 1 était donc bien un artefact du chemin GPU
(la question posée plus bas est résolue : lecture 1, faux
déclenchement — voir section suivante), **pas** la source de l'écart.
L'écart lui-même est donc probablement réel, dans le calcul MLA/MoE
lui-même, indépendant du device.

## Écarté avant de conclure

**Tenseurs manquants dans l'extraction** — écarté par mesure directe :
les 192 tenseurs des 64 experts (gate/up/down_proj) de la couche 1 (MoE)
sont présents un-à-un dans le mini-répertoire extrait, comparés à la
source (`model.safetensors.index.json`). Pas de perte de données côté
extraction.

## Tranché : le message « formats mélangés » est un artefact GPU

Passage 1 (GPU) affichait, sur une conversion **bf16 pure, sans AWQ,
sans quantification** : `régime DÉGRADÉ — piles_ok=False`, cause
`« gate_proj : formats de quantification mélangés entre experts (ni
tout NVFP4, ni tout INT4) »` (message ajouté par moi le 14/09,
`acvram/engine/model.py::_try_build_stacks`). Passage 2 (CPU) ne
l'affiche PAS (`piles_ok=None`, régime NOMINAL) — le chemin CPU ne passe
jamais par la construction en pile (`_try_build_stacks` est un optimiseur
GPU), donc ce contrôle ne s'y exécute pas du tout. **L'écart de logit
étant identique sur les deux passages**, le message DÉGRADÉ n'explique
rien : c'est un faux déclenchement du contrôle d'homogénéité sur du bf16
non quantifié (lecture 1 de la section précédente, confirmée), sans
rapport avec la cause réelle de l'écart.

## Ce qui n'a pas été fait

- Pas de décomposition couche par couche (attention seule vs MLP seul)
  pour isoler où l'écart apparaît en premier — c'est le diagnostic
  qu'Océane prend en charge (couche 0 dense+MLA seule, puis routage MoE).
- Pas de conversion, comme l'exige le protocole sur un verdict réfuté.

## Suite

Pas de conversion GLM-4.7-Flash ce soir. Océane prend le diagnostic par
couche (couche 0 dense+MLA seule, puis routage MoE GLM : sigmoid + bias +
expert partagé) sur son correctif `16bac2f`.

## Script et données

`outils/equivalence-glm-2couches.py`. Logits bruts versionnés :
`revue/donnees-equivalence-glm-14-09/logits-acvram.json` (passage 2,
CPU),  `revue/donnees-equivalence-glm-14-09/logits-hf.json`.

**Incident de procédure du passage 1, corrigé** : l'étape `acvram` a
d'abord tourné sur `cuda:0` (planificateur `auto_plan`, choix
automatique) sans passer par `carte.sh` — `--quant-device cpu` ne force
QUE le device de la recherche AWQ/quantification à la conversion, pas le
device d'exécution du moteur au forward. A chevauché une mesure `ncu` de
Laurine (prévenue). Corrigé pour le passage 2 : `device_override="cpu"`
explicite dans `load_model()`, `--host-exec cpu` à la conversion,
lancé via `systemd-run` (le harnais a tué le premier essai post-
correctif sur un faux « mémoire faible » — `free -h` montrait 68 Gio
disponibles) ; vérifié sans PID sur `nvidia-smi` pendant tout le passage.
