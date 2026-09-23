# Verdict — échelle de bloc ModelOpt : recherchée, pas amax/6 (16/09)

Océane, à sec, ordre Sage §6, suite à `98d1e87` (déquant/requant NVFP4
modelopt DIFFÉRENT).

## Protocole

Sur les tenseurs BRUTS modelopt (`weight` u8, aucun redéquant/requant) :
sous la règle « échelle de bloc = amax_bloc/6 », l'élément qui a défini
l'amax d'un bloc de 16 doit porter le code de magnitude 6 (valeur E2M1
maximale, `mag = code & 0x7 == 7`). Fraction de blocs SANS aucun code à
magnitude 6, sur les mêmes 3 tenseurs que `98d1e87`. Script
`outils/verif-modelopt-bloc-amax-16-09.py`.

## Mesuré

| tenseur | blocs sans code à magnitude 6 |
|---|---|
| lm_head.weight | 42,41 % |
| gate_proj (couche 0) | 49,75 % |
| down_proj (couche 0) | 49,70 % |
| **poolé (3 tenseurs)** | **43,31 %** (39 242 232 / 90 603 520) |

## VERDICT : > 20 % → échelle de bloc réellement cherchée par ModelOpt

Bien au-delà du seuil scellé (43,3 % pooled, chaque tenseur > 40 %) —
pas un instrument désaligné (qui donnerait ≈ 0 %). Fait notable, non
demandé mais qui corrobore directement `98d1e87` : ces fractions
(42,4 % / 49,75 % / 49,70 %) sont **quasi identiques**, tenseur par
tenseur, aux fractions de blocs dont l'échelle différait déjà mesurées
dans `98d1e87` (42,4 % / 49,75 % / 49,70 % — mêmes chiffres). Les deux
mesures pointent le même mécanisme : chaque fois que l'échelle de bloc
n'est pas `amax_bloc/6`, c'est précisément parce qu'aucun élément du
bloc n'a été poussé jusqu'au code 6 — ModelOpt choisit une échelle plus
grande que ce que l'amax brut imposerait (probablement pour absorber un
outlier ou suivre une statistique de calibration), pas une coïncidence
de deux mesures indépendantes.

## Conséquence

Confirme et durcit `98d1e87` : le passage direct de Laurine
(`sage-convertisseur-formats` § 3.1) doit **copier** `weight`/
`weight_scale`/`weight_scale_2` tels quels dans `NVFP4Tensor` — jamais
tenter de les redériver par un `quantize_nvfp4` même corrigé, sauf à lui
ajouter une vraie recherche d'échelle (calibration) équivalente à celle
de ModelOpt, ce que Sage nomme explicitement comme un chantier à part
si le passage direct devait un jour requantifier plutôt que copier.
