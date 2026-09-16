# Verdict — fréquence d'activation des experts GLM : CONFORME (17/09)

poste1, à sec, ordre poste7 (`revue/poste7-avis-exterieur-16-09.md` § point 2,
repris le 17/09 par `revue/poste7-plan-completion-comparatif-17-09.md` § 3,
« résidence dynamique des experts, scellé inchangé »).

## Protocole

`GLM-4.7-Flash-srcbf16-nvfp4` (converti alpha-commun), CPU, un seul
prefill dense (une passe, corpus réel `wiki-gptq.txt`). Aucune sonde à
poser : `MoEBlock._usage_routage` (`model.py:615`) est un histogramme
par expert **toujours actif**, accumulé à chaque pas (chantier colibrì) —
lu après coup sur les 46 couches MoE (`first_k_dense_replace=1`, couche
0 dense). Par couche : fraction du routage total portée par les 50 %
d'experts les plus sollicités (32/64). Script
`outils/frequence-routage-glm-17-09.py N_JETONS`.

## Mesuré

**2048 jetons réels, 8192 décisions de routage par couche (top-4 × 2048).**

| | valeur |
|---|---|
| moyenne (46 couches) | **84,77 %** |
| médiane | 86,45 % |
| pire couche (n° 1) | **65,34 %** |
| meilleure couche (n° 22) | 92,79 % |

Pilote à 64 jetons (256 décisions/couche, moins robuste) : moyenne
93,26 %, pire couche 72,27 % — même verdict, cohérent avec la mesure
finale (les petits échantillons concentrent artificiellement, la mesure
à 2048 jetons est la référence).

## VERDICT : CONFORME à l'attendu (≥ 80 %)

Moyenne 84,77 % ≥ 80 % attendu ; **même la pire couche (65,34 %) reste
au-dessus du seuil de réfutation (60 %)** — aucune couche ne menace le
chantier. La résidence dynamique des experts vaut la peine sur ce MoE :
gain attendu exil ×9 → ×2-3 (`poste7-avis-exterieur-16-09` § point 2)
tient.

## Défaut trouvé en mesurant (hors périmètre de cette tâche, signalé)

Au-delà de `ACVRAM_INSTA_PAS` jetons (256 par défaut), le prefill lève
`RuntimeError: No CUDA GPUs are available` dans
`Engine._vers_hote` (`runner.py:666`, appelée par `_photographier`,
`runner.py:696`) : `torch.empty_like(etat, device="cpu",
pin_memory=True)` exige un contexte CUDA même pour un tenseur CPU, y
compris quand `CUDA_VISIBLE_DEVICES=""`. Touché ici parce que GLM entre
dans la branche `est_hybride`/`_frontiere_insta` du moteur (à vérifier
pourquoi — `layer_types` de GLM ne devrait porter que de la MLA, pas de
récurrence linéaire). Contourné pour cette mesure par
`ACVRAM_INSTA_PAS=4096` (> 2048 jetons, aucune frontière traversée) —
pas corrigé, hors périmètre de la tâche confiée. À signaler pour
quiconque lance un prefill à sec de plus de 256 jetons sur GLM.

## Conséquence

Le chantier « résidence dynamique des experts » reste à sec (§3 de
`poste7-plan-completion-comparatif-17-09` : pas de carte avant la fin du
comparatif à cinq). Prochaine étape naturelle (pas commencée, pas
demandée dans cet ordre) : dimensionner combien d'experts résidents en
VRAM suffisent à cette concentration (32/64 porte 84,77 % en moyenne —
un seuil plus bas, p. ex. 40 %, à chiffrer si poste7 le demande).
