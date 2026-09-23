# Prédiction W4A4 / W4A8 / W4A16 — Llama-2-7B, à sec avant mesure

Manon, 13/09/2026. Bead `anticitoyen-vram-brd` étape 1 (Laurine, MMA FP4
native sm_120a) : la MMA `mxf4nvf4` exige les DEUX opérandes en E2M1 par
bloc de 16, échelle UE4M3 — donc les ACTIVATIONS, pas seulement les poids.
Avant d'écrire un noyau W4A4, mesurer ce que le format coûte en qualité.

## Vérification de l'étalon cité par Laurine

**5,6102 est bien le nôtre** : `outils/campagne-quota.py` (POINTS,
plancher tout-nvfp4), `revue/courbe-du-quota-deux-points.md` et
`revue/SYNTHESE-OCEANE-10-09.md` le citent tous les trois, à 4,7297
bits/poids, régime GPTQ (wiki-gptq.txt, ctx 2048, sha
`e52922746ad09b...`). C'est le régime **W4A16** : poids NVFP4, calcul en
BF16 (aucune perte d'activation). Pas besoin de 5,4141 (celui-là est
l'étalon fp16 externe, plafond théorique hors quantification).

## Ce qui est livré, à sec

* `acvram/quant/fakequant_activation.py` : deux fonctions d'aller-retour
  (« fake quant »), réutilisant `quantize_nvfp4`/`dequantize_nvfp4` telles
  quelles pour l'E2M1 (mêmes fonctions que les poids, donc même format).
  Bloc **16 dans les deux cas** (E2M1 et E4M3) — sinon un écart entre A4
  et A8 mesurerait aussi un écart de granularité d'échelle, pas seulement
  de résolution d'élément.
* `tests/test_fakequant_activation.py` : 6 tests, tous à sec (formes,
  E4M3 plus fin que E2M1, localité par bloc de 16, tenue sur dimension non
  multiple de 16, zéro reste zéro, invariance à l'amplitude).
* `outils/fake-quant-a4-llama2-7b.py` : campagne PPL sur Llama-2-7B-hf,
  poids fake-quantifiés NVFP4 (round-trip, reproduit le régime W4A16
  actuel), activations des 7 projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`) hookées en
  `forward_pre_hook` pour trois régimes :
  - **W4A16** (témoin, hooks absents — doit reproduire 5,6102 PPL) ;
  - **W4A4** (`fake_quantize_nvfp4_activation` sur les 7 entrées) ;
  - **W4A8** (`fake_quantize_e4m3_activation`, repli mxf8f6f4).

## Prédiction chiffrée, seuil scellé

**Seuil de décision (posé par Jérôme, repris ici avant mesure)** : un
régime est acceptable si son écart de PPL relatif au témoin W4A16
(5,6102) est **≤ +1 %**, soit PPL ≤ **5,6663**.

**Prédiction W4A4** : **ÉCHEC du seuil.** La littérature de quantification
(duck.ai 12/09, Q3/Q4 : QuaRot, SpinQuant, DuQuant) est unanime — un W4A4
naïf, sans rotation ni lissage des activations, coûte largement plus
qu'1 % de PPL sur un modèle 7B ; les papiers qui atteignent un W4A4
quasi-sans-perte le font TOUJOURS avec une transformation préalable
(Hadamard, SmoothQuant). Je prédis une PPL **entre 6,5 et 12** (+16 % à
+114 % relatif), la borne haute correspondant à une dégradation qui
rendrait le modèle à peine cohérent. Fourchette large parce qu'aucune
mesure comparable n'existe encore dans ce dépôt pour ce régime précis
(fake-quant des SEPT projections simultanément, pas une seule couche
isolée comme les témoins synthétiques de `test_hadamard_downproj_nvfp4.py`).

**Prédiction W4A8** : **RÉUSSITE probable, marge étroite.** L'E4M3 a
8 bits, une résolution largement suffisante d'après la même littérature
(mxf8f6f4 est déjà en production chez plusieurs moteurs pour l'inférence).
Je prédis une PPL **entre 5,61 et 5,66** — c'est-à-dire un résultat qui
pourrait tomber juste au bord du seuil de 5,6663 selon la couche la plus
sensible (probablement `down_proj`, motif Bridging Gap déjà documenté
dans `piste-hadamard-downproj-refutee.md`).

**Issue qui me gênerait** (règle 4, nommée explicitement) : que le W4A4
naïf passe le seuil de +1 % sans rotation. Ce serait en contradiction
directe avec la littérature citée dans `revue/duck-manon-12-09.md`
(3 modèles concordants), et je me tromperais alors soit sur l'ampleur de
l'effet, soit sur un défaut de méthode (fake-quant appliqué au mauvais
tenseur, hooks qui ne couvrent pas réellement les 7 projections, ou
mesure de PPL qui ne reproduit pas le régime GPTQ étalon — à vérifier en
premier lieu par le témoin W4A16, qui DOIT reproduire 5,6102 avant que le
reste de la campagne ne vaille quoi que ce soit).

## Ce qui reste, sur carte

`python outils/fake-quant-a4-llama2-7b.py --pour-de-vrai`, quand Océane
rend la carte (~1 h de PPL par régime, donc ~3 h pour les trois — à
confirmer avec Jérôme si un sous-ensemble suffit, par exemple sauter le
témoin W4A16 puisqu'il est déjà mesuré par `campagne-quota.py` et se
contenter de vérifier qu'il reproduit 5,6102 sur UNE seule passe rapide
plutôt que la refaire en entier).
