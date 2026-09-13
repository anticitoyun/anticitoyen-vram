# Verdict — énergie BRUTE b=1-4 : acvram bat vLLM à b=1 et b=2

poste3, 14/09/2026. Suite à
[`protocole-energie-brute-faible-lot-14-09.md`](protocole-energie-brute-faible-lot-14-09.md)
et au correctif de checkpoint signalé par chef (le premier essai chargeait
le checkpoint acvram avec vLLM — mauvais format, OOM systématique,
`echec-energie-brute-vllm-oom-14-09.md`). Refait avec le checkpoint ModelOpt
de poste2 (`/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4`,
`max_model_len=4096` comme son duel A2) : chargement réussi, backend NVFP4
`VLLM_CUTLASS` confirmé dans le journal.

## Le tableau apparié

    b    acvram tok/s   acvram J/j BRUT   vLLM tok/s   vLLM J/j BRUT   watts_repos (acvram/vLLM)
    1    226,83         1,4395            198,19       1,4960          68,2 / 89,4
    2    326,15         1,2287            198,91       1,2681          76,2 / 97,1
    3    425,93         1,0011            298,48       0,9220          73,8 / 93,7
    4    540,86         0,8039            397,85       0,6702          74,0 / 94,7

    ratio J/jeton acvram/vLLM :  b=1 0,962  |  b=2 0,969  |  b=3 1,086  |  b=4 1,200
    ratio tok/s   acvram/vLLM :  b=1 1,144  |  b=2 1,640  |  b=3 1,427  |  b=4 1,360

## Prédictions du protocole : verdict mixte

**Confirmée** : « acvram bat vLLM en J/jeton BRUT à un b quelconque de 1 à
4 » — **vrai à b=1 (-3,8 %) ET b=2 (-3,1 %)**. Ce n'est pas une tendance,
c'est un résultat net : sur ces deux points, acvram consomme MOINS de
joules par jeton produit, énergie de repos incluse, sur la même carte, le
même jour.

**Pas vérifiable telle quelle** : « le ratio brut à b=12 resserre l'écart
par rapport au ratio net (2,97) » — je n'ai PAS mesuré le brut vLLM à
b=12 dans cette campagne (seulement 1 à 4), donc je ne peux pas comparer
directement. Ce que je peux dire : le crossover se produit entre b=2 et
b=3 (le ratio J/jeton passe de <1 à >1), donc à b=12 vLLM reprend
largement l'avantage — cohérent avec le ×2,97 net déjà mesuré par poste2,
sans le confirmer chiffre pour chiffre ici.

**Point non confirmé, à corriger** : le chiffre de repos cité par chef
(« nous 17 W, vLLM 64 W ») ne correspond pas à ce que je mesure ici —
**acvram 68-76 W, vLLM 89-97 W** (`base.moyenne` de `energie.py`, 8 s de
repos AVANT chaque bras, modèle déjà chargé). L'écart qualitatif est le
même sens (acvram plus économe au repos), mais l'ampleur est bien plus
faible (~20-25 % plus bas, pas ~4×). Deux lectures possibles, non
tranchées ici : (a) le 17/64 W de chef mesure un repos VRAIMENT froid
(rien chargé) alors que le mien mesure un repos « chaud » juste après un
burst de décodage (horloges pas totalement redescendues) ; (b) le chiffre
de chef vient d'une source que je n'ai pas revérifiée. Signalé plutôt
qu'ignoré.

## Ce que ça change au classement

Le classement « vLLM > acvram » établi par le duel b=12 de poste2 **n'est
pas universel** : à occupation minimale (b=1-2), sur l'énergie BRUTE (celle
que paierait un service à trafic faible, où le GPU repasse au repos entre
requêtes), **acvram gagne**. C'est le créneau que poste7 voulait faire
chiffrer — chiffré, confirmé, à b=1 et b=2 précisément.

## Réserve

Les deux moteurs chargent des checkpoints DIFFÉRENTS du même modèle de
base (acvram : conversion `nvfp4` maison ; vLLM : `NVIDIA ModelOpt FP4`
téléchargé par poste2) — comme pour tous les duels de ce dossier, ce n'est
pas un octet-pour-octet identique, c'est la meilleure quantification NVFP4
disponible de chaque côté. `n_jetons_decodes` diffère légèrement entre
moteurs (200/400/600/800 côté vLLM avec EOS ignoré exactement N_JETONS×b ;
199/398/597/796 côté acvram, à 1 jeton près par séquence — écart
négligeable, sans doute un jeton de préremplissage compté différemment).

## bd

Item 3 de la mesure 2a de poste7 : **fait**, créneau confirmé à b=1-2.
Racine du blocage initial identifiée et documentée
(`echec-energie-brute-vllm-oom-14-09.md`) : ne jamais charger un checkpoint
acvram dans vLLM, les formats NVFP4 ne sont pas interchangeables entre les
deux moteurs.
