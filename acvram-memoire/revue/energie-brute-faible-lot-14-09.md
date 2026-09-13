# Verdict — énergie BRUTE b=1-4 : acvram bat vLLM à b=2, PAS à b=1 (bruit)

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

**Confirmée à b=2 seulement, après répétition** : « acvram bat vLLM en
J/jeton BRUT à un b quelconque de 1 à 4 ». La mesure au coup unique
donnait b=1 -3,8 % et b=2 -3,1 %. chef a demandé de vérifier que la
marge tient : 3 répétitions alternées (acvram, vLLM, acvram, vLLM, acvram,
vLLM), repos 30 s, `scratchpad/campagne-repetition-b1-b2-14-09.sh`.

    b=1, J/jeton BRUT     iter1    iter2    iter3    moyenne   σ      σ %
    acvram                1,4026   1,4064   1,4091   1,4060    0,0033  0,23 %
    vLLM                  1,4042   1,4639   1,4036   1,4239    0,0346  2,43 %
    marge (vLLM-acvram)/vLLM = 1,25 %

    b=2, J/jeton BRUT     iter1    iter2    iter3    moyenne   σ      σ %
    acvram                1,1975   1,2126   1,2095   1,2065    0,0080  0,66 %
    vLLM                  1,2380   1,2412   1,2603   1,2465    0,0121  0,97 %
    marge (vLLM-acvram)/vLLM = 3,21 %

**Seuil de chef (« si σ > 2 pts la victoire n'est pas revendicable ») —
appliqué strictement :**

- **b=1 : NON revendicable.** σ(vLLM) = 2,43 % > 2 points, et surtout la
  marge elle-même (1,25 %) est PLUS PETITE que ce bruit — un iter2 vLLM
  aberrant (1,4639 contre 1,4042/1,4036) suffit à effacer l'écart. Je
  retire l'affirmation « acvram bat vLLM à b=1 » de mon message précédent
  à chef : ce n'était pas faux sur le coup unique mesuré, mais ce n'est
  pas un résultat qui tient à la répétition. **Correction assumée, pas
  cachée** (règle 8/09, une estimation ne se corrige pas en silence).
- **b=2 : revendicable.** σ des deux moteurs (0,66 % et 0,97 %) reste sous
  2 points, et la marge (3,21 %) dépasse le pire des deux σ d'un facteur
  ≥3. **acvram bat vLLM en J/jeton BRUT à b=2, résultat qui tient à la
  répétition.**

**Pas vérifiable telle quelle** : « le ratio brut à b=12 resserre l'écart
par rapport au ratio net (2,97) » — je n'ai PAS mesuré le brut vLLM à
b=12 dans cette campagne (seulement 1 à 4), donc je ne peux pas comparer
directement. Ce que je peux dire : le crossover se produit entre b=2 et
b=3 (le ratio J/jeton passe de <1 à >1), donc à b=12 vLLM reprend
largement l'avantage — cohérent avec le ×2,97 net déjà mesuré par poste2,
sans le confirmer chiffre pour chiffre ici.

**Deux repos, pas un** — requalifié après clarification de chef (14/09) :
le 17 W acvram / 64 W vLLM cité au départ vient de poste7, et désigne le
**repos FROID** (rien chargé sur la carte). Ce que je mesure ici est un
repos **CHAUD** — `base.moyenne` de `energie.py`, 8 s juste après un burst
de décodage, modèle déjà chargé, horloges pas totalement redescendues :
**acvram 68-76 W, vLLM 89-97 W**. Les deux repos sont réels et mesurent
des choses différentes : le froid dit ce qu'un service payé entre deux
requêtes très espacées (redescente complète) coûterait ; le chaud dit ce
qu'il coûte pour un trafic qui ne laisse jamais la carte redescendre
complètement (le cas mesuré par ce protocole, repos de 8 s seulement).
Dans les deux régimes, acvram reste sous vLLM (froid ×3,8, chaud ~20-25 %
plus bas) — le sens ne change pas, l'ampleur si, selon l'espacement réel
du trafic visé.

## Ce que ça change au classement

Le classement « vLLM > acvram » établi par le duel b=12 de poste2 **n'est
pas universel** : à b=2, sur l'énergie BRUTE (celle que paierait un
service à trafic faible, où le GPU repasse au repos entre requêtes),
**acvram gagne, de façon reproductible**. C'est le créneau que poste7
voulait faire chiffrer — chiffré, confirmé à b=2 ; PAS confirmé à b=1, où
le bruit de mesure (vLLM en particulier) dépasse l'écart mesuré.

## Réserve

Les deux moteurs chargent des checkpoints DIFFÉRENTS du même modèle de
base (acvram : conversion `nvfp4` maison ; vLLM : `NVIDIA ModelOpt FP4`
téléchargé par poste2) — comme pour tous les duels de ce dossier, ce n'est
pas un octet-pour-octet identique, c'est la meilleure quantification NVFP4
disponible de chaque côté. `n_jetons_decodes` diffère légèrement entre
moteurs (200/400/600/800 côté vLLM avec EOS ignoré exactement N_JETONS×b ;
199/398/597/796 côté acvram, à 1 jeton près par séquence — écart
négligeable, sans doute un jeton de préremplissage compté différemment).

## Modèle témoin GLM-42B (item demandé par chef, repris d'poste1)

Aucun checkpoint vLLM-natif n'existe pour GLM-4.7-Grande-Heretic-42B :
seuls `models_acvram/GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4` (notre
conversion) et `models_gguf/GLM-4.7-Grande-Heretic-42B-Q4_K_M` (llama.cpp)
existent sur ce poste — pas de `NVIDIA ModelOpt FP4` équivalent à
télécharger sans passer par un nouveau téléchargement (accord utilisateur
requis, comme pour le Qwen3-Coder de poste2). vLLM sait charger du GGUF
directement (`quantization="gguf"`, vérifié dans sa signature), mais ce
fichier `.gguf` mélange un nommage 30B-A3B dans son propre nom de fichier
(`GLM-4.7-30B-A3B-20-2-Heretic-30B-A3B-Q4_K_M.gguf`) — la correspondance
exacte avec l'architecture attendue par vLLM n'est pas vérifiée, et le
chargement GGUF de vLLM est historiquement fragile sur les architectures
MoE. **Pas tenté** : nouvelle combinaison jamais essayée sur ce poste, à
la suite d'une session déjà longue — je préfère vérifier avec chef avant
de lancer un chargement qui pourrait échouer de façon peu informative,
plutôt que de multiplier les tentatives sans lui.

**Suite, décidée par chef (14/09)** : le chemin GGUF est écarté (fragile,
résultat non défendable) ; l'alternative proposée était vLLM en
`--quantization fp8` (dynamique, à la volée, sans conversion préalable) à
partir de la source **bf16**, si elle est sur le poste. **Vérifié dans
`acvram_manifest.json`** de notre propre conversion
(`GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4/acvram_manifest.json`, champ
`model.name`) : **la source de notre conversion est déjà le GGUF Q4_K_M**
(`GLM-4.7-30B-A3B-20-2-Heretic-30B-A3B-Q4_K_M.gguf`), pas un bf16 — et
aucun bf16 HF de ce modèle n'existe ailleurs sur le poste (vérifié sous
`/mnt/4TO_SATACMR_2022/Modeles`, seul le même GGUF y est présent). **Point
parqué, comme convenu : pas de téléchargement sans l'utilisateur.**

## bd

Item 3 de la mesure 2a de poste7 : **fait, avec la correction de la marge
b=1**. Racine du blocage initial identifiée et documentée
(`echec-energie-brute-vllm-oom-14-09.md`) : ne jamais charger un checkpoint
acvram dans vLLM, les formats NVFP4 ne sont pas interchangeables entre les
deux moteurs. Modèle témoin GLM-42B : **parqué** — GGUF écarté (chef),
FP8 dynamique impossible (pas de source bf16 sur le poste, vérifié dans
le manifeste acvram et sur le disque des originaux) ; à reprendre si une
source bf16 est un jour ajoutée, pas avant.
