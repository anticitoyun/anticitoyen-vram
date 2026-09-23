# Verdict — énergie BRUTE, 3 moteurs : acvram ne gagne NULLE PART

## Mise à jour finale du 14/09 — bug à 2 cartes trouvé par Sage, colonne llama.cpp ajoutée

**Le classement « acvram gagne à b=1 » ci-dessous (section précédente) est
LUI-MÊME faux — Sage a trouvé que `campagne-20s-vllm-14-09.py` ne posait
pas `CUDA_VISIBLE_DEVICES=0`** (contrairement au script acvram, qui le
pose). `energie.py` IGNORE cette variable par défaut et somme TOUTES les
cartes NVML visibles — le brut vLLM de toute la colonne précédente
agrégeait donc la 3080 Ti au repos (~28,5 W) avec la 5090 réellement
mesurée. Preuve trouvée par Sage : b=12 vLLM affichait 0,2913 J × 1 437 t/s
= 418 W, AU-DESSUS du plafond de 400 W — impossible sur une seule carte
bridée.

**Corrigé** (`CUDA_VISIBLE_DEVICES=0` posé explicitement dans les trois
scripts ; troisième garde ajoutée à `energie.py.invalidations` : une
mesure `ACVRAM_TYPE=mesure` sur plus d'une carte est désormais signalée,
champ `cartes` publié dans `resume()`) et complété par une colonne
**llama.cpp** (binaire réel sm_120 de Katy, GGUF Q4_K_M) demandée par
Jérôme pour fermer le duel à trois. Mêmes rondes ≥20s, mêmes prompts
synthétiques, un chargement séparé par moteur (accord de Manon, qui fait
pp2048 sur le même llama.cpp en parallèle).

    b     acvram t/s   acvram J/j   vLLM t/s    vLLM J/j    llama.cpp t/s   llama.cpp J/j
    1     233,0        1,4296       196,7       1,3729      319,9           1,1675
    2     340,2        1,1369       306,8       0,9557      275,7           1,2396
    3     412,1        0,9493       460,1       0,6639      —               —
    4     510,4        0,7724       589,9       0,5440      —               —
    12    630,6        0,6188     1 437,9       0,2719      747,8           0,4436

**acvram ne gagne À AUCUN b testé, une fois le bug à deux cartes corrigé.**
À b=1, c'est **llama.cpp** qui gagne (1,1675 J/j, -18,4 % contre acvram,
-15,0 % contre vLLM) — la conclusion « acvram bat vLLM à b=1 » du duel
historique du 9/09 (×1,96 décodage) ne tient plus non plus dans son
régime ≥20s corrigé : llama.cpp devance largement les DEUX. À b=2 et
b=12, vLLM reste devant (b=2 : -15,9 % contre acvram, mais llama.cpp
repasse dernier ici, résultat non monotone à creuser si utile ; b=12 :
vLLM ×2,28 plus efficace qu'acvram, cohérent avec le duel historique de
Manon).

**Correction assumée, publiquement** : les sections « acvram gagne à b=1 »
puis « acvram gagne à b=2 » puis « acvram ne gagne qu'à b=1 » qui suivent
dans ce document ont chacune semblé vraies avec les données disponibles
au moment où elles ont été écrites, et se sont révélées fausses à l'étape
suivante — pour des raisons différentes à chaque fois (bruit de mesure,
puis fenêtre trop courte, puis un bug d'agrégation de cartes). Rien n'est
effacé ci-dessous : la trace complète de la correction progressive reste
lisible, avec la cause de chaque révision.

---

## Historique — étape 2 (avant le bug à 2 cartes, après la fenêtre ≥20s)

Jérôme a demandé de refaire b=1-4 ET b=12 (chiffre officiel) sur une fenêtre
**≥ 20 s** (au lieu des quelques secondes des mesures précédentes), pour que
le limiteur de puissance ait le temps d'agir et que le régime mesuré soit
celui d'un usage soutenu, pas d'un sprint. Contexte trop court (2048 jetons)
pour un `max_tokens` géant : corrigé par des RONDES répétées (jusqu'à
`max_model_len - prompt`) dans UNE seule fenêtre `Energie`, cumulées jusqu'à
≥ 20 s. `scratchpad/campagne-20s-14-09.sh` + `campagne-20s-acvram-14-09.py`
+ `campagne-20s-vllm-14-09.py`.

    b     acvram tok/s   acvram J/j BRUT   vLLM tok/s   vLLM J/j BRUT   durée (acvram/vLLM)
    1     232,99         1,4296            196,60       1,5791          23,0 / 39,0 s
    2     340,21         1,1369            306,58       1,0871          21,0 / 25,0 s
    3     412,14         0,9493            459,90       0,7479          26,0 / 25,0 s
    4     510,36         0,7724            567,78       0,6457          28,0 / 27,0 s
    12    630,59         0,6188          1 437,42       0,2913          31,0 / 32,0 s

    ratio J/jeton acvram/vLLM :  b=1 0,906  |  b=2 1,046  |  b=3 1,269  |  b=4 1,196  |  b=12 2,124

**Le tableau change de forme sur fenêtre longue.** acvram ne gagne plus
qu'à **b=1** (-9,4 %, marge PLUS LARGE que sur le coup court — pas un
artefact de bruit cette fois puisque la fenêtre est 5× plus longue).
**Le gain à b=2, confirmé par 3 répétitions sur fenêtre courte
(voir plus bas), S'INVERSE sur fenêtre ≥20s** : vLLM passe devant
(1,0871 contre 1,1369, vLLM gagne de 4,6 %). L'écart se creuse ensuite
sans discontinuer (b=3 : vLLM -21,2 % de mieux ; b=4 : -16,4 % ; b=12 :
vLLM ×2,12 plus efficace, cohérent avec le duel historique de Manon à
b=12).

**Nouveau chiffre officiel b=12 (acvram) : 630,59 t/s / 0,6188 J/jeton
brut, sur 31 s** — remplace le 712,62 t/s à 3 s du 14/09 (voir
`verdict-chiffre-officiel-duel-b12-14-09.md`) : le régime soutenu perd
~11,5 % de débit par rapport au sprint. vLLM b=12 sur 32 s : 1 437,42 t/s
/ 0,2913 J/jeton — plus rapide que le 1 198,4 t/s du duel de Manon (régime
différent, ne pas comparer chiffre pour chiffre : prompts synthétiques et
protocole de rondes ici, invites fixes chez elle).

**Interprétation honnête du renversement b=2** : les 3 répétitions du
paragraphe suivant restent EXACTES dans leur régime (fenêtres de
quelques secondes) — ce n'est pas une erreur de calcul, c'est un régime
différent qui donne une réponse différente (règle du 14/09 : un chiffre
exact hors de son régime est faux comme décision). Sur un service à
requêtes très courtes et espacées, le calcul court reste pertinent ; sur
un service à charge soutenue, c'est le calcul à 20s qui compte, et il dit
que **le créneau réel où acvram gagne, une fois la carte stabilisée, se
limite à b=1**.

---

## Historique du 14/09 (avant la campagne ≥20s) — conservé, régime « coup court »

Laure, 14/09/2026. Suite à
[`protocole-energie-brute-faible-lot-14-09.md`](protocole-energie-brute-faible-lot-14-09.md)
et au correctif de checkpoint signalé par Jérôme (le premier essai chargeait
le checkpoint acvram avec vLLM — mauvais format, OOM systématique,
`echec-energie-brute-vllm-oom-14-09.md`). Refait avec le checkpoint ModelOpt
de Manon (`/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4`,
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
donnait b=1 -3,8 % et b=2 -3,1 %. Jérôme a demandé de vérifier que la
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

**Seuil de Jérôme (« si σ > 2 pts la victoire n'est pas revendicable ») —
appliqué strictement :**

- **b=1 : NON revendicable.** σ(vLLM) = 2,43 % > 2 points, et surtout la
  marge elle-même (1,25 %) est PLUS PETITE que ce bruit — un iter2 vLLM
  aberrant (1,4639 contre 1,4042/1,4036) suffit à effacer l'écart. Je
  retire l'affirmation « acvram bat vLLM à b=1 » de mon message précédent
  à Jérôme : ce n'était pas faux sur le coup unique mesuré, mais ce n'est
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
largement l'avantage — cohérent avec le ×2,97 net déjà mesuré par Manon,
sans le confirmer chiffre pour chiffre ici.

**Deux repos, pas un** — requalifié après clarification de Jérôme (14/09) :
le 17 W acvram / 64 W vLLM cité au départ vient de Sage, et désigne le
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

Le classement « vLLM > acvram » établi par le duel b=12 de Manon **n'est
pas universel** : à b=2, sur l'énergie BRUTE (celle que paierait un
service à trafic faible, où le GPU repasse au repos entre requêtes),
**acvram gagne, de façon reproductible**. C'est le créneau que Sage
voulait faire chiffrer — chiffré, confirmé à b=2 ; PAS confirmé à b=1, où
le bruit de mesure (vLLM en particulier) dépasse l'écart mesuré.

## Réserve

Les deux moteurs chargent des checkpoints DIFFÉRENTS du même modèle de
base (acvram : conversion `nvfp4` maison ; vLLM : `NVIDIA ModelOpt FP4`
téléchargé par Manon) — comme pour tous les duels de ce dossier, ce n'est
pas un octet-pour-octet identique, c'est la meilleure quantification NVFP4
disponible de chaque côté. `n_jetons_decodes` diffère légèrement entre
moteurs (200/400/600/800 côté vLLM avec EOS ignoré exactement N_JETONS×b ;
199/398/597/796 côté acvram, à 1 jeton près par séquence — écart
négligeable, sans doute un jeton de préremplissage compté différemment).

## Modèle témoin GLM-42B (item demandé par Jérôme, repris d'Océane)

Aucun checkpoint vLLM-natif n'existe pour GLM-4.7-Grande-Heretic-42B :
seuls `models_acvram/GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4` (notre
conversion) et `models_gguf/GLM-4.7-Grande-Heretic-42B-Q4_K_M` (llama.cpp)
existent sur ce poste — pas de `NVIDIA ModelOpt FP4` équivalent à
télécharger sans passer par un nouveau téléchargement (accord utilisateur
requis, comme pour le Qwen3-Coder de Manon). vLLM sait charger du GGUF
directement (`quantization="gguf"`, vérifié dans sa signature), mais ce
fichier `.gguf` mélange un nommage 30B-A3B dans son propre nom de fichier
(`GLM-4.7-30B-A3B-20-2-Heretic-30B-A3B-Q4_K_M.gguf`) — la correspondance
exacte avec l'architecture attendue par vLLM n'est pas vérifiée, et le
chargement GGUF de vLLM est historiquement fragile sur les architectures
MoE. **Pas tenté** : nouvelle combinaison jamais essayée sur ce poste, à
la suite d'une session déjà longue — je préfère vérifier avec Jérôme avant
de lancer un chargement qui pourrait échouer de façon peu informative,
plutôt que de multiplier les tentatives sans lui.

**Suite, décidée par Jérôme (14/09)** : le chemin GGUF est écarté (fragile,
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

Item 3 de la mesure 2a de Sage : **fait**, refermé au troisième essai.
**Conclusion finale : acvram ne gagne à AUCUN des b testés (1/2/3/4/12)
une fois le bug à deux cartes corrigé** — b=1 revient à llama.cpp, b=2 et
b=12 à vLLM. Nouveau chiffre officiel b=12 acvram : 630,59 t/s / 0,6188
J/jeton (31 s), remplace le 712,62 t/s du run court. `energie.py` a reçu
une troisième garde (mesure sur >1 carte sans le dire, sous
`ACVRAM_TYPE=mesure`) suite à ce bug, plus le champ `cartes` dans
`resume()`. Racine du blocage initial (OOM vLLM) identifiée et documentée
(`echec-energie-brute-vllm-oom-14-09.md`) : ne jamais charger un checkpoint
acvram dans vLLM, les formats NVFP4 ne sont pas interchangeables entre les
deux moteurs. Modèle témoin GLM-42B : **parqué** — GGUF écarté (Jérôme),
FP8 dynamique impossible (pas de source bf16 sur le poste, vérifié dans
le manifeste acvram et sur le disque des originaux) ; à reprendre si une
source bf16 est un jour ajoutée, pas avant.
