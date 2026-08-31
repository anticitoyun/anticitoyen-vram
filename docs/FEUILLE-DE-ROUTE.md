# Ce qui est fait et ce qui ne l'est pas

Écrit sans complaisance, parce que le projet que celui-ci remplace annonçait une
extension de mémoire GPU fonctionnelle et livrait un module noyau qui se
contentait d'afficher les nombres passés en paramètres.

## Fait depuis la version 0.1.0

| | ce que ça fait | vérifié par |
|---|---|---|
| Décodage spéculatif | propositeurs n-grammes et modèle brouillon, acceptation exacte | identité des sorties + test de distribution sur 40 000 tirages |
| Cache de préfixe | réutilisation de blocs par hachage chaîné, éviction LRU | identité des sorties, comptage des jetons de prefill |
| Calcul hôte sur processeur | GEMV 4 bits AVX2/scalaire, ctypes, sans dépendance de compilation | numérique face à la référence, compilé et exécuté ici |
| GEMV CUDA réécrit | chargements vectoriels 8 octets, 4 lignes/bloc, découpe sur K | décodage des quartets validé sur processeur ; **noyau jamais compilé** |
| GEMM FP4 tensor cores | chemin `torch._scaled_mm` sondé, avec repli | la sonde rapporte elle-même sa raison dans `acvram doctor` |
| Précision mixte | promotion pilotée par le SNR, plafonnée à 15 % des tenseurs | la promotion élève le SNR mesuré, plafond respecté |
| Harnais de perplexité | `acvram eval`, fenêtre glissante | s'exécute, valeur finie, classe |
| Masque causal décalé | attention correcte pour un prefill par morceaux ou avec cache | démontré différent du drapeau intégré |
| Attention de décodage groupée | un seul appel SDPA par étape au lieu d'un par séquence | égale le résultat séquence par séquence |

Le masque causal décalé mérite une note. `F.scaled_dot_product_attention(
is_causal=True)` aligne son triangle en haut à gauche, ce qui n'est correct que
si la requête couvre toute la séquence. Le cache de préfixe et le prefill par
morceaux produisent tous deux un bloc de requêtes décalé, et le drapeau intégré
aurait masqué les mauvaises cellules — silencieusement, avec une sortie
plausible. Un test affirme que les deux diffèrent.

## Toujours jamais exécuté sur la machine cible

Tout a été écrit et testé sur un portable à i5-3230M et GeForce GT 740M en
pilote 470 : ni CUDA, ni Blackwell, ni Ampere.

* **Les noyaux CUDA n'ont jamais été compilés.** `acvram_kernels.cu` respecte
  par construction la numérique de référence PyTorch, et sa partie la plus
  délicate — le décodage des quartets — a été validée en compilant cette seule
  fonction sur processeur, mais nvcc n'a jamais vu le fichier. Attendez-vous à
  corriger des erreurs de compilation au premier build. Le chemin de référence
  étant numériquement identique, rien n'est cassé entre-temps : seulement lent.
* **Aucun débit n'est mesuré.** Chaque valeur en jetons/s est une estimation du
  planificateur à partir de plaques signalétiques. `acvram bench` les remplace.
* **Le chemin AVX2 processeur n'a jamais tourné.** Il compile, et le chemin
  scalaire qui partage sa structure est testé, mais le processeur de ce portable
  est antérieur à AVX2.
* **Le chemin FP4 tensor cores ne s'est jamais lié.** Il sonde
  `torch._scaled_mm` à l'exécution et explique son échec ; sur une vraie 5090
  avec un torch récent il se liera peut-être, ou bien la disposition des
  échelles devra être ajustée dans `_swizzle_scales`.

## Reste à faire, par ordre de valeur

1. **Un noyau CUDA d'attention paginée.** Le décodage rassemble encore le cache
   de chaque séquence dans un tenseur dense avant de calculer l'attention, ce
   qui déquantifie tout le contexte à chaque couche et à chaque étape. Un noyau
   lisant directement la table de blocs supprimerait cela. C'est désormais la
   plus grosse inefficacité structurelle restante.
2. **Cache LRU d'experts fréquents.** `tiering.py` lui réserve de la VRAM et
   modélise son taux de succès ; rien ne l'implémente. Un modèle MoE va encore
   rechercher chaque expert routé à chaque jeton.
3. **Prefill par morceaux.** La machinerie existe — le masque décalé et le
   prefill partiel fonctionnent — mais l'ordonnanceur ne découpe pas une longue
   invite, si bien qu'un très long prefill bloque encore le décodage.
4. **Spéculation à la EAGLE.** Le propositeur n-grammes est gratuit mais n'aide
   que sur des sorties répétitives ; une tête de brouillon entraînée élèverait
   l'acceptation sur du texte libre sans exiger un modèle séparé.
5. **GEMM groupé pour les MoE.** La boucle sur les experts est en Python, un
   appel par expert routé distinct.
6. **Compensation d'erreur à la GPTQ.** La calibration est la variante bon
   marché d'AWQ ; il n'y a ni mise à jour du second ordre, ni propagation de
   l'erreur d'une couche dans la calibration de la suivante.

## Limites connues

* **Ni authentification, ni limitation de débit.** Écoutez sur `127.0.0.1`
  (valeur par défaut).
* **Ni appel d'outils, ni sortie structurée.** `tools` et `response_format` sont
  acceptés et ignorés.
* **`n > 1` ne produit qu'une seule complétion.**
* **Couverture de modèles** : famille llama, dense et MoE — RMSNorm, RoPE,
  attention à requêtes groupées, SwiGLU. Cela couvre Llama, Mistral, Qwen2/3,
  Mixtral, DeepSeek. Non couverts : attention à fenêtre glissante, blocs
  Mamba/hybrides, MLA, tours visuelles.
* **La perplexité d'un modèle non entraîné ne veut rien dire.** Le harnais
  renvoie une perplexité proche de l'uniforme pour des poids aléatoires, ce qui
  est correct et rappelle aussi qu'il ne discrimine que sur un vrai modèle.

## Première heure sur la machine réelle

```bash
acvram doctor                    # ce qui s'est compilé, ce qui non, et pourquoi
acvram bench --what bandwidth    # les deux liens PCIe et la DDR ; alimente --host-gb-s
acvram bench --what kernels      # noyaux CUDA et processeur face à la référence
pytest -q                        # le chemin de référence doit toujours passer
acvram plan ~/modeles/Qwen3-32B  # le plan correspond-il à docs/MATERIEL.md
```

## Mesuré sur la machine cible (31 août 2026)

La machine (i9-14900K, RTX 5090 32 Gio, RTX 3080 Ti 12 Gio, 96 Gio DDR5) a
tranché plusieurs questions que le développement à l'aveugle laissait ouvertes.

| mesure | valeur | conséquence |
|---|---|---|
| P2P entre cartes | **indisponible** (pont chipset, pas de NVLink) | l'arête GPU↔GPU du graphe mémoire n'existe pas |
| GPU↔GPU par l'hôte | 6,6–7,2 Go/s | pire que la RAM : ne jamais *transiter* par la 3080 Ti |
| RAM épinglée → 5090 | 20,8 Go/s (x8 : les 16 lignes CPU se partagent) | hiérarchie réelle : 5090 → **RAM** → 3080 Ti *résidente* |
| lecture DDR5 | 33,5 Go/s | l'étage hôte se calcule sur place, confirmé |
| GEMV NVFP4 (5090) | 390–528 Go/s après uint4 | ~30 % du pic : marge ×3 dans le noyau |
| GEMV INT4 (3080 Ti) | 456–606 Go/s | 66 % du pic : le schéma est sain |
| Qwen3-14B NVFP4 | 9,5 → 22,3 jetons/s dans la journée | le mur restant est la surcouche Python (~2/3 du temps) |

## Enseignements des systèmes voisins, transposés ici

Lus le 31 août (FlexGen, ZeRO-Inference, TensorRT-LLM, Petals, nakshatra,
ExLlamaV2), retenu ce qui s'applique à *cette* topologie :

1. **CUDA Graphs sur le pas de décodage** — le profil montre ~68 ms de Python
   par jeton contre ~30 ms de GPU : capturer le graphe du pas mono-jeton est
   le levier n° 1, avant toute nouvelle optimisation de noyau.
2. **Affectation de formats par budget, à la EXL2** — remplacer le plancher de
   SNR fixe de la conversion par un sac à dos : quantifier chaque tenseur en
   2–3 formats candidats (déjà fait pour les promus), puis choisir l'ensemble
   qui minimise l'erreur totale sous un budget d'octets. Donne « le meilleur
   modèle qui tient dans N Gio » au lieu d'un seuil arbitraire.
3. **Prefill W4A8** — le chemin tensor-core actuel est W4A4 (~9,5 % d'erreur
   relative par lot) ; passer l'activation en FP8 (Blackwell le fait
   nativement) garderait l'essentiel de la vitesse en divisant l'erreur.
4. **Ordonnancement par blocs, à la FlexGen** — pour un modèle plus grand que
   la VRAM en mode débit : réutiliser chaque couche streamée sur tout le lot
   avant de la remplacer. La machinerie de streaming existe ; c'est
   l'ordonnanceur qui traite aujourd'hui séquence par séquence.
5. **Chargement bi-lien, à la ZeRO-Inference** — les deux liens x8 sont
   indépendants : pour l'étage hôte d'un très grand modèle, chaque carte peut
   tirer sa moitié de couche (≈ 33 Go/s cumulés). *Sans* l'échange GPU↔GPU
   final qui suit chez eux — ici il coûterait plus qu'il ne rapporte.
6. **Brouillon spéculatif sur la carte secondaire** — nakshatra mesure ×2,3–2,7
   sur silicium comparable ; `--speculative draft --draft-device cuda:1` existe
   déjà, il manque le banc qui le prouve ici.

Ce qui ne se transpose **pas** : le placement pair-à-pair de Petals (fait pour
un essaim, pas deux cartes), le KV 4 bits de FlexGen (mesuré ici : INT8 par
(jeton, tête) bat le FP8, et 4 bits dégraderait), l'échange inter-GPU de
ZeRO-Inference (lien plus lent que la RAM).

## Version 0.3.0 (31 août 2026, soir)

* **Graphes CUDA sur le pas de décodage** : capture par godet (lot, blocs KV),
  logits identiques au bit près à l'eager — le chemin fixe est partagé.
  Qwen3-14B : 22,3 → 27,3 jetons/s.
* **Source GGUF** : `acvram convert` lit les .gguf de llama.cpp
  (F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS) et
  reconstruit config, tokenizer BPE et jetons d'arrêt depuis l'en-tête.
  Vérifié de bout en bout sur Qwen3-0.6B-Q8_0 et Qwen3-4B-Q4_K_M.
  Style SentencePiece non reconstruit : passer --tokenizer.
* **Rangement** : les modèles convertis vont par défaut sous
  `/mnt/4TO_SATACMR_2022/Modeles/models_acvram/` (ou `$ACVRAM_MODELS_DIR`).
* **Sampler** : raccourci glouton (argmax direct quand tout le lot est à
  température nulle, pénalités éteintes).
* **Mesuré et tranché** : le brouillon spéculatif 0.6B (cuda:1) devant le 14B
  fait *chuter* le débit (27,3 → 14–18 jetons/s) : la vérification
  (query_lens > 1) est inéligible aux graphes et repasse par l'eager. Rendre
  la spéculation graph-compatible est le prérequis avant de la recommander ;
  le bug de périphérique des probabilités du brouillon est corrigé au passage.
* **EXL3** : format identifié (treillis QTIP, Hadamard signés, codebook MCG) ;
  décodeur à écrire avec exllamav3 installé comme oracle — sans oracle, un
  décodeur faux produit du charabia silencieux.

## Ajouts du 31 août, nuit

* **Source EXL3** : `acvram convert` lit les modèles exllamav3 en déléguant la
  reconstruction du treillis QTIP à `exllamav3` (dépendance optionnelle, de
  conversion seulement — le converti n'en dépend plus). Vérifié :
  Cydonia-24B 6 bpw → NVFP4, servi, 16 jetons/s. L'extension d'exllamav3 se
  compile avec le nvcc des roues pip, même détour que nos noyaux.
* **Registre de backends** (`kernels/backends.py`) : l'abstraction demandée —
  `(format, périphérique) → [backends par priorité]`, repli en cascade jusqu'à
  la référence PyTorch. Quatre backends livrés : cuda-fusionné (sm_86+),
  fp4-tensorcores (sm_100+), cpu-avx2, référence. `acvram doctor` affiche la
  table retenue. En ajouter un (CUTLASS, cuBLASLt, AVX-512) est un
  `register()` — le moteur n'y touche pas. Débit inchangé (dispatch mémoïsé).
* **Coffre de jetons** : `~/.config/acvram/jetons-acvram.sh` — JSON
  {projet: jeton} chiffré AES-256 (gpg symétrique), `ajouter/projets/jeton`,
  et `git-credential-acvram` le sert à git pour outils.nuages.noho.st.

## GEMV NVFP4 optimisé pour Blackwell (1er septembre 2026)

Le goulot n'était pas le schéma mémoire mais le décodage : la table `kE2M1` en
mémoire `__constant__`, indexée par les bits de chaque poids, se sérialise dès
que les fils d'un warp lisent des entrées différentes — c'est-à-dire toujours.
Remplacée par la conversion FP4→half2 *native* de Blackwell
(`__nv_cvt_fp4x2_to_halfraw2`, un octet = deux poids, émulée sans accès mémoire
sur les architectures plus anciennes), plus des fils dimensionnés sur les
paires de blocs que la boucle consomme réellement.

| forme | avant | après | part du pic (1792 Go/s) |
|---|---|---|---|
| 4096×4096 | 390 Go/s | 786 Go/s | 44 % |
| 14336×4096 | 480 Go/s | 1155 Go/s | **64 %** |
| 5120×5120 | 445 Go/s | 852 Go/s | 48 % |
| lm_head 151936×5120 | 528 Go/s | 1031 Go/s | 58 % |

Qwen3-14B de bout en bout : 27,3 → **31,6 jetons/s** (9,5 au début de la
journée). Huit lignes par bloc ont été essayées et retirées : la pression de
registres l'emporte, mesuré plus lent partout. Le prochain palier du décodage
n'est plus le GEMV : à 31,6 jetons/s, le pas se partage entre ~8 ms de GEMV,
l'attention, le cache KV et ~10 ms de Python autour du graphe.
