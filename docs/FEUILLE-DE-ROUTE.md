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

## Campagne de validation du 1er septembre — tout ce qui tourne, tout ce qui ne tourne pas

Testé sur machine, modèle par modèle, fonctionnalité par fonctionnalité :

| quoi | verdict |
|---|---|
| API (12 points : routes, flux SSE, stop, sampling, concurrence ×4, cache de préfixe ×6, usage, embeddings) | ✔ 12/12 après deux correctifs (stop exclu de la sortie ; deltas SSE sans null) |
| safetensors + AWQ (Qwen3-14B, **Nemo-12B-Claude** 28 j/s) | ✔ |
| GGUF Q8_0 / Q4_K_M (Qwen3-0.6B, 4B) | ✔ |
| EXL3 dense (Cydonia-24B) et **MoE** (Qwen3-Coder-30B-A3B, 128 experts) | ✔ — première exécution réelle du chemin MoE |
| **Pipeline hétérogène** --gpus all : NVFP4 sur 5090 + INT4 sur 3080 Ti, un seul modèle | ✔ — première exécution réelle du concept fondateur |
| eval (perplexité), ngram, --fp16, --device cuda:1 | ✔ |
| Modèles « kimi » locaux (kimi-linear, qwen35, qwen35moe) | ✘ hybrides SSM/DeltaNet : refus **explicite** à la conversion — les convertir produisait du charabia silencieux |
| GGUF en fragments multiples | ✘ refus explicite (llama-gguf-split --merge) |

Chantiers de débit relevés par la campagne, par ordre de valeur :
1. **GEMM groupé MoE** : 7-8 j/s seulement sur 3B actifs — la boucle Python
   par expert lance ~1150 petits GEMV par jeton, et le MoE est inéligible aux
   graphes CUDA.
2. **Noyau d'attention paginée fusionné** (int8 → attention sans
   matérialisation bf16) — décisif au long contexte.
3. Profil de la 3080 Ti en NVFP4 (9 j/s sur le 3B, anormalement bas).

## Attention paginée fusionnée + MoE groupé (1er septembre, suite)

* **Noyau d'attention paginée** (`paged_attention`, flash-decoding en deux
  noyaux) : le cache INT8 est lu une fois et déquantifié en registres — plus
  de matérialisation bf16, plus de copie GQA. Formes fixées par le godet de
  blocs, donc rejouable en graphe CUDA ; chemin unique eager/graphe, égalité
  affirmée par les tests. **Contexte 7 000 : 15,4 → 39,6 jetons/s (×2,6)** ;
  court contexte : 32. `ACVRAM_DISABLE_PAGED_ATTN=1` pour revenir au chemin
  déquantifier-puis-SDPA (qui reste la référence des tests).
* **MoE groupé sous graphes** : 7-8 → 17-19 jetons/s sur Qwen3-Coder-30B-A3B.
* **Reliquat Python du pas** : mesuré à ~0,2 ms (construction du lot,
  remplissage, plongement) — les 21 ms attribuées à l'embedding par cProfile
  étaient l'attente GPU imputée au mauvais site. Le pas est GPU-borné.
* **Paquet Debian** : `tools/construire-deb.sh` → `acvram_0.3.0_amd64.deb`.

## Banc uniforme du 1er septembre (200 jetons, greedy, graphes chauds)

| modèle | source | jetons/s | premier passage |
|---|---|---|---|
| Qwen3-0.6B | GGUF Q8_0 | **67** | 45 |
| Qwen2.5-Coder-3B | safetensors+AWQ | **57** | 16 |
| Qwen3B-pipeline (5090+3080 Ti) | safetensors | **37** | — |
| Qwen3-4B | GGUF Q4_K_M | **49** | 32 |
| Nemo-12B-Claude | safetensors+AWQ | **32** | 28 |
| Qwen3-14B | safetensors+AWQ | **30** | 9,5 |
| Cydonia-24B | EXL3 6 bpw | **28** | 16 |
| Qwen3-Coder-30B-A3B (MoE 128 exp.) | EXL3 4 bpw | **24** | 7-8 |

Les trois modèles « kimi » locaux restent refusés à la conversion (hybrides
SSM/DeltaNet), message explicite vérifié sur les trois.

L'amorce du paquet .deb a été exécutée de bout en bout hors dpkg (arbre
extrait, ACVRAM_HOME isolé) : venv, torch cu130, nvcc des roues, doctor
complet. Elle a révélé et fait corriger la résolution des backends par *type*
de périphérique — la 3080 Ti héritait du chemin FP4 de la 5090, et un échec
FP4 sur elle aurait éteint le chemin pour les deux cartes.

## Nuit du 1er au 2 septembre — chantiers 1 à 7

| # | chantier | état |
|---|---|---|
| 1 | écart llama.cpp | profil MoE fait : par pas de 20 ms GPU, ~9 ms de micro-noyaux elementwise (55 000 lancements de 1-2 µs : silu/mul/copies du chemin MoE), ~5 ms d'int8 (attention promue), ~2,5 ms de GEMV groupé à 380 Go/s. Cibles chiffrées : fusion silu×up dans le noyau groupé, attention MoE en nvfp4 non promue, agrandir les tuiles du groupé. |
| 2 | spéculation sous graphes | fait et testé (bit-exact) ; verdict réel : le brouillon inter-GPU reste perdant (16-20 contre 36 t/s) — le goulot est le brouillon eager + le lien 7 Go/s, plus la vérification. ngram conservé pour la recopie. |
| 3 | prefill W4A8 | fait ; en réel : 3 400 jetons/s dans les trois modes (le prefill est borné par l'attention, pas les GEMM) → la précision ×2,4 est gratuite, défaut a8. |
| 4 | budget de bits (sac à dos) | fait : convert --bits-budget, testé serré/large. |
| 5 | conversions en lot | fait : 6 nouveaux modèles valides (parc acvram = 14) ; 31 refus SSM attendus ; 5 conversions mutilées détectées → garde-fou de complétude + refus des archs non traduites. |
| 6 | Gated DeltaNet | cœur mathématique validé contre transformers (prefill < 1e-4, continuité décodage < 1e-3) ; reste mapping GGUF → loader → états par séquence (docs/CHANTIER-GDN.md). |
| 7 | étage RAM du KV | fait : HostKVPool, spill à l'éviction, remontée à l'admission, --host-kv-gib. |

## Investigation TabbyAPI denses 24-31B (1er sept. 2026)

Les 12-19 t/s des denses EXL3 sur TabbyAPI ne sont **pas un bug de
configuration** : reproduits sur GPU libre, chargement propre, 5090 seule
(cydonia 6bpw : 16,2 t/s). Cause : le décodage trellis EXL3 est borné par le
calcul, pas la bande passante — ~290 Go/s effectifs (16 % du pic 5090), et
6bpw aggrave. Les MoE A3B y échappent (3 Go actifs → 91-96 t/s). Remède :
servir les denses par acvram (cydonia : 35 t/s en AWQ safetensors, 26 depuis
l'EXL3 6bpw) ou llama.cpp ; garder TabbyAPI pour les MoE et les petits.

## 2 septembre 2026 — tout le parc, puis les pistes 1 à 7

Règle de la campagne : chaque changement monte la version (pyproject +
`acvram/__init__.py`), commit, push.

| version | contenu | validation |
|---|---|---|
| 0.4.2 | starcoder2 (LayerNorm avec biais, MLP GELU non gaté, fenêtre 4096) | chat cohérent |
| 0.4.3 | Muse-Glimmer 30B (EXL3) : normes centrées +1, plongement RMS-normalisé par ligne, `gate_proj` fusionné par tête dans `q_proj` (chemin `output_gate`), q/k normalisés ×3,87, fenêtres 2048, logits × 0,196 puis softcap 20 | chat cohérent |
| 0.4.4 | `acvram/quant/hfquant.py` : AWQ gemm, compressed-tensors `pack-quantized` (sym/asym) et `nvfp4-pack-quantized`, modelopt NVFP4, déquantifiés en bf16 à la volée puis requantifiés par nos soins | Dolphin (asym), code-qwen3-32b (nvfp4) |
| 0.4.5 | ERNIE-4.5-MoE : softmax + biais de sélection, experts partagés fusionnés, RoPE entrelacé → q/k dé-permutés à la conversion | en cours |
| 0.4.6 | types GGUF à grille (IQ1/IQ2/IQ3, TQ) par le `gguf-py` de llama.cpp | LFM2.5 IQ3_M cohérent |
| 0.4.7 | Nemotron-H et LFM2/LFM2-MoE depuis HF/EXL3 (`backbone.layers.N.mixer.*`, `feed_forward.w1/w3/w2`), gemma4_unified, tokenizer reconstruit depuis `tekken.json` | LFM2 EXL3 cohérent |
| 0.4.8 | **piste 1 : spéculation n-gram sur les hybrides sous graphes** — le lot de vérification (forme fixe k+1) déroule les jetons un à un dans les tampons fixes et photographie l'état après chacun ; retour au dernier accepté | 4B kimi : greedy identique 96/96 ; prose 110 → 153 t/s, liste 108 → 132, code 109 → 98 (acceptation 0,52) |
| 0.4.9 | corrections de 0.4.4 : ordre AWQ inverse `[0,4,1,5,2,6,3,7]`, décalage +8 de compressed-tensors (pas de complément à deux), modelopt fp4 empaqueté (`weight` u8 + `weight_scale_2`) | reconversions en file |
| 0.4.10 | **piste 4 : lots b>1 sous graphes pour les hybrides** — un créneau de tampons fixes par séquence, attention linéaire déroulée par créneau, GEMV partagées (`ACVRAM_HYBRID_SLOTS`, 4) | identique au décodage seul ; 104 t/s seul → 122 (b=2) / 149 (b=3-4) agrégés |
| 0.4.11 | gemma4 HF/EXL3 (`layer_scalar` renommé) ; Nemotron-H HF : `num_layers`, rognage des tenseurs rembourrés à 128 par EXL3 ; garde GGUF `ACVRAM_GDN` active par défaut (`=0` pour refuser) | Nemotron-Nano-9B bf16 HF cohérent |
| 0.4.12 | **piste 6** : `nvfp4_dequant_kernel` écrit 16 poids d'un coup et applique l'échelle globale par expert (124 → 1 200 Go/s, identique bit à bit) | prefill Qwen3-Coder-30B 4 096 jetons : 3 619 → **6 420 j/s** (déquantification 581 → 63 ms) |
| 0.4.13 | biais de routage MoE sur l'appareil des experts (experts en RAM hôte) | Lightning heretic EXL3 cohérent |
| 0.4.14 / 0.4.16 | `ssm_dt`, `ssm_a`, `ssm_d` des GGUF qwen3next masqués par les mappings Nemotron (régression 0.4.0) | conversion Coder-Next 80B en cours |
| 0.4.15 | gemma4 HF/EXL3 : normes **intactes** — `Gemma4RMSNorm` multiplie par w (pas 1 + w comme Gemma 3), et le convertisseur llama.cpp gemma4 a `norm_shift = 0` ; le +1 de 0.4.11 doublait les normes | gemma-4-12B abliterated EXL3, Artemis 31B : « Paris » |
| 0.4.16 | GGUF gemma4 issus d'un vieux convertisseur (normes +1, `attn_q_norm` ≈ 2) reconnus et ramenés à w | gemma-4-12B heretic GGUF : « Paris » |
| 0.4.17 | verrou de compilation JIT orphelin (`~/.cache/acvram/kernels/lock`, FileBaton de torch) retiré au chargement : un processus tué en pleine compilation figeait ensuite tout moteur en veille | diagnostiqué par `faulthandler` + SIGUSR1 |
| 0.4.18 | piles d'experts : repli sur la boucle par expert si la mémoire GPU manque | 80B |
| 0.4.19 | couches hybrides avec MLP/MoE en RAM hôte (activation transférée, porte partagée et biais de score chez les experts) | Qwen3-Coder-Next 80B-A3B (GGUF Q3_K_S) : « Paris », ~7 t/s en eager, experts en RAM |

Le paquet `acvram_0.4.16_amd64.deb` est construit (`sudo dpkg -i` à faire) ;
à reconstruire en 0.4.19 (`tools/construire-deb.sh`).

Leçon de la campagne gemma4 : deux conversions du même modèle par deux
chemins (GGUF sain contre EXL3 faux) comparées tenseur par tenseur ont
désigné la cause en une mesure (`q_norm` 1,02 contre 2,03, cosinus 1) là où
les hypothèses (GQA 16:1, cache des couches globales, RoPE proportionnel,
`layer_scalar`) avaient toutes été vérifiées sans rien trouver.

Pistes restantes :

* **5 — précision KDA** : `chunk_kda` reçoit déjà des entrées fp32 ; mesuré
  contre une récurrence fp64 sur 1 024 jetons : 1,6e-3 relatif (7,5e-3 en
  bf16), 0,50 ms contre 0,60. Le reste vient des `tl.dot` bf16 internes de
  fla : pas de gain sans patcher fla. Rien à changer.
* **6 — prefill Qwen3-Coder** : fait (0.4.12). Les 18 672 `aten::mm` (242 ms)
  sont la décomposition interne de `torch._grouped_mm` sur cette version de
  torch (384 par couche = 128 experts × 3 projections ; aucune trame acvram
  dans leurs piles d'appel) : ~29 TFLOP en 242 ms, soit 60 % du pic bf16 —
  pas de gain bon marché. Prochain palier : une GEMM groupée CUTLASS ou
  directement en NVFP4 (sans déquantification) pour les experts.
* **7 — .deb** : `acvram_0.4.16_amd64.deb` construit.

Pièges de la campagne : un listing de dossiers tronqué à 48 caractères donne
des chemins faux ; un chat lancé pendant qu'on patche importe l'ancien
`model.py` (signature `static_bind`) — retester après tout patch ; deux
conversions concurrentes vers le même dossier mêlent leurs manifestes.

## 2 septembre 2026, après-midi — tout le parc converti (0.4.20 → 0.4.24)

| version | contenu | validation |
|---|---|---|
| 0.4.20 | Gemma 4 26B-A4B : MoE en parallèle du MLP dense (`MoEBlockGemma` : routeur sur x normalisé × échelle × h^-½, softmax, top-k sans renormalisation, échelle par expert ; `ffn_gate_up_exps` scindé ; normes `post_ffw_1/2`, `pre_ffw_2`) | gemma-4-26B-A4B ultra et APEX : « Paris » |
| 0.4.21 | `hfquant` : compagnons (scales, qzeros, weight_scale…) lus dans le bon fragment ; préfixe `model.language_model.` retiré pour tout HF, tour visuelle ignorée | Qwen3-30B AWQ, Qwen3-VL-30B AWQ |
| 0.4.22 | DeepSeek-V2/V3 depuis HF : `kv_b_proj` scindé en k_b/v_b (convention llama.cpp), experts partagés, `scoring_func` | — |
| 0.4.23 | RoPE **YaRN** (rampe beta_fast/beta_slow, échelle d'attention × mscale²), top-k non renormalisé quand `norm_topk_prob` est faux | DeepSeek-Coder-V2-Lite : « Paris » |
| 0.4.24 | placement réajusté au chargement d'après les tailles réelles du manifeste (les promotions int8 ajoutaient ~10 Gio à un 70B) avec 2 Gio de marge | Hermes-4-70B, DeepSeek-R1-Llama-70B, Llama-3.3-70B : « Paris » (MLP partiellement en RAM, ~7 t/s) |

Campagne de conversion : 58 sources restantes (GGUF, EXL3, AWQ, vLLM, HF)
→ **58 converties et validées en chat** après reprises (mmproj pris pour le
modèle, MoE Gemma 4, fragments AWQ, YaRN, 70B). Parc acvram : 110
conversions, 107 alias dans les menus ; `.deb` 0.4.24.

Déplacements vers le 980 PRO (`/media/anticitoyenlm/2TO_2023_980PRO1/Modeles`,
liens symboliques aux anciens emplacements) : 8 familles non-acvram, puis 51
sources converties ; 4To : 1,4 To libres, 980 PRO : 25 Go libres.

Incident : le déplacement par modèle a traversé un lien de famille
(`models/` déjà déplacée) — rsync sur lui-même puis `rm -rf` : la source HF de
DeepSeek-Coder-V2-Lite a été détruite, retéléchargée (30 Go) et reconvertie.
Garde ajoutée : jamais de déplacement si la source est un lien ou déjà sous
la destination.

## 3 septembre 2026 — comparatif des quatre moteurs, sur le même parc

Banc identique pour tous : un prompt unique, 200 jetons, deux mesures dont la
meilleure est retenue, serveur démarré et arrêté entre chaque couple
(`scratchpad/banc-4moteurs.py`). 104 couples mesurés sur les **48 modèles
servis par plusieurs moteurs**.

| moteur | modèles gagnés | médiane | terrain |
|---|---|---|---|
| llama.cpp | 33 | 147 t/s | MoE (Coder-30B 234, LFM2.5 543, Gemma-4-26B-A4B 161) et denses 27-31B (42-54) |
| acvram | 10 | 41 t/s | denses EXL3 6 bpw (Cydonia 41, Muse-Glimmer 31, Skyfall 27), petits modèles (Coder-3B 94) |
| vLLM | 3 | 197 t/s | AWQ MoE (Thinking 204, VL-30B 192, erotic 183) |
| TabbyAPI | 2 | 19 t/s | Qwen3.5-4B 137, Coder-30B EXL3 90 |

Ce que le banc apprend sur acvram, chiffres à l'appui :

* **le MoE est notre faiblesse** : Coder-30B 88 contre 234, Gemma-4-26B-A4B 18
  contre 161 (routage Gemma en boucle par expert), Nemotron-Lightning EXL3 3,1
  contre 195 (le plan met les experts en RAM hôte alors que le GGUF Q4 tient) ;
* **les denses EXL3 sont notre force** : deux à trois fois TabbyAPI, dont le
  décodage trellis est borné calcul ;
* **l'AWQ MoE appartient à vLLM** (batch continu + noyaux Marlin).

Décisions prises : les menus ne gardent qu'un moteur par modèle (celui qui
gagne), 53 alias retirés, 898 Gio de fichiers devenus inutiles déplacés dans
`<disque>/Modeles/a_supprimer/`. Sauvegardes `~/.kimi-code/*.avant-tri-*`.

Deux correctifs sortis du banc : `llamacpp-serveur` ne suivait pas les liens
symboliques (`find` sans `-L`), et un modèle qui remplit la carte faisait
échouer la capture de graphe CUDA au lieu de basculer en eager (v0.4.25).

## 3 septembre 2026 — le MoE rattrapé : v0.4.26 à v0.4.28

Les trois faiblesses relevées par le comparatif se sont révélées être trois
bogues distincts, pas une limite d'architecture.

### Routage Gemma sur le chemin groupé (v0.4.26)

`MoEBlockGemma` forçait `_stack_state = "non"` : le noyau gate-up fusionné
codait SiLU en dur, or les experts de Gemma 4 sont en GELU-tanh. Toute la
couche retombait donc sur la boucle par expert. L'activation devient un
argument du noyau (`acv_act(v, act)`, 0 = SiLU, 1 = GELU-tanh) et un attribut
du bloc, déduit de `experts[0].act` ; les trois chemins (noyau fusionné, GEMV
groupée, prefill) la partagent.

### GEMM groupée NVFP4 au prefill (v0.4.27)

Le prefill matérialisait la pile d'experts en bf16 (`_pile_bf16`) avant
`torch._grouped_mm` : trois passes de plusieurs gigaoctets par couche, la
déquantification dominant le calcul utile d'un facteur six. `nvfp4_gemm_grouped`
garde les poids en 4 bits : chaque bloc déquantifie une tuile 64×64 en mémoire
partagée et la consomme aussitôt en tensor cores (`wmma` bf16 16×16×16). Les
jetons arrivent triés par expert ; l'hôte transmet, par tuile de 16 jetons,
(expert, premier jeton, compte). Repli conservé (`ACVRAM_PREFILL_DEQUANT`).

Banc synthétique 128 experts 768×2048, 1024 jetons top-8 : 1,56 ms → **1,17 ms**
(×1,33), pic de 3,6 Gio supprimé, cosinus 1,000000 contre la référence.

Le noyau relit cependant les poids d'un expert une fois par tuile de 16 jetons,
là où la déquantification les écrit une seule fois quel que soit le lot : il ne
gagne que tant que les experts reçoivent peu de jetons. Prefill mesuré sur
Qwen3-Coder-30B (jetons par expert entre parenthèses) :

| prompt | déquantification | GEMM groupée |
|---|---|---|
| 512 j (32) | 2 384 j/s | **3 159 j/s** |
| 1024 j (64) | 3 900 j/s | **4 093 j/s** |
| 2048 j (128) | **6 180 j/s** | 4 618 j/s |

D'où une bascule sur le nombre moyen de jetons par expert, seuil 64
(`ACVRAM_MOE_GEMM_MAX`). Prefill de bout en bout : 512 j 3 393 j/s,
1024 j 4 114 j/s, 4096 j **8 200 j/s** (référence du 2 septembre : 6 420).

### Le plan comptait des MLP fantômes (v0.4.28)

`_octets_reels` ne trouve aucun tenseur `.mlp.` pour un bloc Mamba2/GDN pur.
Le réajustement gardait alors la taille **nominale** de la couche : 29,6 Gio
imaginaires sur Nemotron-Lightning, d'où 22 MLP réels exilés en RAM hôte alors
que le modèle entier tient sur la carte. Une couche décrite par le manifeste
sans tenseur de MLP en compte désormais zéro. `_reajuster_plan` sait de plus
**remonter** en VRAM (le plan est figé à la conversion et ne savait que
descendre) ; échappement par `ACVRAM_PLAN_FIGE`.

Nemotron-Lightning : 22 MLP en RAM hôte → 0, poids réels 27,6 → 17,2 Gio pour
30,1 de carte. Non-régression vérifiée : DeepSeek-R1-70B descend toujours
35 MLP en RAM hôte.

### Résultat (banc direct, 128 jetons, RTX 5090)

| modèle | avant | après | meilleur rival |
|---|---|---|---|
| Nemotron-Lightning-30B EXL3 | 3,1 | **158** | llama.cpp 195 |
| Gemma-4-26B-A4B | 18 | **87** | llama.cpp 161 |
| Qwen3-Coder-30B-A3B | 88 | **99** | llama.cpp 234 |
| LFM2.5-8B-A1B | 208 | **244** | llama.cpp 543 |
| Cydonia-24B EXL3 | 41 | **47** | TabbyAPI 16 |

Le tri des menus du 2 septembre a été **annulé sur demande** : les 53 alias
retirés sont rétablis (270 alias, dont 3 conversions acvram jusque-là hors
menus), les dossiers sortis de `a_supprimer` sont revenus dans leur famille
avec leurs liens. Les correctifs postérieurs au tri (alias à points illisibles
en TOML) ont été réappliqués après restauration.

Reste à combler contre llama.cpp : Coder-30B (99 contre 234) et LFM2.5
(244 contre 543) — le décodage MoE y est encore borné par la GEMV groupée à
une paire (jeton, expert) par tranche de grille.

## 3 septembre 2026, après-midi — le décodage : v0.4.30 à v0.4.35

Le profil d'un pas de décodage (Qwen3-Coder-30B, 48 couches, un jeton)
montrait 10,5 ms dont un tiers seulement dans les produits matriciels. Le
reste partait en petits noyaux : environ vingt-cinq lancements par couche,
chacun payant quelques microsecondes de rampe pour quelques kilooctets. Sur ce
terrain, supprimer un lancement vaut autant qu'accélérer une GEMV.

### Conflits de banques en mémoire partagée (v0.4.30)

Le produit ligne-activation fait lire à la voie « l » les 32 flottants
`[l*32, l*32+32)` : rangés à plat, ils tombent tous dans la même banque et le
warp sérialisait en 32 accès. Un flottant de bourrage tous les 32 décale
chaque voie d'une banque. Noyau MoE gate-up mesuré seul : **516 → 832 Go/s**.

La même version empile q, k et v en une GEMV (deux des trois étaient minuscules
à cause des têtes KV groupées) et corrige un défaut plus ancien : `gate_up` et
`qkv_proj` étaient déclarés en attributs de **classe**, ce qui masque le module
enregistré par `nn.Module.__setattr__` — `self.gate_up` restait `None` et la
fusion gate/up n'avait jamais servi depuis son introduction.

### RoPE et normes de tête fusionnées (v0.4.31, v0.4.32)

Le chemin PyTorch coûtait, par couche, deux `index_select` sur les tables
cos/sin, deux tranches, deux négations, deux concaténations et quatre
produits. `rope_inplace` tourne q et k en place en un lancement, reçoit les
tables complètes plus les positions — l'indexation se fait dans le noyau — et
applique au passage les normes RMS par tête : une tête tient dans un bloc, sa
somme des carrés ne coûte qu'une réduction. Le noyau suit aussi le pas de q et
k, tranches de la projection empilée, ce qui évite deux copies.

### L'écriture du cache KV (v0.4.32)

Le gisement le plus lourd, invisible dans le profil par noyau parce qu'il était
éparpillé : l'écriture du cache quantifié enchaînait, **par couche et pour
chacun de k et v**, un abs, un amax, deux conversions, une division, un round,
un clamp, une conversion de sortie puis la dispersion. `kv_write_int8` fait
tout en un lancement, un bloc par (jeton, tête). Échelles identiques à la
référence, quantification identique à une unité près sur 3 valeurs sur 8704.

**121 → 157 t/s** sur Qwen3-Coder-30B.

### N activations par lecture de poids (v0.4.33)

Les noyaux GEMV bouclaient sur les lignes d'activation à l'extérieur et
relisaient toute la matrice pour chacune. Un pas de vérification spéculative à
cinq jetons coûtait donc cinq fois le trafic d'un pas simple, un lot de huit
requêtes huit fois — la spéculation et les lots ne pouvaient pas payer. Le
poids est désormais lu une fois et sert aux N activations, gardées en
registres ; NV est instancié exactement de 1 à 8, au-delà on passe par
tranches.

| | avant | après |
|---|---|---|
| INT8 5120×2048, N=5 | 5,0× le coût de N=1 | **2,4×** |
| N=8 | 8,0× | **3,5×** |

Débit agrégé Qwen3-Coder-30B : b=1 163 t/s, b=2 249, b=4 392, **b=8 509**.

Le proposeur n-gram tient aussi son index au fil de l'eau (une entrée par jeton
et par longueur) au lieu de balayer 4096 jetons en Python à chaque pas, et
surveille son rendement : en dessous de 0,15 jeton gagné par pas il se met en
veille et réessaie périodiquement. Prose 136 → 145 t/s, code 111 → 126.

### Quatre lancements de moins par couche (v0.4.34, v0.4.35)

`moe_reduce` (pondération, somme des top_k et conversion en un) ; le poids du
routeur gardé dans le type de l'entrée ; `topw` en fp32 de bout en bout —
l'aller-retour bf16 coûtait deux copies ; l'attention paginée acceptant q en
bf16 et rendant du bf16 ; le résidu différé entre couches, absorbé par la
normalisation d'entrée de la suivante.

Régression attrapée par la série de non-régression : `topw` restant en fp32,
`index_add_` de la boucle de repli refusait une source d'un autre type que la
destination et Nemotron-Lightning plantait au premier jeton (v0.4.35).

### Ce qui a été essayé et rejeté sur mesure

* **Noyau gate-up « large »** (R lignes par warp, lectures groupées pour tenir
  plus d'octets en vol) : 831 → 594 Go/s à R=4, la pression de registres coûte
  plus que le gain de latence.
* **Routage MoE fusionné** (produit du routeur dans le noyau de top-k) :
  157 → 115 t/s — avec un seul jeton la grille tombe à **un bloc** pour lire un
  mégaoctet de poids, là où cuBLAS occupe toute la carte.
* **Projection MoE down fusionnée** (pondération et somme dans le noyau) :
  121 → 119 t/s, huit fois moins de blocs.
* **GEMV INT8 « un warp par ligne »** : 1797 → 1411 Go/s, le noyau historique
  est déjà au plafond sur ces formes.
* **Spéculation n-gram non adaptative** : coûte 5 à 16 % sur du texte peu
  répétitif, d'où la veille automatique.

Effet de bord instructif : 4 Ko de mémoire partagée ajoutés au noyau de
routage, **même inutilisés**, coûtaient 7 % de débit.

### Résultat (banc direct, 128 jetons, RTX 5090)

| modèle | 2 septembre | 3 septembre | meilleur rival |
|---|---|---|---|
| Nemotron-Lightning-30B EXL3 | 3,1 | **158** | llama.cpp 195 |
| Gemma-4-26B-A4B | 18 | **111** | llama.cpp 161 |
| Qwen3-Coder-30B-A3B | 88 | **166-172** | llama.cpp 234 |
| LFM2.5-8B-A1B | 208 | **270-297** | llama.cpp 543 |
| GLM-4.7-Flash | 33 | **86** | llama.cpp 167 |
| Cydonia-24B EXL3 | 41 | **52** | TabbyAPI 16 |
| Skyfall-31B EXL3 | 27 | **39** | TabbyAPI 17 |

Le pas de décodage est passé de 10,5 à 6,2 ms, et la part des produits
matriciels d'un tiers à plus de la moitié : ce qui reste à gagner est
désormais dans les GEMV elles-mêmes, ou dans la suppression des lancements
qui subsistent.

Le rebanc au protocole du comparatif (serveur démarré et arrêté par modèle) n'a
pu être mené qu'un modèle : **agents-a1-4b-kimi 43,7 → 50,5 t/s**. Il reste à
faire sur les 33 couples pour actualiser le tableau des quatre moteurs.

## 3 septembre 2026, soir — les deux derniers lancements : v0.4.36 et v0.4.37

Huit pistes avaient été listées après le profilage du décodage ; deux ont tenu
la mesure, six ont été écartées **sur mesure** — ce qui vaut d'être écrit, parce
que chacune paraissait évidente sur le papier.

### v0.4.36 — le RMSNorm tournait avec un bloc de 256 fils, quelle que soit la
largeur

`rmsnorm_bf16_kernel` réduisait sur 256 fils, du 8B au 70B. Sur un modèle à
`H = 5120`, chaque fil traitait 20 éléments et la réduction en arbre se payait
sur une seule chaîne de warps. Le bloc est désormais choisi selon la largeur :

```c
const int th = H >= 2048 ? 1024 : (H >= 1024 ? 512 : 256);
```

Qwen3-Coder-30B-A3B : **172 → 183,7 t/s** (banc direct, 128 jetons, 5090).
Commit `cd23f02`.

### v0.4.37 — l'attention paginée lançait deux noyaux pour une seule tranche

`paged_attn_partial_kernel` écrivait toujours des accumulateurs partiels, puis
`paged_attn_reduce_kernel` les normalisait — même quand le contexte tient dans
une seule tranche (`PA_CHUNK = 512`), c'est-à-dire dans l'immense majorité des
pas de décodage courants. Le noyau partiel écrit maintenant directement la
sortie normalisée quand `C == 1`, et le second lancement disparaît. Le brouillon
spéculatif est passé au même régime : choix glouton, sans softmax ni tenseur de
probabilités.

**183,7 → 184,9 t/s**, texte identique sur un contrôle de 700 jetons.
Commit `757f3e6`.

### Les six pistes écartées

| piste | attendu | mesuré |
|---|---|---|
| attention en NVFP4 (poids q/k/v/o) | moins d'octets lus | 184,9 → 173,5 : `nvfp4_gemv` plafonne à 767 Go/s contre 1780 pour `int8_gemv` |
| produit `__half2` dans le noyau NVFP4 | deux e2m1 par instruction | 832 Go/s, inchangé — le noyau est lié à la lecture, pas au calcul |
| noyau gate-up « large » (une tuile par bloc) | une passe de poids | 831 → 594 Go/s, pression de registres |
| routage MoE fusionné en un bloc | un lancement de moins | 157 → 115 t/s : un seul bloc pour 1 Mo de logits |
| MoE down fusionné | 8× moins de blocs | 121 → 119 t/s |
| GEMV INT8 par warp | moins de synchronisations | 1797 → 1411 Go/s |

La leçon des deux journées tient en une ligne : **au décodage, ce qui coûte
n'est presque jamais l'arithmétique** — c'est le nombre de lancements, la
largeur des lectures, et la pression de registres. Une piste qui réduit les
FLOPs sans réduire les octets lus ne gagne rien.

### Ménage

Le modèle d'essai `Qwen3-Coder-30B-A3B-snr15` (17 Gio), converti pour mesurer
le plancher de SNR à 15 dB, a été supprimé : la conversion de référence tient
la qualité et le débit.

## Nuit du 3 au 4 septembre 2026 — huit pistes passées au banc : v0.4.38 à v0.4.43

Les pistes ouvertes par la bibliographie et le profilage ont toutes été
instruites. Trois ont donné un gain, trois se sont fermées sur mesure, deux
restent ouvertes mais hors de portée sans réentraînement. Le fait marquant :
**les trois gains sont des bogues, pas des optimisations** — du code qui ne
faisait pas ce qu'il annonçait.

### v0.4.38 — la 3080 Ti ne compilait aucun noyau

`_ensure_cuda_home` n'exigeait CUDA 12.8 que si une carte Blackwell était
visible. Avec `CUDA_VISIBLE_DEVICES=1`, il retenait le `nvcc` **système en
12.0**, qui n'a pas `cuda_fp4.h` — que notre source inclut inconditionnellement.
La compilation échouait, un avertissement passait inaperçu, et acvram tournait
sur ses noyaux de référence en Python. L'exigence est celle du source, pas de
l'architecture visée.

### v0.4.39 — le prefill court déquantifiait tout le modèle

Nsight Systems, sur un dense de 27B : `int8_dequant_kernel`, **11 % du temps
GPU, 127 instances**, 803 µs en moyenne. Le modèle porte exactement **128
tenseurs INT8** — un par tenseur, une fois par prefill. `int8_matmul` basculait
sur « déquantifier le tenseur entier puis `F.linear` » dès **huit** jetons.

Croisement mesuré sur un tenseur 5120×5120 par groupes de 128 :

| jetons | GEMV multi-N | déquantification + linear |
|---|---|---|
| 8 | 0,053 ms | 0,565 ms |
| 64 | 0,409 | 0,561 |
| **88** | *croisement* | |
| 256 | 1,632 | 0,617 |

Seuil porté à 80 (`ACVRAM_INT8_GEMV_MAX`). Temps jusqu'au premier jeton :

| invite | avant | après |
|---|---|---|
| 16 jetons | 286,8 ms | **193,9** (−32 %) |
| 32 | 285,1 | **211,3** (−26 %) |
| 64 | 285,5 | **245,2** (−14 %) |
| ≥ 128 | inchangé | inchangé |

Le plateau parfaitement plat à 285 ms pour toute invite de 16 à 128 jetons
était le coût fixe de la déquantification, indépendant du travail réel.

### v0.4.42 — hors Blackwell, le NVFP4 passait par l'émulation

Sur la 3080 Ti enfin capable de compiler, le micro-banc a montré l'anomalie
d'un coup : `int8_gemv` à **777 Go/s** (le plafond de la carte est à 770),
`nvfp4_gemv` à **144**. Les intrinsèques `__nv_cvt_fp4x2_to_halfraw2` et
`__nv_cvt_fp8_to_halfraw` n'ont d'instruction matérielle qu'à partir de sm_100
et sm_89 ; en dessous, le toolkit part en émulation logicielle — appelée seize
fois par lecture de poids et par ligne.

Remplacées par un décodage en registres. Les huit magnitudes E2M1 (0, 0,5, 1,
1,5, 2, 3, 4, 6) sont toutes des multiples d'un demi : 0, 1, 2, 3, 4, 6, 8, 12
tiennent chacune sur un quartet, donc la table entière est la constante
`0xC8643210`, lue par décalage. Une table en mémoire constante aurait été
sérialisée huit fois par warp, l'index différant d'un fil à l'autre. L'e4m3 se
reconstruit par assemblage de bits, cas sous-normal et unique motif NaN de
l'E4M3FN compris.

| | avant | après |
|---|---|---|
| `nvfp4_gemv` (3080 Ti) | 144 Go/s | **446 Go/s** |
| Qwen3-4B | 57,2 t/s | **110,2** (+93 %) |
| Qwen2.5-Coder-3B | 75,0 t/s | **122,5** (+63 %) |
| Qwen3-4B sur 5090 | 132,2 t/s | 132,0 — inchangé |

Justesse vérifiée au bit près entre l'intrinsèque de la 5090 et l'arithmétique
de la 3080 Ti : mêmes somme et norme, motifs NaN inclus. Le correctif vaut pour
toute carte antérieure à Blackwell. Pour situer, le `llama-server` du port 8081
fait 168,6 t/s sur le même Qwen3-4B, mais en Q4_K_M (2,5 Gio) contre nos
3,7 Gio : à octets égaux, la parité est atteinte.

### v0.4.43 — le cache de préfixe ne publiait jamais rien

`_finish` vidait `seq.blocks` **avant** que `_register_complete_blocks` ne soit
appelé, quelques lignes plus loin. Une requête qui s'arrête au premier jeton ne
publiait donc rien du tout, et toute séquence perdait ses blocs non encore
publiés. Compteur à l'appui : `publiés = 0` après trois requêtes partageant une
amorce de 361 jetons. La publication a lieu désormais avant la restitution.

| requête (amorce commune de 361 jetons) | avant | après |
|---|---|---|
| 1 | 416,7 ms | 416,7 ms |
| 2 | 123,3 | 123,3 |
| 3 | 121,6 | **45,1** |

704 jetons d'invite servis par le cache au lieu de zéro. Sorties identiques
avec et sans cache, vérifié sur deux requêtes complètes. Sur les hybrides à
récurrence linéaire le cache reste coupé — les blocs KV n'y suffisent pas à
restaurer l'état GDN.

### v0.4.40 et v0.4.41 — la tête MTP, livrée mais pas rentable

Les couches `nextn` étaient jetées à la conversion (`continue` dans `gguf.py`,
filtre sur `mtp.` dans `convert.py`). Elles sont désormais conservées sous
`model.mtp.<n>`, quantifiées au format de la dernière couche, chargées en
`MTPHead` (enorm, hnorm, eh_proj, bloc de transformeur, shared_head_norm) avec
son propre cache, et exposées par `--speculative mtp` / `auto`.

Deux points établis par la mesure. En **forçage enseignant**, la tête prédit
correctement le jeton *suivant le suivant* dans **50 %** des cas : elle est
saine et correctement branchée — l'ordre `[plongement ; état caché]` donne ces
50 %, l'ordre inverse donne **zéro**. Et l'état à reprendre après un pas
spéculatif est celui du dernier jeton **accepté**, pas la dernière ligne du lot.

Mais dans la boucle réelle, l'acceptation plafonne à 13,5 % et le débit tombe
de 37,6 à 21,6 t/s. Deux causes :

- **le cache de la tête se pollue** : les positions écrites pendant le
  brouillonnage le sont avec les états produits par la tête elle-même ; une fois
  les jetons acceptés, ces écritures restent et l'attention lit un contexte qui
  n'est pas celui de la cible ;
- **le surcoût par jeton brouillon dépasse la tête** : chaque proposition
  construit un lot en Python, hors graphe CUDA, et traverse **`lm_head` en
  entier** — le plus gros produit matriciel du modèle, environ 1,4 couche à lui
  seul. La tête coûte donc près de 2,4 couches par jeton brouillon, pas une.

Rentabiliser le MTP demande un chemin à formes fixes avec graphe pour la tête,
et un `lm_head` restreint aux candidats plausibles. Livré, testé, **non activé
par défaut**.

### Trois pistes fermées sur mesure

**H-Scale — affiner les échelles NVFP4 : rien à prendre.** Le SNR de sortie ne
bouge pas de ±0,01 dB quand l'échelle globale varie de 0,6 à 1,5 fois sa valeur
nominale ; il reste collé à 20,45 dB. Les échelles par bloc de 16 sont exactes
et absorbent tout : le 20,45 dB est le **plancher intrinsèque de l'e2m1**, pas
un défaut de réglage. La rotation de Hadamard n'apporte pas davantage — gain
médian **−0,02 dB** sur seize tenseurs réels, ce qui confirme le choix déjà en
place de la réserver à l'INT4 par groupes de 128.

**GEMM groupée du prefill MoE : le seuil est déjà au bon endroit.** Trois voies
comparées, 32 experts, K = 2048, M = 768 :

| jetons/expert | noyau maison | bf16 + `_grouped_mm` | `_scaled_mm` par expert |
|---|---|---|---|
| 16 | **0,11 ms** | 0,31 | 10,54 |
| 64 | **0,29** | 0,36 | 10,81 |
| 128 | 0,57 | **0,40** | 10,35 |
| 512 | 2,06 | **0,64** | 10,31 |

Le croisement tombe entre 64 et 128 jetons par expert — exactement le seuil
`ACVRAM_MOE_GEMM_MAX` retenu le 3 septembre. La voie CUTLASS par expert
(`torch._scaled_mm` en boucle) est vingt fois plus lente : trente-deux
lancements et autant de quantifications d'activation.

**Parcimonie d'activation : réelle, mais inexploitable telle quelle.** Sur un
dense de 27B, la part des canaux intermédiaires du MLP sous un seuil du maximum :

| seuil | canaux concernés |
|---|---|
| 0,1 % | 14,4 % |
| 1 % | **56,5 %** |
| 5 % | 91,6 % |

Et la qualité tient : à 1 %, la sortie reste cohérente et fidèle ; à 5 %, elle
dégénère en répétitions. Il y aurait donc 56 % de la lecture de `down_proj` à
économiser. Mais la parcimonie est **purement contextuelle** : la part des
canaux faibles pour au moins 90 % des jetons est de **0,98 %** en moyenne, et
de **0 %** en médiane. Aucun élagage statique, aucun réordonnancement ne
groupera ces canaux — et sans regroupement, sauter des canaux isolés détruit la
coalescence des lectures. C'est exactement le problème que SharQ et DejaVu
résolvent par un prédicteur appris ; hors de portée sans réentraînement.

### Ce que la campagne apprend

Le profilage a d'abord démenti l'hypothèse de départ. Sur la 5090,
`nvfp4_gemv` atteint **1206 Go/s** sur une forme réelle, quand la lecture pure
d'un tenseur par `torch.sum` plafonne à **1071** : les noyaux ne sont pas le
problème. Vérifié au passage que ce plafond ne tient pas au bridage — à 600 W
la lecture pure donne 1072 Go/s, à l'identique.

Les trois gains de la nuit viennent tous du même endroit : **du code qui
n'exécutait pas le chemin qu'il annonçait**. Un seuil de bascule laissé à sa
valeur de mise au point, une compilation qui échouait en silence, une
publication faite après une libération. Aucun n'aurait été trouvé sans mesurer
ce que la machine fait réellement, plutôt que ce que le code dit qu'elle fait.

## 4 septembre 2026 — la seconde salve : v0.4.44 et v0.4.45

### v0.4.44 — le cache de préfixe atteint enfin les hybrides

Le cache était coupé net dès qu'un modèle portait des couches à récurrence
linéaire — Qwen3.5, 3.6 et 3.8, Nemotron-Lightning, Kimi-Linear, LFM2, soit une
grande part du parc. La raison était juste : les blocs KV ne suffisent pas à
reprendre une invite, puisque l'état récurrent vit hors du cache paginé.

Cet état est désormais **photographié** aux frontières régulières du prefill,
en RAM hôte épinglée. Sur un 27B, ce sont 48 couches récurrentes et 149,6 Mio
par instantané : un tuple par couche, la fenêtre de convolution (10240 × 3) et
la matrice delta (48 × 128 × 128), en fp32. Trois instantanés sont conservés,
en éviction par ancienneté (`ACVRAM_INSTA_PAS`, `ACVRAM_INSTA_MAX`).

Le prefill se coupe une fois, à la plus grande frontière multiple du pas
(256 jetons par défaut) strictement intérieure à l'invite. Ce choix ne dépend
que de la longueur de l'invite : deux requêtes partageant une amorce tombent sur
la même frontière tant qu'elles restent dans la même tranche. À l'admission,
l'appariement des blocs KV est **plafonné** à la frontière dont on tient
l'instantané, et un appariement partiel est rejeté en bloc — état et clés
doivent coïncider exactement, sinon on repart de l'invite entière.

| amorce partagée | sans instantané | avec |
|---|---|---|
| 601 jetons | 381,5 ms | **300 ms** (−21 %) |
| 1 681 jetons | 788,2 ms | **309 ms** (−61 %) |

L'épinglage de la mémoire hôte compte : sans lui, la prise d'instantané faisait
passer la première requête de 3 152 à 7 457 ms ; avec, elle coûte 6 %. Le
découpage du prefill en deux passes change l'ordre des calculs récurrents, donc
la sortie diverge légèrement après quelques dizaines de jetons — au même titre
qu'un changement de taille de lot, et sans perte de cohérence vérifiée sur deux
requêtes complètes.

### v0.4.45 — le repli silencieux ne l'est plus

Le correctif du 3 septembre sur la 3080 Ti avait révélé le vrai danger : quand
la compilation des noyaux échoue, acvram bascule sur ses implémentations de
référence, dix fois plus lentes, et ne le signale que par un `warnings.warn`
noyé dans la sortie de chargement. Le chargement d'un modèle vérifie désormais
`build_info()` et écrit un avertissement franc sur la sortie d'erreur, avec la
cause et le renvoi à `python -m acvram doctor`.

Dans le même commit, le sampler glouton cesse de matérialiser tout le
vocabulaire : `log p(choisi)` se calcule en `logit − logsumexp`, une réduction
au lieu d'un `log_softmax` complet de 152 000 entrées suivi d'un `gather` d'une
seule case ; et le `clone()` des logits, qui n'existe que pour les pénalités
écrivant en place, n'a plus lieu quand aucune pénalité n'est demandée. Gain non
mesurable sur le débit (176,4 contre 176,6 t/s) — le code est simplement plus
juste.

### Deux pistes de plus fermées sur mesure

**Précision mixte intra-tenseur : le bruit est réparti, pas concentré.** Si
l'erreur de quantification NVFP4 tenait dans quelques canaux de sortie, il
suffirait de promouvoir ces lignes-là en INT8 au lieu du tenseur entier — les
128 tenseurs promus d'un 27B coûtent 22 % du temps de décodage pour 15 % des
tenseurs. Mesuré sur huit tenseurs réels, la part des lignes portant la moitié
du bruit va de **34,7 % à 48,2 %**, et il en faut 68 à 79 % pour en porter 80 %.
La distribution est quasi uniforme — ce qui est cohérent avec le plancher
intrinsèque de l'e2m1 constaté la veille. Promouvoir la moitié des lignes pour
enlever la moitié du bruit ne vaut pas mieux que promouvoir le tenseur.

**Le seuil de bascule NVFP4, lui, est au bon endroit.** Le jumeau du bogue INT8
corrigé la veille a été balayé sur un dense de 27B : TTFT d'une invite de 16
jetons à **194,7 ms** avec le seuil à 8, 202,8 à 32, 265,1 à 64, 283,1 à 128.
Le chemin W4A8 ne matérialise pas le poids entier à chaque appel, contrairement
à la déquantification INT8 ; monter le seuil ne fait que perdre. Rendu réglable
(`ACVRAM_NVFP4_GEMV_MAX`) et commenté, valeur inchangée.

### Nsight Compute reste hors d'atteinte

`ncu` refuse les compteurs matériels (`ERR_NVGPUCTRPERM`) : le pilote les
réserve à l'administrateur. Le déblocage est un paramètre de module et donc un
redémarrage — la marche à suivre est dans `MATERIEL.md`. En attendant, `nsys`
suffit à compter les noyaux et à voir où va le temps, mais pas à savoir ce qui
plafonne un noyau donné.

## 4 septembre 2026 — le parc rebancé, et l'énergie enfin mesurée

Trente-huit modèles, protocole du banc direct (sept tours, meilleur retenu,
128 jetons), cartes à leurs limites habituelles — 5090 à 400 W. Pour la
première fois la puissance réellement tirée est échantillonnée pendant la
mesure, ce qui donne la colonne qui manquait à tous les tableaux précédents.
Résultats bruts dans `rebanc-04sept.tsv`. Aucun échec, aucune régression.

### Ce que les correctifs ont rapporté

| modèle | 3 sept | 4 sept | |
|---|---|---|---|
| ornith-1.5-35B-A3B | 101,5 | **140,8** | +39 % |
| huihui-qwen3.6-35B-A3B abliterated | 63,4 | **80,2** | +26 % |
| qwen3-30B-A3B-thinking AWQ | 146,6 | **177,1** | +21 % |
| kat-coder-v2.5-dev | 122,1 | **140,6** | +15 % |
| qwen3.6-35B-A3B uncensored | 117,3 | **131,5** | +12 % |
| les vingt denses 27-32B | ~36,7 | ~37,5 | +2 % |

Les gains à deux chiffres sont tous des MoE, et ils viennent du seuil INT8 :
leurs tenseurs promus étaient déquantifiés en entier à chaque passe.

### Trois régimes, et un écart d'énergie de neuf pour un

| famille | t/s | W tirés | jetons/kJ |
|---|---|---|---|
| MoE 30B (AWQ, GGUF, EXL3) | 170-177 | 220-242 | 700-800 |
| MoE 35B | 131-141 | 193-202 | 680-712 |
| MoE 26B Gemma | 123-124 | 237 | 519-523 |
| denses 9B | 107 | 285-288 | 373-378 |
| MoE 42-47B GLM | 61-85 | 182-183 | 334-465 |
| **denses 27-32B** | **33-39** | **311-346** | **97-124** |

Le fait le plus net de ce tableau n'est pas le débit mais l'énergie. Un MoE de
30 milliards de paramètres rend **800 jetons par kilojoule** ; un dense de
27 milliards en rend **120**. Le rapport est de sept, et il monte à **neuf**
entre le meilleur (huihui-qwen3.6-35B à 880 jetons/kJ, pour 91 W seulement) et
le pire (gemma-4-31B et awaxis-31B à 97, pour 341 W).

Deux causes se cumulent : un MoE lit une fraction de ses poids par jeton, donc
il va quatre fois plus vite ; et il sollicite moins la mémoire, donc il tire
150 W de moins. Le débit et la consommation vont dans le même sens, ce qui
double l'écart.

Vingt denses 27B mesurés entre **37,3 et 38,9 t/s**, pour 311 à 325 W et 119 à
124 jetons/kJ — d'origines, de quantifications et de formats différents
(Q4_K_M, Q5_K_M, Q6_K, EXL3 5 bpw). À ce point de régularité, ce n'est plus le
modèle qu'on mesure mais la bande passante GDDR7 : la seule façon de déplacer
ce plateau est de lire moins d'octets.

## 4 septembre 2026, soir — v0.4.46 : le décodage FP4 se faisait en logiciel

Nsight Compute, débloqué le jour même par le paramètre de module, a renvoyé du
`nvfp4_gemv_kernel` un verdict que `nsys` ne pouvait pas donner : **le noyau est
limité par le calcul, pas par la mémoire** — 64,9 % de débit SM contre 47,1 %
de DRAM, et un pipeline **ALU saturé à 42,3 %**, soit le double du FMA. Un GEMV
qui passe son temps dans l'unité entière n'a rien à faire de la bande passante.

Le désassemblage a nommé le coupable. Dans la boucle interne, pour 4 096
instructions : **835 LOP3, 392 SHF, 193 PRMT, 128 SEL**. Aucune conversion
matérielle. Or `e2m1_pair` appelle bien `__nv_cvt_fp4x2_to_halfraw2`, l'
intrinsèque prévu pour cela, sous la garde `__CUDA_ARCH__ >= 1000`.

La garde était insuffisante. `cuda_fp8.h` n'émet l'instruction
`cvt.rn.f16x2.e2m1x2` que si `__CUDA_ARCH_FAMILY_SPECIFIC__` est défini — ce que
nvcc ne fait **que** pour les cibles à suffixe, `sm_120f` ou `sm_120a`. Compilé
en `sm_120` générique, comme nous le faisions, l'intrinsèque retombe
silencieusement sur une émulation : le quartet est promu en E2M3, puis converti
en half par arithmétique entière. Vingt-cinq instructions par paire de poids,
appliquées à **chaque poids de chaque tenseur NVFP4 à chaque jeton**.

`_arch_flags` demande désormais la forme *family-specific* pour toute capacité
supérieure ou égale à 10.0. Le repli PTX reste générique — une famille ne se
compile pas en PTX portable. Échappement par `ACVRAM_ARCH_FAMILY=0`.

| au désassemblage | sm_120 | sm_120f |
|---|---|---|
| LOP3 | 835 | **0** |
| PRMT | 193 | **0** |
| SEL | 128 | **0** |
| SHF | 392 | 63 |

| micro-banc (5090, tenseur en L2) | sm_120 | sm_120f | |
|---|---|---|---|
| `nvfp4_gemv` 17408×5120 | 42,55 µs | **21,43 µs** | ×1,99 |
| `int8_gemv` 5120×5120 | 13,17 µs | 11,81 µs | ×1,12 |

| banc direct, 128 jetons | sm_120 | sm_120f | |
|---|---|---|---|
| qwen36-27b-heretic-exl3 (dense) | 38,9 | **43,4** | +11,6 % |
| artemis-31b-exl3 (dense) | 36,7 | **39,7** | +8,2 % |
| gemma4-12b-heretic-gguf | 68,9 | **72,5** | +5,2 % |

Les sorties sont **identiques bit à bit** entre les deux compilations : la
conversion E2M1 vers half est exacte des deux côtés, seul le nombre
d'instructions change. Le plateau des denses, tenu pour une limite de bande
passante GDDR7 depuis le 3 septembre, était donc pour une part une limite
d'unité entière — la première fois qu'il bouge.

Le gain sur `int8_gemv`, qui ne décode aucun quartet, vient de `e4m3_to_float` :
les échelles de bloc passaient par le même mécanisme d'émulation.

## 4 septembre 2026, soir — v0.4.47 : la première réponse n'est plus la seule de son espèce

En lançant la suite complète après le passage à `sm_120f`, un test échouait :
`test_speculative_generation_under_graphs`, qui affirme qu'à température nulle
la spéculation rend la même sortie que le décodage ordinaire. Le premier réflexe
— accuser le changement d'architecture — était faux : le test échouait déjà, et
sa cause n'était pas la spéculation.

En inversant l'ordre des deux exécutions comparées, le motif est apparu : ce
n'est pas la spéculation qui diverge, c'est **la première génération d'un
processus** qui diffère de toutes les suivantes. Le test la mettait simplement en
premier. Le cache de préfixe, soupçonné ensuite, n'y était pour rien non plus :
lui aussi n'était incriminé que par l'ordre des essais.

Un relevé couche par couche a placé la divergence dès `layers.0.self_attn.q_proj`
— entrée identique, sortie écartée de 0,12 — donc dans la multiplication
elle-même. Le journal des backends a donné le reste : le chemin
`fp4-tensorcores` servait **5** GEMM à la première passe, puis **aucune** à
toutes les suivantes.

`nvfp4_mm_tensorcore` entoure son `torch._scaled_mm` d'un `except Exception` qui
pose `_OK = False` — extinction **globale, définitive et muette** du chemin FP4.
Or l'exception venait d'une seule couche du modèle : 688 colonnes, soit 344
octets empaquetés, et `_scaled_mm` exige une dimension contractée multiple de
16 octets. Une forme que le chemin ne sait pas prendre éteignait donc les tensor
cores pour *toutes* les autres, pour le reste de la vie du processus.

Deux conséquences, l'une de justesse et l'autre de vitesse :

* la première requête d'un serveur répondait par un autre chemin numérique que
  les suivantes — à température nulle, un jeton différent ;
* passé cette première requête, **vingt GEMM de prefill par passe** sur les
  vingt-neuf du modèle retombaient sur les noyaux fusionnés.

La forme est désormais écartée **avant** l'appel, là où elle doit l'être :
`padded_in % 32` ou `qweight.shape[-1] % 16` rendent `None`, poliment, et le
backend suivant prend le relais. L'extinction globale ne concerne plus qu'une
panne réelle du chemin, et elle s'annonce par un avertissement — le repli muet
avait déjà été corrigé pour la compilation des noyaux en v0.4.45, il restait ici.

| par passe de prefill, modèle témoin | avant | après |
|---|---|---|
| GEMM servies sur tensor cores FP4, 1re passe | 5 | **20** |
| GEMM servies sur tensor cores FP4, passes suivantes | 0 | **20** |
| formes refusées | — | 688 seulement |

`tests/test_improvements.py::test_first_generation_matches_the_next_ones` fixe
l'invariant : trois générations successives, sorties identiques, chemin FP4
toujours debout à la fin. Le test échoue sur le code d'avant.

## 4 septembre 2026, fin de journée — le parc rebancé après sm_120f et le repli FP4

Trente-huit modèles, sept tours, meilleur retenu, 128 jetons, 5090 à 400 W,
puissance échantillonnée à 50 ms pendant les tours. Résultats bruts dans
`rebanc-04sept-soir.tsv`, à comparer à `rebanc-04sept.tsv`.

**Gain sur les trente-huit modèles, médian +7,2 %, jusqu'à +15,9 %.**

| famille | matin | soir | | W tirés | jetons/kJ |
|---|---|---|---|---|---|
| MoE 30B (AWQ, GGUF, EXL3) | 170-177 | **182-189** | +6,7 % | 242→177 | 704→**1032** |
| MoE 35B | 131-141 | **146-147** | +4,4 % | 200→155 | 700→**940** |
| MoE 26B Gemma | 123-124 | **131** | +5,9 % | 238→231 | 521→567 |
| denses 9B | 107 | **114** | +6,3 % | 288→210 | 375→**540** |
| MoE 42-47B GLM | 61-85 | **64-89** | +5,0 % | 183→182 | 334→351 |
| **denses 27B** | 37,3-38,9 | **41,4-45,1** | **+10 à +16 %** | 314→258 | 120→**165** |
| denses 31-32B | 33,5-34,9 | **36,0-37,9** | +7,5 à +8,6 % | 346→298 | 98→**123** |

Trois enseignements.

**Les denses gagnent le plus**, ce qui était attendu : ce sont eux qui décodent
le plus de poids NVFP4 par jeton, donc eux qui payaient le plus cher l'émulation
logicielle de la conversion E2M1. Le plateau des denses 27B passe de ~37,5 à
~41,5 t/s — quinze modèles d'origines, de quantifications et de formats
différents, toujours aussi serrés, mais quatre jetons par seconde plus haut.

**Le prefill des MoE est transfiguré.** TTFT de 176 à 36 ms sur
Qwen3-Coder-30B, de 256 à 47 sur agentworld-35B, de 118 à 39 sur Gemma-4-26B.
C'est le correctif v0.4.47 : passé la première requête, vingt GEMM de prefill
sur vingt-neuf retombaient sur les noyaux fusionnés.

Les denses 27B, eux, affichent un TTFT en hausse (166 → 235 ms) — mais c'est le
protocole du matin qui était en cause, pas le code. À protocole identique, sur
qwen3.8-27B-UD, le code du 4 septembre matin donne 37,4 t/s et **237 ms**, le
code d'aujourd'hui 41,5 t/s et **235 ms** : le débit monte de 11 %, le TTFT ne
bouge pas. Le chemin W4A4 des tensor cores FP4 est d'ailleurs trois fois
meilleur que le W4A8 sur ce prefill — 224,7 ms contre 686,0 en retirant le
backend `fp4-tensorcores` du registre. La garde de forme de v0.4.47 ne coûte
rien à personne.

**L'énergie baisse partout.** Médiane du parc **354 → 420 jetons par
kilojoule**. Un MoE de 30 milliards rend maintenant **1032 jetons/kJ** contre
704, en tirant 177 W au lieu de 242 ; le record du parc est à **1369**
(huihui-qwen3.6-35B, pour 61 W). Moins d'instructions entières par poids
décodé, c'est directement moins de watts.

### Ce que ce rebanc a appris sur la façon de mesurer

Une première passe donnait les deux Gemma-4-26B à 53 t/s au lieu de 124, avec
dans leur journal, et le leur seul, `graphes CUDA désactivés : mémoire
insuffisante pour la capture`. La cause n'était ni le code ni une carte
occupée : le banc appelait `load_model()` **sans** `max_model_len`, alors que le
serveur le passe (`acvram/cli.py`). Or c'est au chargement que se décide le
budget KV — sans contexte annoncé, il prend jusqu'au dernier octet de la carte :
979 blocs, 15 664 jetons, pour un moteur ouvert à 220. Il ne restait plus de
quoi capturer un graphe, et seuls les deux modèles dont le graphe est le plus
gourmand en pâtissaient. Contexte annoncé, Gemma remonte à 130,9 t/s.

Le correctif écrit dans la foulée — plafonner le budget KV à ce que le contexte
exige — a été **retiré** : mesuré à 129,9 contre 130,0, il n'apportait rien et
n'aurait ajouté qu'une limite arbitraire. Deux autres écarts isolés
(qwen3.5-35B à 126 au lieu de 147, qwen3.5-9B à 99 au lieu de 114) ont disparu
à la remesure : une charge concurrente passait pendant leur tour.

## 4 septembre 2026 — la 3080 Ti mise à l'épreuve, et v0.4.48

La seconde carte n'avait jamais servi acvram : jusqu'au 3 septembre elle ne
compilait aucun noyau (le nvcc système, en 12.0, ne fournit pas `cuda_fp4.h`,
corrigé en v0.4.38) et retombait sans le dire sur les implémentations de
référence. Le bogue corrigé, la comparaison honnête devenait possible.

### Le banc, cartes et modèle nommés

RTX 3080 Ti (GPU 1), **275 W**, 12,3 Gio dont 5,2 occupés en permanence par le
llama-server du port 8081 — laissé intact, il fait partie du décor. Modèle
`Qwen3-4B-Instruct-2507`, servi par les deux moteurs : en GGUF Q4_K_M pour
llama.cpp, converti en NVFP4 pour acvram. Sept requêtes identiques, meilleure
retenue, 128 jetons.

| moteur (3080 Ti, 275 W) | t/s | W tirés | jetons/kJ |
|---|---|---|---|
| llama.cpp, Vulkan, KV q4_0, flash-attn | **170,1** | 216 | 787 |
| acvram, CUDA, NVFP4, KV int8 | 121,7 | 257 | 474 |

acvram tient **72 %** du débit de llama.cpp sur cette carte, en tirant 19 % de
plus. L'écart est réel et il n'a rien d'infamant pour une carte Ampere : sm_86
n'a ni tensor cores FP4 ni l'instruction de conversion E2M1 — le gain de
v0.4.46 ne s'y applique pas, le décodage des quartets s'y fait en registres.
llama.cpp, lui, y est chez lui depuis des années et son cache KV en q4_0 lit
deux fois moins que notre int8.

Ce que la 3080 Ti apporte au poste est donc clair : elle n'est pas un second
moteur acvram, elle est une carte d'appoint où llama.cpp sert mieux. Le port
8081 garde sa raison d'être.

### Ce que le profil de la 3080 Ti a révélé — et qui vaut pour les deux cartes

`nsys` sur le décodage montre `nvfp4_gemv_kernel` à 55 %, `int8_gemv_kernel` à
14 %, et un troisième larron inattendu : **`gemv2T_kernel_val`, 10 % du temps,
130 instances — exactement une par passe — à 900 µs chacune.** C'est la
projection de sortie.

Qwen3-4B partage sa table de plongements entre l'entrée et la sortie. À
l'entrée, c'est un *gather* d'une ligne ; à la sortie, une projection qui relit
**la matrice entière à chaque jeton** : 152 000 × 2 560 en bf16, soit 742 Mio,
**24 % de tous les octets lus par jeton**, pour une seule couche. Le
convertisseur laisse ce tenseur en bf16 — il n'est pas dans une couche, et
l'option `--lm-head-format` ne l'atteint pas puisqu'il ne s'appelle pas
`lm_head.weight`.

La projection prend désormais une **copie quantifiée** de la table, la table
restant en bf16 pour le gather. 741 → 379 Mio relus par jeton.

| | 3080 Ti | 5090 |
|---|---|---|
| tête liée en bf16 | 115,4 t/s | 160,1 t/s |
| tête liée en **int8** | **121,7** (+5,5 %) | **170,2** (+6,3 %) |
| jetons/kJ | 450 → 475 | 722 → 779 |

**La qualité ne bouge pas** : perplexité 25,138 en bf16, 25,109 en int8. Dix
modèles du parc sur quarante partagent leurs plongements, et aucun n'a besoin
d'être reconverti — la quantification se fait au chargement.

Deux précautions. La copie s'ajoute à la table au lieu de la remplacer, donc le
chargement exige le double de sa taille en mémoire libre, sans quoi il garde le
bf16 et le dit. Et les quantifieurs passent par une copie float32 du tenseur
entier — 1,45 Gio d'un coup sur cette table, ce qui ne tenait pas à côté du
modèle sur la 3080 Ti : la quantification se fait par paquets de 8 192 lignes,
bit pour bit identique puisque les échelles sont par ligne. Réglage par
`ACVRAM_TETE_LIEE` (`int8` par défaut, `bf16` pour revenir en arrière).

## v0.4.49 — le prix des promotions, et pourquoi il ne suffit pas (4 septembre)

Le décodage dense est limité par la bande passante : 79 % du temps part dans les
deux GEMV, et le noyau lit déjà la mémoire à 74,6 % du pic. Les instructions ne
rendent plus rien — il faut lire moins d'octets. Le plus gros poste évitable est
la précision mixte : sur `Huihui-Qwen3.8-27B-abliterated-Q5_K`, le plancher de
25 dB promeut 130 tenseurs en int8, soit 2 821 Mio, 32 % des octets relus à
chaque jeton.

Trois planchers mesurés à code identique, 8 192 jetons de contexte, corpus
d'évaluation de 16 383 jetons :

| plancher | taille  | t/s  | jetons/kJ | perplexité |
|----------|---------|------|-----------|------------|
| 25 dB    | 18,50 Gio | 41,8 | 162 | **42,591** |
| 22 dB    | 18,50 Gio | 41,8 | 162 | idem 25 dB |
| 0 (aucun)| 16,02 Gio | **46,2** | **189** | 43,447 |

Le plancher à 22 dB donne exactement le même modèle que 25 : en NVFP4 le rapport
signal/bruit de sortie est **quasi constant, 20,1 à 20,7 dB sur les 505 tenseurs
quantifiables**. Le plancher n'est donc pas un réglage continu mais un
interrupteur : au-dessus de 21 dB il promeut tout ce que le quota autorise,
en dessous il ne promeut rien.

### Le prix, pas le mérite

Le gain d'une promotion est lui aussi constant — 21,4 à 24,2 dB pour tout le
monde. Ce qui varie, c'est le prix : 0,1 Mio pour une porte `linear_attn.alpha`,
39 Mio pour un `mlp.up_proj`, 559 Mio pour le `lm_head`. Le rendement en
décibels par mébioctet varie donc d'un facteur **4 921**. Or `max_promotions`
compte des *tenseurs*, pas des octets : le quota s'épuise dans l'ordre de
rencontre, et les projections MLP le consomment avant que les 96 portes
`alpha`/`beta` — 10 Mio à elles toutes — soient seulement vues.

D'où `--promotion-cout-max`, un prix plafond en mébioctets ajoutés
(`promotion_cout_max_mib`, 0 = sans plafond, comportement inchangé). Le prix se
chiffre avant de quantifier, par la largeur nominale des formats
(`BPW_NOMINAL`) : le mesurer exigerait de quantifier deux fois tout le modèle.

### Résultat négatif, et il compte

`--promotion-cout-max 1` produit exactement la variante espérée : 96 promotions,
uniquement des portes, **16,04 Gio** — la taille de « sans promotions » — et
**47,3 t/s**. Sa perplexité est de **43,747**, c'est-à-dire *au niveau de
l'absence de promotions* (43,447), pas à celui du plancher plein (42,591).

Les portes ne sont donc pas le siège de la perte. Les 2,0 % de perplexité que
paie « sans promotions » sont portés par le volume des poids, pas par un petit
sous-ensemble critique — et le SNR par tenseur, mesuré couche à couche, ne
prédit pas l'effet sur la sortie du modèle. Un critère de promotion utile devra
mesurer la sensibilité de la *sortie* à chaque tenseur, pas la fidélité du
tenseur à lui-même. C'est le chantier suivant, pas un réglage.

L'arbitrage, lui, reste entier et appartient à l'utilisateur : `--snr-floor 25`
pour la qualité, `--snr-floor 0` pour 13,4 % de mémoire et 10,6 % de débit en
plus. Le défaut ne change pas.

## v0.4.50 — le plancher de SNR passe à zéro par défaut (4 septembre)

Arbitrage tranché : `snr_floor` vaut **0**, plus aucune promotion par défaut.
Le décodage est limité par la bande passante, et 2,0 % de perplexité se paient
moins cher que 13,4 % de mémoire et 10,6 % de débit. `--snr-floor 25` rétablit
l'ancien comportement pour qui préfère l'inverse.

Les modèles déjà convertis gardent leurs promotions : le changement ne vaut que
pour les conversions à venir. Le parc de 38 modèles ne récupère les 10,6 % qu'en
étant reconverti.
