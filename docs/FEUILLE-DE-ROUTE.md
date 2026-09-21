# Ce qui est fait et ce qui ne l'est pas

Écrit sans complaisance, parce que le projet que celui-ci remplace annonçait une
extension de mémoire GPU fonctionnelle et livrait un module noyau qui se
contentait d'afficher les nombres passés en paramètres.

## 8 septembre 2026 — les joules à côté des secondes

Le planificateur optimisait des **secondes**. L'objectif posé est double :
vitesse et jetons par kilojoule. Un plan peut être plus rapide et plus cher en
énergie, et rien ne le disait.

Le cas est mesuré : à débit égal à 5 % près, le chemin processeur coûte 183
secondes de temps processeur pour 200 jetons contre 25 par le bus. Sept fois
plus, **invisible pour le compteur de la carte**, qui ne voit que ce qu'elle
consomme elle-même. `_estimate` comptait la même durée pour les deux.

`Plan` porte désormais `est_joules_par_jeton` et `est_jetons_par_kj`, calculés
de trois sources : les cartes pendant la durée du jeton, pondérées par la part
de leur limite de puissance réellement atteinte ; les cartes alimentées qui ne
portent rien ; et le temps processeur, qui est le terme que le modèle en
secondes ne voyait pas.

**Ils valent `None` tant que les constantes ne sont pas mesurées**, et les trois
sont à zéro. Rendre None quand on ne sait pas vaut mieux qu'un chiffre
crédible : celui-ci servirait à choisir un plan. C'est la troisième fois de la
journée qu'une structure est posée sans sa valeur, après le coût fixe par
transfert et le nombre de copies.

**Une réserve à lever avant de renseigner quoi que ce soit.** Les 183 secondes
sont `utime + stime` du processus, donc du temps processeur réellement
consommé — mais **une attente active y compte en plein alors qu'elle consomme
peu**. Si le chemin processeur attend le bus en tournant, une part de ces 183
secondes n'est pas de l'énergie. Comparer `utime` et `stime` séparément, ou
passer au wattmètre de prise, tranche la question. Tant qu'elle ne l'est pas,
on ne sait pas si le terme doit compter des secondes de calcul ou des octets
déplacés.

Le chiffre restera de toute façon une **borne basse** : mémoire vive,
ventilateurs et carte mère ne sont pas modélisés.

Ordre de grandeur avec des valeurs plausibles, pour montrer l'enjeu et non pour
le trancher : 10 J par jeton tout sur carte contre 118 avec un MLP sur
processeur, soit un rapport de onze à débit égal.

## 8 septembre 2026 — la trace de routage, avant tout cache d'experts

**Le chiffre qui manque à tout le monde.** Aucun taux de succès mesuré n'existe
pour une pile à 512 experts routés 10. `cached_expert_fraction` est un rapport
de capacité, c'est-à-dire le taux qu'on obtiendrait si le routage était
uniforme — or il ne l'est pas, et c'est toute la raison d'être d'un cache.

`acvram/memory/trace_routage.py` journalise, pour chaque jeton et chaque
couche, les experts choisis, **dans l'ordre d'émission**. L'ordre porte
l'information : savoir qu'un expert est demandé 8 % du temps ne dit pas s'il
l'est en rafale ou dispersé, et ces deux régimes ne donnent pas le même taux.

Activé par `ACVRAM_TRACE_ROUTAGE=/chemin`. Éteint, il coûte **22 ns par appel**,
un test de booléen : la fonction sort avant de toucher aux tenseurs. Allumé, il
synchronise — assumé, une trace d'ordre ne peut pas faire autrement, et elle
n'est jamais active en service.

`taux_de_succes()` rejoue la trace à travers un cache de capacité donnée, en
politique du moins récemment servi ou du moins fréquemment servi. **Les deux
sont fournies parce que le choix n'est pas évident** : un routage en rafale
favorise la première, des experts chauds stables la seconde. La trace
tranchera, pas nous.

**Un défaut trouvé en l'essayant**, et qui aurait faussé tout rejeu : la
première version repartait du compteur cumulé à chaque couche, si bien que la
couche 1 numérotait 3, 4, 5 les jetons que la couche 0 appelait 0, 1, 2. Les
mêmes jetons sous deux noms, et un rejeu qui aurait cru voir deux fois plus de
trafic qu'il n'en passe. La base est désormais figée au premier appel d'un
passage. Deux tests le vérifient, dont un sur la numérotation elle-même.

L'index de couche descend depuis `DecoderLayer`, seul endroit qui le connaisse,
plutôt que par les trois sites de construction du bloc. Il vaut -1 si personne
ne l'a posé : une trace pleine de -1 se voit, une trace qui invente des numéros
non.

## 8 septembre 2026 — ma lecture réfutée sur l'ampleur, et le compte dérivé

**Le test que j'avais proposé s'est retourné contre moi, et c'est son intérêt.**
J'avais écrit : diviser les copies par trois doit faire tomber la pente d'un
tiers, sinon mon terme est faux. Mesuré : 3,11 ms deviennent 2,88, là où ma
lecture prédisait 2,07 et où une domination des latences aurait donné 1,04.

De ces deux points, à volume inchangé :

| Part | Valeur | Fraction |
|---|---|---|
| Latences fixes | 0,345 ms | 11 % |
| Le reste | 2,765 ms | 89 % |

Le reste correspond à 6,5 Go/s sur un bus mesuré à 18,7. **L'essentiel du coût
n'est donc ni la latence par transfert ni le débit nominal**, et il reste à
nommer : débit effectif plus bas, ou synchronisation par expert que le nombre
de copies ne change pas. Le terme constant existe, il ne domine pas.

`transfer_fixed_us` reste à **zéro** malgré ces deux points : ils viennent d'un
seul modèle et d'une seule couche. En tirer une constante générale serait
refaire l'erreur qu'on venait de corriger deux fois.

**Le nombre de copies par expert était faux deux fois, il n'est plus écrit.**
D'abord des blocs de 4 Mo, soit cinq copies — facteur dix-huit. Puis trois
copies par expert — facteur trois. Le vrai compte est neuf : trois projections,
et trois tenseurs par poids NVFP4 copiés un par un.

Il est désormais **dérivé de la classe** par
`NVFP4Tensor.transferts_par_poids()`, qui compte ses champs de type tenseur.
Quand les trois tenseurs voyageront dans un seul tampon épinglé, le compte
suivra tout seul. Une constante écrite ailleurs serait restée fausse en
silence — c'est arrivé deux fois en une heure.

## 8 septembre 2026 — le nombre de transferts, corrigé par la mesure

La courbe du 8 septembre a tranché deux choses sur le terme constant ajouté en
0.4.75.

**Le terme existe.** Une couche exilée transfère 18,4 Mio en 3,2 ms, soit
5,7 Go/s effectifs sur un bus mesuré à 18,7 Go/s. Le transfert est borné par la
latence, pas par le débit. La pente est régulière — 3,2 ms par couche entre 18
et 27 couches exilées, 3,6 entre 27 et 36 — sans plateau. Cinq cases sur le même
état, références se recoupant à 0,2 %.

**Sa valeur reste inconnue.** La courbe ne sépare pas le coût fixe par transfert
du coût par octet : les deux croissent ensemble avec le nombre de couches. Il
faut pour cela chronométrer un transfert d'expert sur trois tailles écartées
d'un facteur quatre. `transfer_fixed_us` reste donc à zéro.

**Le nombre de transferts, lui, était faux d'un facteur six.** La première
version supposait des blocs de 4 Mo, soit cinq copies par couche. Le relevé en
montre **trente** : dix experts routés, trois tenseurs chacun, environ 0,6 Mio
par copie. Le compte est désormais dérivé de `n_experts_active` du modèle et non
supposé, avec le nombre de tenseurs par expert en constante nommée — vraie des
piles vues jusqu'ici, fausse le jour où l'une en aura quatre.

**Un chiffre qui déplace l'objectif énergétique**, relevé au passage et qui
n'entre pas encore dans le modèle : à débit égal à 5 % près, le chemin
processeur coûte 183 secondes de temps processeur pour 200 jetons contre 25 par
le bus. Sept fois plus, et invisible pour le compteur de la carte — qui ne voit
que ce qu'elle consomme elle-même.

## 8 septembre 2026 — trois défauts corrigés par lecture, aucun mesuré ici

Corrections écrites après lecture du code, sans exécution sur les cartes : une
courbe de mesure y tournait. Ce qui est mesuré et ce qui ne l'est pas est dit
pour chacune.

### L'extinction du chemin FP4 était globale, elle est par capacité de carte

`fp4_gemm.py`. Un échec d'appel réel posait `_OK = False` pour tout le
processus. Sur une machine à cartes inégales, un échec survenu sur celle qui n'a
pas les tensor cores FP4 privait aussi celle qui les a. Le repli restait correct,
mais le gain était perdu.

L'état est désormais rangé par **capacité** `(major, minor)` et non par index de
carte : renuméroter avec `CUDA_VISIBLE_DEVICES` attribuerait sinon l'état à la
mauvaise. `fp4_mm_available(device)` accepte la carte, et l'appelant la passe.

Une **garde préventive** est posée en tête de `nvfp4_mm_tensorcore` : une carte
sous sm_100 n'atteint plus l'appel réel. Mieux vaut ne pas essayer que d'essayer,
échouer, et réparer les dégâts de l'échec.

Ce qui reste **global à dessein** : les échecs indépendants de la carte — type
`float4_e2m1fn_x2` absent, `_scaled_mm` absent, désactivation par
l'environnement. Les rendre par carte multiplierait le coût de sonde sans rien
changer au verdict.

**Ce qui n'a pas été vérifié** : le comportement sur deux cartes de capacités
différentes en conditions réelles. La correction est écrite, pas éprouvée.

### La fraction d'experts en cache n'est plus majorée d'un facteur inventé

`tiering.py`. Le rapport capacité sur octets hôtes était multiplié par 1,3 au
titre d'un « biais de routage » plausible et **jamais mesuré**. Un facteur
inventé qui majore rend le plan optimiste : il annonce moins d'octets traversant
le lien qu'il n'en passera. Sans mesure, on prend la borne haute du coût.

**Ce que cette correction ne change pas, et il faut le dire** : le choix du plan.
`cached_expert_fraction` n'entre que dans `_estimate`, donc dans
`decode_tok_s`, donc dans le troisième critère de `_rang` — celui qui ne
départage que les plans **sans exil**, où cette fraction vaut zéro. Elle corrige
le débit annoncé, pas la décision.

### Le modèle de coût n'avait aucun terme constant

`_estimate` ne comptait que des octets. Or il est optimiste d'un facteur 2,4 à
3,4 sur soixante et un modèles, avec un **plancher d'environ 3 ms par jeton**
qu'aucun terme en octets ne peut produire, et une pente d'environ 2,5 ms par
doublement de taille — qui suit le nombre de blocs transférés, pas leur volume.

Un coût fixe par transfert d'expert est ajouté, réglable par
`PlannerOptions.transfer_fixed_us`.

**Sa valeur est zéro et n'est pas mesurée.** Le modèle reste donc celui d'avant,
faux mais connu. Une constante inventée le rendrait faux **et** crédible, ce qui
est pire. Pour la fixer : chronométrer un transfert d'expert sur trois tailles
écartées d'un facteur quatre ; l'ordonnée à l'origine est la valeur, la pente
doit retrouver la bande passante du lien. Si l'ordonnée est nulle, le plancher
vient d'ailleurs — réveil de carte ou synchronisation par couche — et ce terme
n'est pas le bon.

**Sur la mesure qui motive tout cela** : les soixante et un modèles ont été
mesurés une fois chacun. La dispersion entre répétitions n'est connue que sur un
seul, à environ 4 %. Un facteur constant sur des cartes très différentes est soit
une vraie loi, soit un artefact commun aux deux mesures.

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

Remesuré le 13 septembre, moteur chaud et cache de préfixe coupé : le
croisement est vers 50 jetons par expert (48 : +2 %, 64 : −11 %), défaut
abaissé à 48 (`revue/banc-prefill-moe-12-09.md`).

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

## v0.4.51 — les convertis sur le SSD, les originaux sur le HDD (4 septembre)

Le parc était rangé à l'envers : les originaux (GGUF, EXL3, AWQ, HF) occupaient
physiquement le SSD NVMe de 2 To, plein à 99 %, et le HDD CMR de 4 To ne les
voyait qu'à travers 79 liens symboliques, tandis que les 103 modèles convertis
— ceux qui se chargent à chaque lancement — dormaient sur le HDD.

Réorganisation, 3,1 Tio déplacés par `scratchpad/migrer.py` (copie dans un
`.X.partiel`, vérification des octets et du nombre de fichiers, échange, puis
suppression de la source ; files entrelacées selon l'espace libre) :

* `/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram` — les convertis,
  seuls occupants du SSD ; c'est aussi là que `acvram convert` écrit par défaut.
* `/mnt/4TO_SATACMR_2022/Modeles/<catégorie>` — tous les originaux, en vrais
  dossiers ; plus aucun lien vers le SSD. Toutes les configurations passaient
  déjà par ces chemins : rien n'a changé pour elles.
* `/mnt/4TO_SATACMR_2022/Modeles/models_acvram` — un lien vers le SSD, pour les
  scripts qui gardent l'ancien chemin.

Une conversion lit donc l'original sur le HDD (lecture séquentielle, ce que
sait faire un CMR) et écrit le converti sur le SSD ; un lancement ne touche
que le SSD.

## v0.4.52 — le parc reconverti sans promotions (nuit du 4 au 5 septembre)

Les 38 modèles des menus ont été reconvertis avec `snr_floor = 0`
(`outils/reconvertir-sans-promotion.py` : conversion dans un `.neuf` voisin,
vérification du manifeste, échange — l'ancien modèle reste servi jusqu'au
dernier instant). Sources lues sur le HDD, sorties écrites sur le SSD.

| famille | exemple | avant → après | gain |
|---|---|---|---|
| denses 27-32B | Qwen3.8-27B | 18,51 → 16,03 Gio | −13,4 % |
| denses 31-32B (VL, gemma) | Qwen3-VL-32B | 21,50 → 18,21 Gio | −15,3 % |
| denses 9B | Qwen3.5-9B | 7,31 → 6,06 Gio | −17,1 % |
| MoE 35B-A3B | Qwen3.5-35B-A3B | 19,72 → 18,91 Gio | −4,1 % |
| MoE 30B AWQ | Qwen3-30B-A3B | 16,98 → 16,46 Gio | −3,1 % |

Total : **684 → 625 Gio (−8,6 %)**. Les MoE gagnent peu : leurs experts
n'étaient déjà jamais promus. Ornith-1.5-35B-A3B **grossit** de 3,6 % : sa
source porte des couches MTP que l'ancien convertisseur jetait et que le
nouveau conserve (v0.4.40) — plus lourdes que les 0,83 Gio de promotions
retirées, mais elles rendent le brouillon spéculatif possible.

Un piège du même ordre a coûté une demi-nuit : la vérification exigeait le
même nombre de tenseurs qu'avant, et refusait donc tout modèle dont les
couches MTP étaient désormais conservées (866 tenseurs contre 851). Corrigé :
le nouveau doit contenir tous les anciens, sans promotion, et peut en avoir
davantage.

Le comparatif des quatre moteurs (`outils/banc-4moteurs.py`, 134 couples sur
67 modèles) est relancé sur ce parc.

## 5 septembre 2026 — onze modèles muets au comparatif : trois causes, une méthode (v0.4.53 à v0.4.55)

Le comparatif des quatre moteurs (`outils/banc-4moteurs.py`, 134 couples sur
67 modèles) a d'abord donné 11 échecs sur 67 côté acvram, tous avec le même
symptôme : réponse vide, puis `CUDA error: an illegal memory access`. Jamais
hors serveur (`banc1.py` passait sur les mêmes modèles), jamais dans les deux
cents premiers jetons.

### La chasse

* **Cerner.** Prefill sain (5 jetons OK sur le prompt long), crash entre 120 et
  180 jetons générés ; indépendant du gabarit de chat, du cache de préfixe, de
  l'allocateur (segments extensibles, sans cache), de cuDNN, des noyaux
  hybrides, des GEMM de prefill, d'un workspace cuBLAS explicite. Dépendant de
  la **précapture des godets** (`ACVRAM_WARM_GRAPHS=0` guérit) et du **rejeu**
  (`ACVRAM_GRAPHS_EAGER=1`, mêmes formes sans capture, guérit). Le godet
  capturé à la demande est sain ; le même godet capturé pendant la précapture
  est mort.
* **Reproduire court.** `ACVRAM_WARM_GRAPHS=256`, prompt de 4 jetons, 300
  jetons : crash au premier rejeu du godet précapturé, à chaque fois.
* **Nommer le noyau.** `compute-sanitizer` est inutilisable ici : pas de PTX
  dans un binaire `sm_120f`, et la faute vient d'un noyau du pilote. `cuda-gdb
  -batch` (l'option `set cuda memcheck` n'existe plus en CUDA 13) arrête sur
  l'exception matérielle même dans un graphe rejoué : `memcpy32_post`, grille
  50 × 256 — soit 51 200 octets, 5 lignes × 5120 × bf16.
* **Trouver le tenseur.** Nouvel outil de diagnostic, `ACVRAM_TRACE_PTRS` :
  à la capture, `_empreinte_adresses` relève l'adresse de **tout** tenseur
  atteignable depuis le modèle (28 519 sur GLM-4.7-Flash) ; à chaque rejeu, il
  compare. Ceux qui ont bougé sont la cause. Une première version qui excluait
  les sous-modules ne voyait rien : la leçon vaut d'être notée.

### Les trois causes

1. **Le tampon de l'état caché MTP** (`_garder_hidden`) était alloué par
   `clone()` à la forme du pas : 5 lignes dans un graphe de vérification
   spéculative, 1 ligne au pas suivant, donc **réalloué** entre les deux. Le
   graphe spéculatif précapturé gardait l'adresse de l'ancien, rendu au pool.
   Corrigé : tampon réservé avant capture (`reserver_hidden`, 16 lignes au
   moins), écriture dans `buf[:n]`, le moteur pose `n` à chaque pas — un rejeu
   ne joue aucune affectation Python — et la tête ne lit que ces lignes. Effet
   collatéral probable : la tête MTP lisait un état périmé entre deux captures,
   ce qui peut expliquer son taux d'acceptation décevant (v0.4.40).
2. **La RoPE du chemin MLA** (`linear_attn.rope_emb`) s'étend au godet
   courant, `bucket + 1` : 1025 lignes au premier godet précapturé, 2049 puis
   3073 aux suivants. `_capture` ne pré-étendait que la RoPE de `self_attn`.
   Sur GLM-4.7-Flash, le graphe lisait des tables mortes, le routeur MoE
   recevait du bruit et la GEMV groupée des experts plantait sur des indices
   aberrants — c'est elle que cuda-gdb nommait ; l'empreinte a désigné
   `_cos`/`_sin`, seuls tenseurs sur 28 519 à avoir bougé. Corrigé : toute
   `RotaryEmbedding` du modèle est réservée à `max_model_len + MLA_BUCKET + 1`
   avant capture, et `_ensure` comme `tables32` refusent de réallouer sous
   capture.
3. **La tête de sortie rembourrée** (muse-glimmer-30b : 202 048 jetons,
   202 112 lignes) : vers le 250e jeton une colonne de rembourrage gagnait
   l'argmax et le plongement suivant tombait hors table (`IndexError`).
   Corrigé : `_logits_finaux` coupe au vocabulaire.

Et un artefact du banc : à 260 t/s, `urllib` lit 200 jetons dans un seul
bloc de 8 Kio, tous horodatés pareil — débit « infini ». Lecture non
bufferisée désormais, et le flux `completions` renvoie enfin son `usage`.

### La règle

Sous graphes CUDA, **toute allocation paresseuse est un fantôme en puissance**
— un `clone()` à la forme du pas, un cache étendu à la demande, une pile
construite à la première passe. Réserver à la taille finale avant la capture,
refuser la réallocation sous capture, et laisser `ACVRAM_TRACE_PTRS` armé au
premier doute : il désigne le coupable en un rejeu.

Le budget KV est par ailleurs borné par la VRAM réellement libre au
chargement (`_borner_kv_par_la_vram`, marge 1,5 Gio ou 5 %) : le budget du
manifeste raisonne sur des tailles nominales et laissait parfois 50 Mio.

## 5 septembre 2026, soir — le pire écart du comparatif : 26 → 196 t/s (v0.4.57, v0.4.58)

Le comparatif du jour donnait à acvram 34 modèles gagnés sur 68, mais deux
effondrements : Nemotron-Lightning-30B à 26 jetons/s contre 195 pour
llama.cpp. La médiane d'écart valait **+0 % en mono-carte et −84 % en
multi-cartes** — deux modèles seulement étaient étalés sur les deux GPU, et
c'étaient deux des trois pires.

### Une seule cause, quatre conséquences

`_build_layer` déduisait la taille d'une couche de la configuration, en
supposant que toutes se ressemblent : une attention plus un MLP, MoE au-delà
de `first_k_dense_replace`. Nemotron-H dément cette supposition — il alterne
23 couches Mamba sans MLP, 23 couches MoE sans attention et 6 couches
d'attention pure. Le compte annonçait **103 milliards de paramètres pour 31,6
réels**, sur six modèles du parc. La suite s'enchaîne :

1. le planificateur croit devoir exiler 27 Gio en mémoire vive et prend les
   deux cartes ;
2. les 14 dernières couches, attribuées à la 3080 Ti, sont **quantifiées en
   int4_awq** — sm_86 n'a pas de FP4 ;
3. sept couches MoE se retrouvent donc avec des experts int4 à échelles AWQ,
   que `_try_build_stacks` refuse d'empiler ;
4. elles retombent sur la boucle par expert : 142 indexations booléennes par
   jeton, chacune une synchronisation. Le profil dit tout — **619 ms de CPU
   pour 165 ms de GPU**, et la carte à 87 W.

### Les correctifs

Les paramètres se comptent désormais sur les tenseurs réels
(`_formes_du_point_de_controle` + `_affiner_couches`), lus dans le manifeste
d'un modèle converti, l'en-tête d'un GGUF (noms de llama.cpp traduits, experts
empilés dépliés) ou les en-têtes des fragments safetensors : quelques
kilooctets, aucun poids chargé. Plus aucun écart au-delà de 15 % sur les 110
modèles du parc, contre neuf avant.

Et au chargement, un plan figé qui étale sur deux cartes un modèle tenant sur
la première le rapatrie (`_rapatrier_sur_une_carte`, échappement
`ACVRAM_PLAN_FIGE`) — les modèles déjà convertis y gagnent sans reconversion.

| | avant | rapatriement seul | reconverti |
|---|---|---|---|
| débit | 26,4 t/s | 27,1 | **196,4** |
| puissance | 87 W | 87 | 141 |
| jetons/kJ | 249 | 310 | **1388** |
| plan | 38/14 couches, 15 en RAM hôte | 52 sur une carte | 52, tout NVFP4 |

llama.cpp fait 195,4 t/s à 214 W sur le même modèle : à débit égal, acvram
consomme **34 % de moins** (1388 jetons/kJ contre 913).

La leçon dépasse ce modèle : un planificateur qui devine la taille d'une
couche se trompera sur toute architecture qui sort du moule, et l'erreur ne se
voit pas — elle se lit dans un débit trois fois trop bas, six mois plus tard.
Les formes sont dans les fichiers ; il suffit de les lire.

### Le parc remis d'aplomb

Les sept modèles dont le plan figé était dégradé par l'ancien comptage ont été
repassés (`scratchpad/reconv-nemotron.py`, un modèle n'est remplacé qu'après
vérification que son nouveau plan tient sur une carte sans mémoire vive) :
cinq Nemotron en GGUF, deux en EXL3 — ces derniers ont d'abord été refusés par
la garde, le temps d'apprendre à lire les treillis d'exllamav3 (v0.4.59).

| modèle | avant | après | le meilleur des autres |
|---|---|---|---|
| Nemotron-Lightning Q6_K | 26,4 t/s | **196,4** | llama.cpp 195,4 |
| Nemotron-Lightning exl3-6bpw | 26,0 | **200,3** | TabbyAPI 170,1 |
| NVIDIA-Nemotron-3-Nano-30B | — | **200,7** | — |

Plus aucun modèle du parc n'a de plan dégradé. Au tableau du comparatif,
acvram passe de 34 à **36 modèles gagnés sur 68**, et la famille hybride-MoE
de onze modèles perdants à neuf, d'un écart médian de −45 % à −31 %.

Reste, par ordre de poids : les hybrides à experts (GLM-4.7 à −45 %, MLA plus
MoE), les MoE ordinaires (−15 %), les denses (−12 %) — et un modèle qui
déborde vraiment en mémoire vive, Qwen3-Coder-Next à 10 t/s contre 152, où
c'est le chemin d'exécution hôte qu'il faudra revoir, pas le placement.

## 5 septembre 2026, nuit — le godet MLA : cinq modèles, +7 % (v0.4.61)

Gisement suivant du comparatif : les hybrides à experts, GLM-4.7 en tête à
−45 %. Le diagnostic écarte d'emblée les causes de la veille — comptage juste
(29,9 G réels contre 29,9 comptés), plan sain, experts tous en NVFP4, piles
construites sur les 46 couches. Le profil différentiel (20 puis 60 jetons, la
différence isole le décodage) donne 10,9 ms de GPU par jeton, dont **2,6 ms
pour `mla_scores` et `mla_reduce`** — premier poste après les GEMV.

L'attention MLA balaie tout le godet de cache latent, pas seulement les
positions écrites. À godet fixe de 1024, une séquence de 264 jetons en payait
quatre fois trop. Un godet fixe et fin coûterait à l'inverse un graphe par
palier : 256 pour un contexte de 32 768, bien au-delà des seize qu'on capture.

Des **paliers doublants** depuis 128 tiennent les deux bouts — une courte
séquence ne lit que ce qu'il lui faut, le contexte entier ne demande que neuf
paliers :

| modèle | godet fixe | paliers | gain |
|---|---|---|---|
| GLM-4.7-Flash-Uncensored | 90,8 t/s | 98,4 | +8,4 % |
| GLM-4.7-Flash | 90,6 | 97,9 | +8,1 % |
| GLM-4.7-Grande-42B | 64,9 | 69,6 | +7,2 % |
| DeepSeek-Coder-V2-Lite | 163,2 | 174,3 | +6,8 % |
| Kimi-Linear-35B | 187,3 | 192,1 | +2,6 % |

Sur une réponse de mille jetons, GLM tient 80,9 t/s avec dix godets capturés :
les paliers suivent sans faire exploser le nombre de graphes.
`ACVRAM_MLA_BUCKET` règle le plancher.

Le profil désigne le chantier suivant, plus lourd : **environ 1600 lancements
de noyaux par jeton** — 330 GEMV et 939 opérations élémentaires — au point
qu'un seul `cudaGraphLaunch` coûte 2 ms. Le chemin MLA enchaîne trop
d'opérations PyTorch non fusionnées.

### Qwen3-Coder-Next : ce n'est pas le chemin, c'est la densité

Le pire écart restant du comparatif — 10 jetons/s contre 152 — n'est pas un
défaut d'exécution. Les trois stratégies de débordement donnent la même chose
(défaut 10,4 t/s, calcul sur processeur 9,0, plan figé 9,3) : le mode
automatique choisissait déjà la meilleure.

Ce qui diffère, c'est le volume. llama.cpp sert un **Q3_K_S de 32,2 Gio**,
soit 3,25 bits par poids ; notre conversion NVFP4 en fait **43,1 Gio**, soit
4,64 bits. Sur une carte de 30 Gio utiles, les deux débordent — mais eux de
deux gigaoctets, nous de treize, ce qui exile seize couches d'experts sur
quarante-huit.

Aucun réglage ne rattrapera 40 % de bits en plus. Un modèle de 80 milliards de
paramètres ne tiendra sur cette carte qu'en dessous de trois bits par poids ;
c'est un format à écrire, pas un chemin à corriger. En attendant, le
comparatif doit se lire ainsi : sur ce modèle, la comparaison oppose deux
densités autant que deux moteurs.

## 5 septembre 2026, nuit — une GEMV au lieu de trois : les projections NVFP4 empilées (v0.4.62)

Le profil différentiel de `Qwen3-Coder-30B` accusait le lancement des noyaux
plutôt que le calcul : 5,25 ms de GPU par jeton, dont 1,41 ms de `nvfp4_gemv`
répartis sur **193 appels par jeton**, soit quatre par couche. Trois d'entre
eux sont q, k et v, qui lisent tous la même activation. Le chemin INT8 les
empilait depuis longtemps ; le chemin NVFP4, non.

Micro-banc, une couche de 48 :

| | temps |
|---|---|
| trois GEMV séparées | 28,6 µs |
| une GEMV empilée | 9,6 µs |

Soit 0,91 ms par jeton sur 48 couches, 17 % du temps GPU.

### L'obstacle : trois échelles globales, pas une

Un tenseur NVFP4 porte une échelle scalaire par-dessus ses échelles de bloc.
Celles de q, k et v diffèrent d'un facteur deux (1,29e-4, 5,75e-5, 7,78e-5) et
les ramener à une valeur commune ferait passer les deux autres par un arrondi
e4m3 — 6 % d'erreur relative, pour une optimisation censée ne rien changer.

Le noyau accepte donc désormais une **échelle par ligne de sortie**
(`global_scale_rows`), lue une fois par bloc de `ROWS_PER_BLOCK = 4` lignes ;
l'empilement refuse tout segment qui ne commence pas sur un multiple de cette
hauteur, faute de quoi une ligne emprunterait l'échelle de sa voisine. Chaque
segment garde ainsi exactement les bits qu'il avait : la sortie fusionnée est
**identique à la concaténation des trois séparées**, écart maximal 0,000e+00,
et le texte généré ne change pas d'un caractère.

### Pas un octet de plus en mémoire

Le prefill continue d'appeler les projections une à une. Les conserver telles
quelles à côté de la pile doublerait leurs poids ; sur un dense de 27 milliards
de paramètres, gate et up recopiés coûteraient plusieurs gibioctets. Après
l'empilement, chaque original devient donc une **vue** de la pile — un découpage
sur la dimension de sortie, contigu par construction, sans copie.

### Ce que ça donne

| modèle | sans fusion | avec fusion | écart | VRAM |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B (MoE, 3 G actifs) | 185,3 t/s | 200,9 t/s | **+8,4 %** | 21 577 → 21 657 MiB |
| gemma-4-26B-A4B (MoE, 4 G actifs) | 98,9 t/s | 100,3 t/s | +1,4 % | 18 819 → 19 003 MiB |
| Qwen3.8-27B (dense) | 38,5 t/s | 39,5 t/s | +2,6 % | 18 393 → 18 377 MiB |

Le gain va d'abord aux modèles dont le décodage est borné par le lancement des
noyaux, c'est-à-dire ceux qui lisent peu de poids par jeton : `Coder-30B` en
active trois milliards et gagne 8,4 %, le dense de 27 milliards reste borné par
la bande passante mémoire et n'en tire que 2,6 %. Les 80 à 184 MiB d'écart de
mémoire sont le cache de l'allocateur après les concaténations, pas des poids
dupliqués — l'empreinte réelle ne bouge pas, et le dense en rend même 16.

### Une fusion qui n'atteignait que le tiers de sa cible

La première mesure ne donnait rien sur deux modèles sur trois. Cause : le
chargeur ne fusionnait gate et up que si le MLP était en INT8 —
`m.gate_proj.qweight.__class__.__name__ == "INT8Tensor"` — condition écrite du
temps où seul l'INT8 savait s'empiler, et jamais relue. Seule l'attention
profitait donc du nouveau chemin. Les experts d'un mélange en restent exclus,
eux, et pour une bonne raison : ils passent par le chemin groupé, qui empile
déjà les 128 d'un coup.

Sept tests fixent l'invariant, dont l'égalité au bit près de la GEMV empilée
sur GPU (`tests/test_fusion_nvfp4.py`), et `ACVRAM_FUSION_NVFP4=0` rend le
chemin d'avant pour toute mesure ultérieure.

## 5 septembre 2026, nuit — deux résultats négatifs sur GLM-4.7 (mesurés, écartés)

`GLM-4.7-Flash` rend 98 t/s là où llama.cpp en donne 163. Le profil différentiel
désigne d'abord le processeur : 11,9 ms d'hôte par jeton contre 9,8 ms de GPU,
dont **9 ms dans neuf `cudaMemcpyAsync`**. Ces copies partent bien de mémoire
paginée, où `non_blocking=True` ne veut rien dire.

Les faire transiter par des tampons hôtes épinglés persistants — deux jeux
alternés, table de blocs assemblée côté hôte puis transmise d'un bloc — n'a rien
donné, et pire :

| modèle | copies paginées | tampons épinglés |
|---|---|---|
| GLM-4.7-Flash | 98,4 t/s | 98,0 t/s |
| GLM-4.7-Grande-42B | 71,0 t/s | 72,0 t/s |
| Qwen3-Coder-30B-A3B | 201,0 t/s | **174,6 t/s** |

Treize pour cent de perte sur le modèle le plus rapide : la copie hôte vers hôte
ajoutée coûte plus que ce que l'épinglage fait gagner, et les 9 ms du profil
étaient de l'attente du GPU comptée à l'appelant, pas un coût d'hôte. Le
changement est retiré. Leçon à retenir : sur un profil PyTorch, un
`cudaMemcpyAsync` cher se lit comme une synchronisation, pas comme un transfert.

Ce que le profil dit vraiment, lui, tient dans le décompte des lancements :
**environ 1 740 noyaux par jeton**, dont 283 `nvfp4_gemv` — six par couche sur
47 — et quelque 1 080 noyaux élémentaires, soit 23 par couche. À deux
microsecondes de latence chacun, le seul fait de les lancer explique les 9,8 ms
de GPU. Un `cudaGraphLaunch` à 3,4 ms par jeton dit la même chose autrement :
son coût suit le nombre de nœuds du graphe. C'est là qu'est le gisement, pas
dans les transferts.

## 5 septembre 2026, nuit — l'attention latente lançait une GEMV de trop (v0.4.63)

Suite du décompte précédent : sur `GLM-4.7-Flash`, six multiplications par
couche et 47 couches. L'une d'elles n'avait pas à exister. `q_a_proj`
[768, 2048] et `kv_a_proj` [576, 2048] lisent toutes deux l'état caché, comme
dans toutes les variantes DeepSeek-V2 et GLM, et `fuse_projections` les
écartait explicitement :

```python
if self.q_a_proj is not None:
    self.q_kv = None
    return False
```

Le code ne savait empiler que la paire `q_proj`/`kv_a_proj` des modèles sans q
de bas rang, et seulement en INT8. Les deux restrictions tombent : la paire
empilée est choisie selon l'architecture, et l'empilement passe par l'INT8 puis
par le NVFP4 de la v0.4.62.

| modèle | avant | après |
|---|---|---|
| GLM-4.7-Flash-Uncensored-Heretic | 98,4 t/s | 103,4 t/s (+5,1 %) |
| GLM-4.7-Grande-Heretic-42B | 71,0 t/s | 74,3 t/s (+4,6 %) |

Texte identique dans les deux cas. La mesure porte deux fois le même code, à
0,1 % près : le bruit est bien plus petit que l'écart.

Reste, sur ces modèles, l'essentiel du décompte : environ 1 080 noyaux
élémentaires par jeton, 23 par couche. C'est le prochain gisement, et il ne se
prendra pas par des empilements mais par des noyaux fusionnés.

## 5 septembre 2026, nuit — un type de jeton oublié, trente-cinq modèles atteints (v0.4.64)

Cherchant à vérifier qu'une optimisation ne changeait pas le texte, on a
découvert que le texte était déjà faux. `GLM-4.7-Flash` interrogé en
conversation répondait par une cascade de balises vides — `</arg_value>`, puis
`<think></think>` jusqu'à la limite de jetons.

Le gabarit rendait bien `[gMASK]<sop><|user|>…<|assistant|><think>`, mais
l'encodeur en tirait `…, 154828, 27, 26779, 29` : `<`, `think`, `>` en trois
jetons ordinaires au lieu du 154841 qui existe pourtant dans le vocabulaire.

### La cause

Le GGUF classe ses jetons par type : 3 pour les jetons de contrôle, 4 pour ceux
que l'auteur du modèle a définis. La conversion ne rendait insécables que les
premiers :

```python
if i < len(ttypes) and ttypes[i] == 3
```

Le chemin SentencePiece, lui, prenait déjà les deux (`in (3, 4)`) — la
divergence entre les deux chemins était l'indice, et personne ne l'avait vue.
Sur GLM, `<think>`, `</think>` et les neuf balises d'appel d'outil sont toutes
de type 4.

### La portée

**Trente-cinq modèles convertis** portaient un vocabulaire incomplet, dont
`Huihui-Qwen3.6-35B-A3B-abliterated`, l'un des deux que le comparatif du jour
notait comme muets sous gabarit de conversation. Il répond depuis :

> La capitale de la France est **Paris**. C'est la ville la plus peuplée du
> pays et son centre politique, culturel et économique.

Tous les modèles Qwen3 du parc perdaient `<tool_call>` et `<tool_response>` :
la mention « agent-ok » de leurs fiches était optimiste.

`outils/reparer-jetons.py` répare un modèle déjà converti sans toucher un seul
poids — le GGUF d'origine porte les types, seul le `tokenizer.json` est à
réécrire, et l'ancien est gardé en `.avant-jetons`.

### Ce qui reste

`GLM-4.7-Flash` va mieux sans être guéri : le préfixe est correct, mais le
modèle répète `</think>`. C'est un autre défaut, à chercher ailleurs que dans
le vocabulaire.

## v0.4.64 — la normalisation de l'attention latente passe au noyau

Sept noyaux élémentaires — conversion, carré, moyenne, racine inverse, deux
multiplications, reconversion — deux fois par couche et quarante-sept couches.
Le noyau `rmsnorm_bf16`, qui sert déjà toutes les autres normalisations du
modèle, fait le même calcul en un lancement. Et q comme k tournent désormais
d'un seul bloc, la rotation étant point à point.

| modèle | v0.4.63 | v0.4.64 |
|---|---|---|
| GLM-4.7-Flash-Uncensored-Heretic | 103,4 t/s | 113,4 t/s (+9,7 %) |
| GLM-4.7-Grande-Heretic-42B | 74,3 t/s | 83,1 t/s (+11,8 %) |

Une honnêteté s'impose sur l'exactitude. Un premier test sur quatre lignes de
tirage avait conclu à l'égalité au bit près ; sur trente et un millions de
valeurs, l'écart est de six par million, toujours d'un seul cran de la grille
bfloat16. Le noyau reproduit fidèlement les arrondis intermédiaires mais somme
les carrés dans un autre ordre, et sa réduction par échange de warp donne
parfois un dernier bit différent. Aucune des deux sommes n'est plus vraie que
l'autre, et c'est déjà le noyau qui sert partout ailleurs. Le test dit
maintenant ce qui est garanti — un écart borné à un cran sur moins d'un
dix-millième des valeurs — plutôt qu'une égalité que quatre lignes avaient fait
croire.

## 6 septembre 2026 — comparatif 20260906, appels d'outils, et ce que le banc ne voyait pas (v0.4.65)

### Le comparatif refait

Même protocole que la veille, 132 couples sur 66 modèles, quatre moteurs, une
nuit de mesures après quatre versions d'acvram.

| moteur | modèles gagnés | médiane t/s | médiane j/kJ | chargement médian |
|---|---|---|---|---|
| acvram | 33 | 107 | 492 | 35 s |
| llama.cpp | 24 | 155 | 717 | 195 s |
| vLLM | 5 | 190 | 1064 | 137 s |
| TabbyAPI | 4 | 57 | 237 | 163 s |

La médiane d'acvram passe de 89 à 107 t/s. llama.cpp n'a pas bougé (+0,4 % de
médiane sur 39 modèles communs), ce qui fait de lui un bon étalon : les écarts
sont bien les nôtres. Vingt-six modèles gagnent plus de 5 %, dont les cinq à
attention latente (GLM-4.7-Flash 89 → 112, Grande 63 → 80) et les Nemotron
Lightning (+19 %). L'écart médian des hybrides à experts, le pire poste, passe
de −31 % à −19 %.

### Ce que le banc ne voyait pas

Neuf modèles reculent, et ils ont tous un point commun : leur vocabulaire a été
réparé la veille (v0.4.64). Le banc note le nombre de jetons produits, et il
suffit de le lire :

| modèle | 5/09 | 6/09 |
|---|---|---|
| KAT-Coder-V2.5-Dev | 481 t/s sur **4 jetons** | 165 t/s sur 118 jetons |
| Qwen3.6-35B-A3B-APEX | 477 t/s sur **4 jetons** | échec |

Un modèle qui produit quatre jetons puis s'arrête n'a pas de débit. Les 481 t/s
de la veille étaient la vitesse d'une réponse vide, et les « régressions » de
Qwen3.5-35B (234 → 154) ou Qwen3.8-27B-UD (66 → 50) sont du même ordre : hier
du texte dégénéré que la spéculation par n-grammes acceptait par paquets,
aujourd'hui du texte. Le comparatif de la veille surestimait acvram sur les
modèles au vocabulaire cassé ; celui-ci est le premier qu'on peut lire sans
astérisque. Leçon pour le banc : une mesure sur moins de 150 jetons ne vaut
rien, et la spéculation par n-grammes doit être notée à côté du débit.

### Les appels d'outils n'existaient pas

Étape 5 de la liste : « revalider agent-ok ». Quatre modèles Qwen3, un outil
`meteo`, une question sur Lyon. Aucun n'a rendu d'appel : `Qwen3-Coder-30B`
expliquait poliment qu'il n'avait pas accès à la météo. Le serveur acceptait le
champ `tools` et ne le transmettait à rien.

Trois pièces manquaient :
- les outils et les messages complets (`tool_calls`, `tool_call_id`, `name`)
  passent au gabarit Jinja, qui sait les rendre — c'est lui qui décrit les
  outils au modèle, dans le format que ce modèle a appris ;
- la sortie est relue : `<tool_call>{json}</tool_call>` (Qwen3, Nemotron,
  Hermes, KAT) et le XML de Qwen3-Coder
  (`<function=nom><parameter=clé>valeur</parameter></function>`) deviennent des
  `tool_calls` au format OpenAI, avec `finish_reason: "tool_calls"` ; un bloc
  dont le JSON ne se lit pas reste dans le texte, mieux vaut un appel manqué
  qu'un appel inventé ;
- en flux, tout ce qui suit un `<tool_call>` est retenu et rendu à la fin en
  un seul delta structuré, et un `<` isolé attend quelques caractères le temps
  de savoir s'il ouvre la balise.

Après quoi `Qwen3.8-27B-UD` rend `meteo {"ville": "Lyon"}` et
`Qwen3-Coder-30B` produit son XML. Dans le même mouvement, `chat_template_kwargs`
atteint le gabarit, comme chez vLLM et llama.cpp : `{"enable_thinking": false}`
coupe la réflexion d'un Qwen3.

### Deux familles qui déraillent sur les prompts longs

`Qwen3.6-35B-A3B-APEX` termine après deux jetons sur le prompt du banc et répond
correctement à une question courte ; `Huihui-Qwen3.6-35B-A3B` s'arrête à
quarante jetons au banc et répond « user » dès que la description des outils
allonge le prompt. Les deux sont des hybrides à état récurrent. `GLM-4.7-Flash`,
à attention latente, déraille dès treize jetons quand le préfixe contient les
jetons de conversation, alors que llama.cpp raisonne sur le même préfixe.

Ce qui est établi pour GLM : poids fidèles (cosinus 0,995 sur toute la tête,
plongement exact), routage identique à llama.cpp jusqu'au biais de sélection,
noyau de GEMM groupée juste sur les neuf cas d'un nouveau test — tuiles
partielles comprises — et même charabia avec le repli en bfloat16. Ni les
graphes, ni le cache de préfixe, ni le vocabulaire. Reste le calcul de
l'attention latente elle-même au prefill, et pour Qwen3.6 le prefill des
couches à état : c'est le prochain chantier, et il commence par une
comparaison des logits contre llama.cpp, préfixe par préfixe.

### Le parc garde 59 modèles aux anciennes promotions

Le profil de `LFM2.5-8B-A1B` (étape 3 : 362 t/s contre 542) montre 47 appels
par jeton d'une GEMV **INT8** — le modèle a été converti avec le plancher de
SNR à 25 et garde 56 tenseurs promus. Il n'est pas seul : 59 modèles du parc
sont dans ce cas, dont `Ornith-1.0-35B` (309 tenseurs), `Kimi-Linear` (271) et
`Qwen3-Coder-30B-A3B` (193). La reconversion de la veille ne couvrait que les
38 alias du menu d'alors. Les reconvertir est la suite logique de la décision
du plancher à zéro.

`outils/reparer-jetons.py` retrouve désormais un GGUF dont le nom ne
correspond pas exactement au dossier converti : quatre modèles de plus réparés,
dont `Ornith-1.0-35B` et `Falcon-H1R-7B` (198 jetons perdus).

## 6 septembre 2026, aube — la v0.4.63 avait cassé l'attention latente (v0.4.66)

Le charabia de GLM-4.7 en conversation, poursuivi toute la nuit, n'avait rien à
voir avec la conversation. **Tout prefill de dix jetons ou plus** rendait
« de de de de », quel que soit le texte. Un `git worktree` sur la v0.4.62 l'a
tranché en trois minutes : treize et trente-deux jetons y donnent du texte
sain. Le coupable est donc la v0.4.63, la mienne.

### La cause

La v0.4.63 fait passer la pile `q_a_proj`/`kv_a_proj` dans
`MLAttention._proj_entree` sans condition sur le nombre de jetons. Une pile
porte une échelle globale par ligne (`global_scale_rows`, v0.4.62), que seul
le noyau GEMV lit. Au-delà de huit jetons, le backend `fp4-tensorcores`
(priorité 110) et le chemin W4A8 prennent `t.global_scale` — celle du premier
segment — pour toute la pile : le latent kv sortait à une échelle fausse d'un
facteur quarante. L'attention et le MLP étaient protégés par un
`if … and t <= 8` que je n'avais pas reproduit.

### Le correctif

Les deux chemins tensor cores déclinent une pile (`return None`) ; le
dispatch la déquantifie alors avec son échelle par ligne, exacte au bit près,
avant un produit bfloat16. Un test couvre désormais n ∈ {1, 4, 8, 9, 16, 64} :
au bit près contre les GEMV séparées sous huit jetons, contre un produit
bfloat16 exact au-delà.

### Deux pièges de diagnostic, à retenir

- **Un détecteur qui ment.** Mon critère de charabia (`len(set(texte)) <= 2`)
  jugeait « de de de de » acceptable. J'ai conclu que le texte ordinaire
  passait à toutes les longueurs et cherché quatre heures du côté des jetons
  de conversation, du vocabulaire, du routage, de la GEMM groupée et du cache.
  Un test de texte affiche le texte, il ne rend pas un booléen.
- **Une référence approximative.** Le premier test de la pile la comparait aux
  projections séparées, qui passent elles-mêmes par les tensor cores FP4 à
  2–9,5 % d'erreur : la pile corrigée, exacte, y paraissait fausse à 90 %.
  L'étalon d'un chemin de prefill est un produit bfloat16, pas un autre chemin
  de prefill.

### Conséquence sur le comparatif 20260906

Les trois GLM y ont été mesurés en charabia : 112, 80 et 107 t/s de « de de
de ». Ces trois lignes sont refaites ci-dessous, et le banc apprend au passage
à écrire un aperçu du texte produit à côté du débit.

### Les lignes GLM refaites (7 h 06)

| modèle | débit | aperçu du texte produit |
|---|---|---|
| GLM-4.7-Flash-Uncensored-Heretic | 111,7 t/s | « 1. Analyze the Request: Topic: Virtual Memory in a… » |
| GLM-4.7-Grande-Heretic-42B | 80,1 t/s | « 1. Analyze the Request: Topic: Virtual memory (VM)… » |
| Huihui-Kimi-Linear-REAP-35B | 192,9 t/s | « La mémoire virtuelle est un mécanisme fondamental… » |

Les débits sont ceux de la veille au dixième près : le charabia se générait
exactement à la vitesse du texte, et c'est bien pour cela qu'un débit seul ne
prouve rien. La colonne `apercu` du banc existe pour ça. `NEO-CODE` n'a pas
démarré en quinze minutes pendant cette passe — un convertisseur zombie de la
reconversion avortée saturait la carte — et sera remesuré après la reconversion.

## 7 septembre 2026, nuit — le planificateur croyait le processeur plus rapide qu'une carte graphique (v0.4.67)

Le comparatif du 6 septembre donnait acvram à parité avec llama.cpp : sur les
34 modèles mesurés par les deux moteurs, avec au moins 150 jetons, le ratio
médian est de 0,97 et acvram gagne sur 15. La moyenne ne disait donc rien, et
tout l'écart restant tenait dans la queue. Un seul modèle en portait
l'essentiel : Qwen3-Coder-Next en Q3_K_S, 10,3 jetons par seconde contre
155,1. Facteur quinze.

### Deux fausses pistes, écartées par les données

La première était le format. Q3_K_S est un K-quant, et l'on pouvait croire
qu'il n'avait pas de chemin de lecture chez nous. Les débits médians par
famille l'écartent : Q4_K sur dix-huit modèles à 0,89, Q5_K sur neuf à 1,14,
Q6_K sur deux à 1,14, et un format trois bits, IQ3_M, à 368 jetons par seconde.
Réserve d'honnêteté : Q3_K n'a qu'un représentant dans tout le parc, et c'est
le modèle en cause ; format et modèle restent confondus.

La seconde était le débordement, et c'était la mienne. Le converti pèse
44 gigaoctets pour 33 de source — NVFP4 fait 4,5 bits par poids là où la
source en fait 3,4, donc convertir ce modèle le fait grossir d'un tiers — et
le plan exilait 14,11 Gio en mémoire hôte, soit 34 % des poids. Explication
séduisante, et fausse : un Qwen3.5 de 4 milliards a **42 %** de ses poids en
RAM hôte et rend 200 jetons par seconde. L'erreur du planificateur, tracée
contre le taux d'exil, est plate : 3,38× entre 2 et 6 % d'exil, 2,38× entre 6
et 15 %, 2,41× au-delà de 15 %. La proportion n'explique rien.

### La cause

Trois lignes de `acvram/memory/tiering.py` :

```
144:  host_compute_gb_s: float = 70.0    # DDR5 streaming reads, measured by bench
459:  lp.mlp_exec = "cpu" if opts.host_compute_gb_s > link else "gpu"
521:  seconds += active / (opts.host_compute_gb_s * 1e9)
```

Soixante-dix gigaoctets par seconde est un débit de lecture séquentielle en
DDR5. Ligne 521 il chiffre un produit de matrices NVFP4 qui doit déballer de
l'E2M1 sans instruction native et reconstruire les échelles par bloc : ce
n'est pas une lecture, c'est du calcul dense. Ce n'est pas une constante mal
calibrée, c'est la mauvaise grandeur — aucune valeur de débit mémoire ne
représentera jamais le coût de ce noyau.

Ligne 459, la même constante décide *où l'on calcule*. Soixante-dix étant
supérieur aux 18,7 du bus, tout perceptron exilé était calculé sur le
processeur. Sur ce modèle, **77 % du temps prédit tombait dans cette branche**,
la moins bien estimée de toutes.

Le pire est ailleurs. La sélection de configuration classait les candidats sur
`est_decode_tok_s`, donc sur cette estimation, et le manifeste porte :

```
"cuda:1 laisse oisif a dessein : le modele tient sans lui, et ajouter
 une tranche plus lente au pipeline couterait du debit."
```

Le modèle ne tenait pas sans elle. Quatorze gibioctets de perceptrons sont
partis sur le processeur pendant que **les 12 Gio de la 3080 Ti dormaient**.
Une constante fausse a fait croire le processeur plus rapide qu'une carte
graphique, et a choisi le plus lent des trois chemins en écrivant que c'était
délibéré.

### Ce qui change

* `host_gemm_gb_s`, nouvelle option, vaut `None` par défaut : le débit de
  calcul hôte n'a jamais été mesuré. Tant qu'il ne l'est pas, `host_exec=auto`
  ne choisit plus le processeur — on ne s'engage pas sur un chemin dont on
  ignore le coût, alors que le transfert vers le GPU est borné par le bus.
* La sélection de configuration écarte d'abord l'exil de poids de couche,
  et n'optimise le débit estimé qu'ensuite. Les plongements, qui vivent
  toujours en RAM, ne comptent pas comme un exil.
* L'avertissement sur une carte oisive dit désormais la vérité quand des poids
  sont en RAM hôte, au lieu d'affirmer que le modèle tient sans elle.

Nouveau plan de Coder-Next : la 3080 Ti prend les couches 35 à 47, 9,5 Gio ;
les poids en RAM hôte tombent de **14,11 Gio sur 16 couches calculées par le
processeur à 3,9 Gio sur 4 couches transférées au GPU**. Aucun changement sur
les modèles qui tenaient déjà : Nemotron-Lightning Q6 et GLM-4.7-Grande
laissent toujours la seconde carte oisive, avec l'ancien libellé, et n'ont que
leurs plongements en RAM.

### Ce qui n'est pas mesuré, et doit l'être

Le gain réel. Rien n'a pu être mesuré cette nuit : la machine compilait deux
ROM à trente-deux fils, charge moyenne 130. Trois mesures attendent :

1. Chronométrer une seule couche processeur, prédite à 0,3 ms. Si elle tient
   en 0,3 ms, toute cette analyse tombe.
2. L'occupation réelle du bus pendant le décodage, avant le correctif, pour
   avoir un point de comparaison — pas la puissance : 70 W sur une carte
   bridée à 400 dit l'attente, pas la saturation.
3. Coder-Next rebancé.

### Un défaut plus large, découvert en chemin

Le planificateur n'avait jamais été confronté à une mesure. Sur les 61 modèles
du parc, sa prédiction est optimiste d'un facteur 2,4 à 3,4. En écart absolu,
hors Coder-Next : 6,26 ms par jeton en médiane, avec une pente de +2,46 ms par
doublement de taille. Ce n'est donc ni un plancher fixe pur ni un pur défaut
de débit ; les deux termes coexistent. Environ 3 ms de plancher aux petites
tailles, plus une composante croissante qui accuse les 1792 Go/s de plaque
retenus pour une carte bridée à 400 W et qu'aucun noyau réel n'atteint.

Un biais presque uniforme ne change pas l'ordre des candidats, ce qui explique
que personne ne l'ait vu. Mais celui-ci n'était pas uniforme : il a basculé un
arbitrage serré, et c'est exactement ce qui a coûté la 3080 Ti.

## 7 septembre 2026, nuit — correction de la section précédente, et les deux vraies causes (v0.4.68)

La section v0.4.67 attribue au correctif un effet qu'il n'a pas. Je l'ai
écrite en comparant le plan que rend le code d'aujourd'hui au plan **stocké
dans le manifeste**, et j'en ai conclu que le correctif recrutait la 3080 Ti.
Une comparaison stricte le dément : `git worktree` sur 6b15c32, même script,
mêmes options, les 111 modèles du parc replanifiés par les deux versions.

| | avant (6b15c32) | après (v0.4.67) |
|---|---|---|
| modèles dont le plan change | — | **1 sur 111** |
| Coder-Next, cartes | cuda:0 + cuda:1 | cuda:0 + cuda:1 |
| Coder-Next, poids exilés | 3,32 Gio | 3,32 Gio |
| **Coder-Next, couches calculées par le processeur** | **4** | **0** |

Identique aux contextes 8192 et 32768. Le code d'avant recrutait donc déjà les
deux cartes. Le seul effet réel de la v0.4.67 est de retirer quatre couches de
la branche processeur — utile, mais bien moins que ce que j'avais écrit.

### Alors d'où venait le plan à une carte et seize couches processeur ?

Du manifeste, écrit à la conversion et jamais rejoué. `_plan_from_manifest`
reconstruisait le plan **verbatim**, étages compris, et `_reajuster_plan` ne
sait que faire descendre des MLP, jamais remonter. Aucune amélioration du
planificateur n'atteignait un modèle déjà converti — et il y en a 111.

### Et une seconde cause, dans le chargeur, qui défaisait le correctif

`_reajuster_plan` contenait :

```python
l.mlp_storage = "cpu"
if hasattr(l, "mlp_exec"):
    l.mlp_exec = "cpu"      # <-- ici
```

Descendre un perceptron en RAM dit où il est **stocké**, pas où il est
**calculé**. Cette ligne forçait le calcul sur processeur à chaque
chargement, écrasant la décision du planificateur. C'est elle, et non le
planificateur, qui a mis seize couches de Coder-Next sur le processeur au
banc du 6 septembre. Le correctif de la v0.4.67 aurait été défait ici.

### Ce qui change en v0.4.68

* `_replanifier` : quand les cartes du plan figé ne sont pas celles de la
  machine, le planificateur est rejoué, puis réajusté sur les octets réels.
  Sinon le plan figé fait foi, pour que les mesures du parc restent
  comparables.
* Un MLP descendu en RAM garde `mlp_exec = "gpu"` : ses poids traversent le
  bus tant que le débit de calcul hôte n'est pas mesuré.

Plan effectif de Coder-Next au chargement, désormais : les deux cartes, dix
couches exilées, **aucune calculée sur le processeur**, 8,49 Gio stockés en
RAM mais seulement **164 Mio actifs par jeton** — soit 8,8 ms sur un bus à
18,7 Go/s, donc un plafond de bus vers 114 jetons par seconde, contre 10,3
mesurés avant. 151 tests passent.

### Toujours pas mesuré

Le gain reste inconnu : la machine compile encore. Et une remarque de
relecture, retenue et non appliquée : la fonction de rang de la v0.4.67
compte les octets **stockés**, alors que le coût se paie en octets **lus par
jeton**. Sur ce modèle le rapport est de quarante-cinq — 40,6 Gio de
perceptrons stockés pour 0,896 Gio actif. Un plan qui exilerait 3 Gio
d'experts (60 Mio lus) serait donc classé derrière un plan qui exilerait
2 Gio de poids denses (2 Gio lus), trente fois plus cher. À corriger en
comptant `mlp_active_bytes * (1 - cached_expert_fraction)` — après la mesure,
pas avant.

## v0.4.69 — de quoi mesurer les deux correctifs séparément

Les v0.4.67 et v0.4.68 sont empilées et aucune n'est mesurée. Les mesurer
ensemble ne dirait pas laquelle agit. Deux témoins d'environnement rendent
chaque moitié débrayable, et donnent une matrice à quatre cases que l'on peut
parcourir dans la même session, sur la même machine au repos :

| témoins | cartes | couches exilées | calculées CPU | stocké | actif/jeton |
|---|---|---|---|---|---|
| aucun (corrigé) | 2 | 10 | **0** | 8,49 Gio | 164 Mio |
| `ACVRAM_MLP_HOTE_CPU=1` | 2 | 10 | 6 | 8,49 Gio | 164 Mio |
| `ACVRAM_SANS_REPLAN=1` | 1 | 18 | 16 | 15,28 Gio | 335 Mio |
| les deux | 1 | 18 | 18 | 15,28 Gio | 335 Mio |

La troisième ligne reproduit exactement la condition du banc du 6 septembre,
qui a rendu 10,3 jetons par seconde. `ACVRAM_PLAN_FIGE` désactive aussi la
replanification, pour rester cohérent avec son nom.

### Combien de plans figés sont réellement périmés : deux sur cent onze

La replanification de la v0.4.68 ne se déclenche que si les cartes du plan figé
ne sont pas celles de la machine. Objection légitime : faut-il replanifier
systématiquement, au risque de changer le comportement de modèles qui marchent,
ou reconvertir le parc, au prix de plusieurs heures de GPU et de disque ?

La question se tranche par la mesure plutôt que par le jugement. Plan figé de
chaque manifeste comparé au plan que rend le planificateur d'aujourd'hui, sur
les 111 modèles du parc :

* **109 plans figés sont déjà identiques** à celui d'aujourd'hui ;
* 2 diffèrent : Qwen3-Coder-Next (une carte au lieu de deux, seize couches
  processeur au lieu d'aucune) et `qwen3b-pipeline` (deux cartes au lieu d'une,
  aucun exil ni avant ni après).

Les deux divergences sont des désaccords sur le nombre de cartes, donc toutes
deux déjà couvertes par la règle conservatrice. Replanifier systématiquement
changerait exactement les mêmes deux modèles. Il n'y a donc rien à reconvertir,
et aucune décision à prendre : la règle en place attrape la totalité des cas.

## v0.4.70 — une empreinte du planificateur dans le manifeste

Le constat « 109 plans figés sur 111 sont déjà ceux d'aujourd'hui » est vrai
aujourd'hui et cesse de l'être à la prochaine correction du planificateur. La
garde de la v0.4.68 se déclenche sur un désaccord de **matériel** ; or une
correction du modèle de coût change les plans sans qu'aucune carte ne bouge.
La garde ne verrait rien, les 111 manifestes redeviendraient périmés en
silence, et le défaut se redécouvrirait par le même chemin : un modèle vingt
fois trop lent et une soirée pour comprendre pourquoi.

`empreinte_planificateur()` hache le code qui décide — `plan_placement`,
`_estimate`, `auto_plan` — et l'écrit dans le plan du manifeste. Le chargeur
replanifie si le matériel a changé **ou** si l'empreinte diffère, et dit lequel
des deux motifs s'applique. Les manifestes existants n'ont pas d'empreinte,
donc ils replanifient tous : sans risque, puisque 109 d'entre eux rendent le
même plan, et les 2 autres sont ceux qu'on voulait corriger.

### Protocole de la mesure à venir

La matrice à quatre cases se mesure **en encadrant la série par deux mesures de
la même case de référence**, au début et à la fin. Une machine qui vient de
compiler deux ROM met du temps à retrouver des caches et des fréquences
stables ; sans cet encadrement, une partie de l'écart entre la première et la
dernière case ne serait que du réchauffement. Si les deux mesures de référence
diffèrent de plus de quelques pour cent, aucune des quatre cases ne vaut rien.

## v0.4.71 — l'empreinte du planificateur retirée : elle ne servait à rien

L'empreinte de la v0.4.70 supposait qu'une replanification coûtait cher. Personne
ne l'avait mesuré. Mesure faite :

| opération | coût |
|---|---|
| `auto_plan` sur Coder-Next, 48 couches | 5,1 ms |
| `auto_plan` sur un modèle moyen | 2,1 ms |
| `detect_rig` | 73,2 ms |
| **replanification complète au chargement** | **148 ms** (Coder-Next), 54 ms (petit modèle) |

Contre un chargement qui lit quarante-quatre gigaoctets sur disque, c'est
gratuit. L'empreinte coûtait à elle seule 1,86 ms, soit plus du tiers d'une
replanification, pour éviter une replanification.

Elle est donc retirée, et le plan est **toujours** recalculé au chargement à
partir du matériel constaté. On supprime avec elle sa maintenance et la question
« ai-je bien haché toutes les fonctions dont le plan dépend ? », dont on
découvre la mauvaise réponse six mois plus tard, par le même chemin qu'ici.
`ACVRAM_PLAN_FIGE` et `ACVRAM_SANS_REPLAN` gardent l'ancien comportement pour
reproduire une mesure ancienne.

C'est la même leçon que `host_compute_gb_s` : un mécanisme dont personne n'a
vérifié qu'il servait à quelque chose finit par coûter plus que ce qu'il évite.

## 7 septembre 2026, aube — le chemin « exilé, calculé sur la carte » n'avait jamais tourné (v0.4.73)

Onze mesures de référence dans la nuit, toutes entre 9,7 et 10,5 jetons par
seconde sur Coder-Next : le banc reproduit le 10,3 du 6 septembre. Les cases
corrigées, elles, ont échoué six fois à six endroits différents, chacun un
défaut du chemin « poids stocké en RAM hôte, calculé sur la carte », qui
existait dans le planificateur et n'avait jamais été exécuté sur un modèle
réel puisque le chargeur forçait toujours le calcul processeur.

| n° | défaut | correction |
|---|---|---|
| 1 | le chargeur forçait `mlp_exec = "cpu"` (v0.4.68) | garde `gpu` |
| 2 | capacité d'étage sur la mémoire totale, 5,1 Gio de llama-server ignorés (v0.4.72) | mémoire libre |
| 3 | le routeur, lu à cru par `MoEBlock`, restait sur le processeur | routeur résident |
| 4 | `_rehydrate` ignorait INT8, format des experts partagés | branche ajoutée |
| 5 | double tampon par expert : deux copies de la couche entière sur la carte | `ExpertPool` par couche, dimensionné pour les experts routés |
| 6 | course : le flux annexe écrivait un bloc alloué sur le flux courant sans attendre | événement d'allocation et de libération |

Le sixième est établi par bisection sur le serveur, à conditions du banc :
lancements CUDA bloquants → succès ; synchronisation après chaque couche →
succès, 10,0 t/s ; préchargement d'avance désactivé → succès, 10,0 t/s ;
copies du pool sur le flux de calcul → échec ; sans graphes → échec. La
correction de la course est écrite ; **les tests unitaires passent, l'essai
serveur n'a pas été refait dessus** (pause demandée). Repli prouvé :
`ACVRAM_SANS_PRECHARGE=1`.

Test à sec de la couche 32 exilée sur pool : 17 Mio sur la carte au lieu de
0,88 Gio, écart 5·10⁻⁴ sur une amplitude de 0,12, 3,1 ms par jeton. Mais les
deux cases qui marchent rendent 10,0 t/s, soit la référence, alors que 18
couches à 3,1 ms ajoutées aux 97 ms de la référence donneraient 6,5. Le
chiffre à sec est le cas le plus défavorable, pas le coût courant, et deux
chemins très différents rendant le même débit disent que le goulot est
ailleurs. Mesure suivante, en service : faire varier le nombre de couches
exilées, 0, 9, 18, même prompt, références de part et d'autre.

Ajouts de diagnostic : le serveur journalise la pile de ses erreurs ;
`ACVRAM_POOL_SYNC`, `ACVRAM_SANS_PRECHARGE`, `ACVRAM_SYNC_COUCHES`.
Notés, non traités : l'échec FP4 tensor cores sur la 3080 Ti éteint le chemin
pour toutes les cartes ; 39 Gio de RSS avec les experts épinglés ; la
fonction de rang compte les octets stockés, pas lus.

## 7 septembre 2026 — ce que la bibliographie a condamné dans notre propre code

Second relevé fait le 7 (voir `docs/BIBLIOGRAPHIE.md`, fiches 15 à 26). Deux
conséquences directes pour le code, et deux mesures que personne n'a publiées.

**Un chiffre qui arbitre et qui n'a jamais été mesuré.** `tiering.py` calcule
`cached_expert_fraction` comme le rapport entre la VRAM restante et les octets
d'experts exilés, multiplié par 1,3 « pour le biais de routage », plafonné à 1.
Sur Qwen3-Coder-Next il vaut **2,9 %**, et `_estimate` s'en sert pour décider
d'un placement. L'article `arXiv:2608.07911` montre qu'un taux de succès annoncé
de 37,99 % s'explique à 96 % par la seule fragmentation de capacité, que l'ordre
de rejeu d'une trace déplace l'écart à l'optimum de 44,9 à 30,8 points, et
qu'une relecture incohérente gonfle les politiques de récence de 27 à 29 %.
Notre 2,9 % n'est donc pas un taux de succès : c'est un vœu avec une décimale,
et il pèse sur un arbitrage. À mesurer avant d'écrire le cache d'experts que la
feuille de route promet depuis le début.

**Le lien PCIe est en huit voies, pas seize.** Vérifié le 7 dans
`/sys/bus/pci/devices` : la 5090 est capable de 32 GT/s et la 3080 Ti de 16,
mais les deux sont négociées à **x8**. La génération lue au repos vaut 1, par
économie d'énergie. La topologie mesurée donne 18,7 Go/s vers la 5090 et 11,4
vers la 3080 Ti, soit 59 et 72 % du théorique. Ce point-là est mesuré et le
code le lit correctement ; les 1792 Go/s de bande passante mémoire, eux,
restent une valeur de plaque jamais confrontée.

**Deux mesures que le relevé ne trouve chez personne.** Le taux de succès d'un
cache d'experts pour 512 experts routés 10 — la littérature s'arrête à 64, 128
ou 256, où le rapport entre octets stockés et octets lus par jeton est de
quelques unités contre 45 chez nous. Et l'énergie par jeton sur des cartes
bridées : les mesures publiées sont sur H100, H200 ou une 4060 Ti à pleine
puissance. Le banc a déjà les colonnes W et j/kJ.

**Et une convergence à retenir.** `arXiv:2512.02189` mesure les cœurs
tensoriels de Blackwell à 96-99 % du pic et conclut que le goulot est la bande
passante et le lancement des noyaux ; `arXiv:2608.21240` mesure l'autre bout et
trouve 73 à 88 % du temps par couche dans le transfert des experts. Deux angles
opposés, une même réponse : ce n'est pas le calcul. C'est ce que la nuit du 6 au
7 avait montré autrement — le planificateur se trompait sur un débit et sur un
plancher, jamais sur un noyau.

## 8 septembre 2026, 06:20 — la courbe des couches exilées, en service

Cinq cases sur l'arbre figé `17f5566`, même prompt du banc, 200 jetons à
température zéro, serveur relancé entre les cases, deux références en
encadrement. Texte identique mot pour mot dans les cinq cases. Les deux
références se recoupent à 0,2 %.

| case | décodage, 199 jetons | débit | premier jeton | processeur du serveur |
|---|---|---|---|---|
| référence a, 18 couches calculées sur le processeur | 19 801 ms | 10,1 t/s | 6 273 ms | **183 s** |
| 18 couches exilées, transférées par le bus | 20 919 ms | 9,5 t/s | 4 835 ms | 25 s |
| 27 couches exilées | 26 585 ms | 7,5 t/s | 5 432 ms | 32 s |
| 36 couches exilées | 33 035 ms | 6,0 t/s | 10 515 ms | 42 s |
| référence b | 19 769 ms | 10,1 t/s | 7 346 ms | **182 s** |

Trois choses que la courbe établit, et une qu'elle réfute.

**La pente existe dès 27, elle est régulière : 3,2 ms par couche exilée et
par jeton** entre 18 et 27, 3,6 entre 27 et 36. Le chiffre à sec de la
veille, 3,1 ms sur une couche isolée, était donc le coût courant et non le
cas défavorable. L'hypothèse « plateau puis falaise » est réfutée.

**Les deux chemins coûtent la même chose par couche.** Le calcul sur
processeur revient à 2,9 ms par couche et par jeton, le transfert par le bus
à 3,2. C'est pourquoi les deux références et la case 18 rendent le même
débit : non parce que l'exil serait gratuit, mais parce que ses deux formes
se valent. À 18 couches, l'exil pèse **58 ms sur les 105 ms du jeton**, plus
de la moitié. Le reste, 30 couches résidentes et l'attention, vaut 47 ms,
soit un plafond de 21 jetons par seconde si rien n'était exilé.

**Le processeur, lui, n'est pas équivalent.** 183 secondes de temps
processeur contre 25 pour le même travail, sept fois plus, invisibles pour
le compteur d'énergie de la carte. C'est le point 8 du protocole d'énergie
sous sa forme la plus brutale : un moteur qui calcule sur le processeur paraît
sobre à NVML et ne l'est pas. Pour l'objectif d'efficacité énergétique, le
chemin par le bus est déjà le bon, à débit égal.

**Le transfert est borné par la latence, pas par le débit.** 18,4 Mio par
couche en 3,2 ms font 5,7 Go/s effectifs sur un bus mesuré à 18,7 : trente
copies séparées par couche, dix experts fois trois tenseurs, chacune payant
son lancement. Une seule copie contiguë par expert, ou par couche, est la
prochaine mesure — et c'est le terme constant que Manon a posé dans le modèle
de coût, dont l'ordonnée à l'origine est maintenant chiffrable.

Ce que la courbe ne dit pas : le gain de ne pas faire grossir le modèle à la
conversion. Les 18 couches exilées viennent des 4,5 bits par poids de NVFP4
contre 3,4 à la source ; à taille égale, aucune ne le serait, et le plafond
de 21 jetons par seconde deviendrait le débit.

## 8 septembre 2026, matin — v0.4.77 : fusions, et la fenêtre du bus

Fusion des branches de travail : granularité des transferts dérivée de
`n_experts_active` au lieu d'être supposée (0.4.76), copie contiguë des poids
exilés — un poids NVFP4 voyageait en trois tenseurs, 1 620 copies par jeton
sur 18 couches, désormais une par poids — et horodatages `[mesure]` du premier
et du dernier jeton de chaque requête sur la sortie du serveur, pour borner
les fenêtres d'instrument sans messagerie.

La fenêtre du bus pendant un décodage réel de 400 jetons, alignée à la
seconde sur les horodatages du client : **479 Mio par jeton, 4,9 Gio/s
cumulés, par salves** — la carte 0 alterne 6,9 Gio/s et zéro, la carte 1 est
muette dix secondes sur vingt et une — sur des liens mesurés à 18,7 et 11,4.
**18 % du lien utilisé : la borne n'est pas le débit.** 290 ko par copie si
le compte de 1 728 tient, dans la fourchette prévue. Réserve écrite : la
fenêtre n'a couvert que 56 % du décodage, la première moitié.

La copie contiguë ne gagne que 7 % à sec (2,88 ms contre 3,11 par couche) :
diviser les lancements par trois ne suffit pas, le coût est dans la
sérialisation des salves, pas dans le nombre de copies. Prochaine cible : le
recouvrement — les copies de la couche N pendant l'attention de N, et les
deux cartes en même temps.

## 8 septembre 2026 — seconde fenêtre : la copie contiguë mesurée en service, et la première énergie

Fenêtre entièrement couverte, segment découpé sur les lignes `[mesure]` du
serveur, 40 s, 400 jetons.

**Le débit passe de 9,5 à 9,99 jetons par seconde** : +5 % pour trois fois
moins de copies. La prédiction « borné par la latence des copies » tombe —
elle exigeait 15,4. La prédiction « rien ne bouge » tombe aussi : le bus
transporte 541 Mio par jeton contre 479 mesurés hier, mais la fenêtre d'hier
ne couvrait que 56 % du décodage, sa première moitié : les deux fenêtres ne se
comparent pas entre elles, seuls les débits en jetons se comparent. Ce qui
borne est la **sérialisation de la chaîne** — routage, copie, calcul, expert
après expert — et le profil le montre mieux que toute moyenne : la 3080 Ti a
le bus à moins de 50 Mo/s treize secondes sur quarante pendant que le décodage
attend, avec des plages entières à 7 Mo/s. **Cible unique : recouvrir.**

**Première mesure d'énergie du protocole**, bornée par les horodatages
journalisés, ligne de base de même durée immédiatement après :

| | décodage | repos après |
|---|---|---|
| RTX 5090 | 76,2 W, 2 970 J, 38 °C, SM 2 745-2 842 MHz | 27,2 W |
| RTX 3080 Ti | 102,5 W, 3 997 J, 34-35 °C, SM 1 755-2 010 MHz | 52,8 W |

400 jetons : 6 967 J bruts, 3 847 nets, soit **17,4 J/jeton brut, 9,6 net** —
57 et 104 jetons par kilojoule. Trois faits à porter au comparatif : aucune
des deux cartes n'approche son plafond (76 W sur 400, 102 sur 275), donc ce
régime n'est pas celui des cartes contraintes ; la 3080 Ti consomme plus que
la 5090 en transférant quatre fois moins ; et le repos pèse 45 % de la
fenêtre, ce qui rend le choix brut/net décisif et oblige à l'annoncer.

## 8 septembre 2026, matin — v0.4.79 : le chemin direct mesuré en service, +31 %

Série de trois cases sur `652741c`, arbre propre, mêmes conditions que toute
la nuit, texte identique mot pour mot dans les trois :

| case | décodage, 199 jetons | débit | processeur du serveur |
|---|---|---|---|
| chemin direct | 14 868 ms | **13,4 t/s** | 19 s |
| chemin par masques (échappement) | 19 380 ms | 10,3 t/s | 24 s |
| chemin direct, contre-mesure | 14 689 ms | **13,5 t/s** | 19 s |

Les deux cases directes se recoupent à 0,7 %. Sur ce modèle, le pire du parc,
le débit passe de 10,3 (comparatif du 6) à 13,5 jetons par seconde :
**ensemble, +31 %**, un passage par case — l'écart est grand devant le
recoupement de 0,7 %, mais c'est un intervalle qu'une série à trois passages
donnera, pas un point. La décomposition — copie contiguë environ +5, chemin
direct environ +26, celui-ci remplaçant à un jeton les masques par expert et
leurs trente et une synchronisations hôte par couche — est **plausible, pas
mesurée** : les deux gains viennent de fenêtres différentes et leur
additivité n'a pas de case témoin « contiguë sans direct ».

Fusionnés dans la même version : l'énergie NVML au banc (branche oceane,
`7e386eb`) — lecture du compteur par ctypes, colonnes J, J_net, W_repos,
j_kJ_net, bridages et invalidations calculées, l'ancienne classe Watt retirée
(elle lançait un nvidia-smi toutes les 200 ms sur la machine qu'elle mesurait
et ne voyait que la carte 0) — et la trace de routage rejouable (branche
manon, `3865418`) : `ACVRAM_TRACE_ROUTAGE=/chemin`, 22 ns par appel éteinte,
rejeu par `taux_de_succes()` sous deux politiques. Aucune trace encore prise
sur un vrai modèle. 153 tests.

Reste ouvert, dit d'avance : le chemin par masques garde le lot > 1 et le
prefill ; la décision « trois passages au banc » (dispersion contre temps de
série) ; et la prise de la première trace réelle.

## 8 septembre 2026, 07:55 — premier comparatif énergétique instrumenté

Trois passages par case, énergie du même passage que le débit publié, ligne de
base après chaque case, invalidations calculées par le banc lui-même
(`docs/comparatif-energie-20260908.tsv`).

| modèle | moteur | t/s | j/kJ net | W | dispersion |
|---|---|---|---|---|---|
| Huihui-35B-A3B-Opus | **acvram** | **293,9** | **2 073** | 184 | 21,3 % ⚠ |
| Huihui-35B-A3B-Opus | llamacpp | 208,5 | 1 218 | 171 | 5,2 % ⚠ |
| Huihui-35B-A3B-abl | **acvram** | 159,2 | 946 | 130 | 26,0 % ⚠ |
| Huihui-35B-A3B-abl | llamacpp | **208,4** | **1 161** | 174 | 5,5 % ⚠ |
| Coder-Next Q3_K_S | acvram | 13,4 | 428 | 184 | 1,3 % |
| Coder-Next Q3_K_S | **llamacpp** | **154,5** | **867** | 271 | 1,0 % |
| Qwen3.8-27B Q6_K | **acvram** | **61,7** | **276** | 270 | 5,2 % ⚠🔌 |
| Qwen3.8-27B Q6_K | llamacpp | 42,3 | 154 | 348 | 12,8 % ⚠🔌 |
| Qwen3.8-27B exl3 | **acvram** | **49,4** | **219** | 271 | 4,9 % 🔌 |
| Qwen3.8-27B exl3 | tabby | 46,7 | 179 | 311 | 0,3 % 🔌 |

Ce que la contradiction a laissé debout — critère : l'écart doit valoir au
moins trois fois la dispersion maximale du pairage :

* **Cinq appariements, pas quatre** — le premier décompte omettait Coder-Next,
  c'est-à-dire la pire défaite, et l'omission est consignée ici parce qu'elle
  est le genre d'erreur qui ne pardonne pas en publication.
* **Établis** : Qwen3.8-27B Q6, acvram +46 % de débit et +79 % d'efficacité
  (rapport écart/dispersion 3,6) ; exl3 contre Tabby, +5,8 % (rapport 19) ;
  et la défaite Coder-Next, −91 % (rapport 70).
* **Non conclusifs** : les deux Huihui, dispersions de 21,3 et 26,0 % pour des
  écarts de +41 et −24 % — rapports 1,9 et 0,9, aucun chiffre publiable quelle
  que soit la statistique. La médiane et l'étendue remplaceront le meilleur
  des trois, qui favorise le moteur le plus bruyant.
* **Le bridage en puissance existe bel et bien** : sur le dense 27B, les cas
  marqués 🔌 ont touché le plafond logiciel — llama.cpp à 348 W. L'hypothèse
  d'hier « jamais bridées en décodage » ne valait que pour le MoE exilé ; le
  comparatif devra publier le régime par cas, pas par machine.
* Les deux cases Huihui d'acvram dispersent à 21 et 26 %, et l'explication
  facile — la spéculation dépendrait du texte — ne tient pas : à température
  zéro le texte sort identique, donc les n-grammes aussi. Ou bien le chemin
  n'est pas déterministe sur ces modèles, défaut à part entière, ou bien la
  variation vient d'ailleurs, froid, cache, ordonnancement. Le contrôle le
  moins cher : une empreinte du texte par passage dans le TSV, qui dira
  lequel sans relancer de série.
* Coder-Next, onze fois plus lent mais seulement deux fois moins efficace :
  la carte consomme peu pendant qu'elle est lente. L'énergie perdue est dans
  l'attente, pas dans le travail — la chaîne sérialisée vue au profil du bus,
  chiffrée cette fois en joules. Argument de plus pour le recouvrement.
* Coder-Next reste perdu 11 fois : c'est le modèle 3 bits, la garde de
  conversion refuse désormais de le faire grossir, et le format ~3,5 bits
  (spécification de la nuit, quantiles 3 bits à 3,25 bpw, table symétrique)
  est la voie. Le SNR de la spécification bat les entiers 3,5 bits à taille
  moindre ; perplexité et choix du bloc restent à mesurer.

## 8 septembre 2026 — v0.4.86-88 : le format Q3N de bout en bout, à trois

Chronologie d'une après-midi : spécification (quantiles 3 bits, table
symétrique, échelle FP8 par bloc, 3,25 bits/poids — et la première version de
la table, asymétrique à la NF4, écartée parce qu'un bloc de 32 y battait un
bloc de 16, ce qui est impossible) ; implémentation Python et intégration aux
formats, à la garde de grossissement et à la bascule automatique des sources
3 bits ; noyau CUDA `q3n_gemv` ; plan de test adversarial écrit avant de voir
le noyau, que le noyau traverse entier ; contrat public `q3n_gemm` **sans
plancher de capacité** — Q3N n'a besoin d'aucun cœur tensoriel, la 3080 Ti y
a droit, et c'est écrit pour que personne ne l'ajoute par mimétisme du miroir
FP4. 227 tests.

**Précision d'honnêteté sur l'équivalence du noyau.** « Au bit près » était
trop dit : l'écart mesuré contre la référence par déquantification, en
float32, va de 1,2e-7 à 4,8e-7 absolu sur des amplitudes de 0,4 à 3,7 — soit
l'ordre de quelques ulps, ce qu'on attend de deux ordres d'accumulation
différents, pas une égalité binaire. La validation qui reste due, sur les
poids réels une fois la conversion finie : référence en float64, formes
réelles, écart maximal et quadratique moyen publiés, compte des valeurs
saturées. Une égalité au bit près sur formes réelles serait un signal à
inspecter, pas un succès.

**Deux validations distinctes, une seule faite.** L'écart en 1e-7 valide
l'arithmétique du GEMV — il calcule bien le produit demandé — et ne dit rien
de ce que le format coûte au modèle : la même référence part des mêmes poids
déquantifiés. La validation de **format** — q3n contre poids d'origine,
couche par couche, erreur relative attendue de l'ordre du pour cent (15 dB de
SNR ≈ 18 % d'erreur quadratique relative sur les distributions de la
spécification), distribution des écarts et valeurs aberrantes — reste due sur
les poids réels, et se confirme à la perplexité. Un q3n rapide et exact
arithmétiquement peut rendre un modèle plus bête sans que le comparatif de
débit le voie jamais.

**Ce qui décidera du bloc 16 contre 32** : la lourdeur de queue des vrais
poids d'experts. Sur lognormale, l'écart entre blocs passe de 0,85 dB (sigma
nul) à 2,91 dB (sigma 1,5) — plus la queue est lourde, plus le bloc de 16
gagne et moins le quart de bit se défend. Mesure au harnais processeur sur le
dossier converti, avant toute perplexité.

## 8 septembre 2026 — le bloc 32 confirmé sur poids réels, et la question de la source

Mesure sur 60 tenseurs d'experts réels de Coder-Next (`outils/
mesurer-queue-experts.py`, processeur seul) : la queue est **quasi
gaussienne**, sigma inter-blocs 0,03, et l'écart bloc 16 contre 32 vaut
1,36 dB — pas les 2,5 à 3 qui auraient justifié le quart de bit. **Le bloc 32
reste**, pas de reconversion. Deux pièges évités en chemin, consignés dans le
commit de mesure : un estimateur de sigma qui lisait la grille de la source
au lieu de la queue du modèle, et une table à zéro exact qui gagnait 3,8 dB —
par alignement sur la grille source uniquement, elle perd 1,3 dB sur du
continu ; la table ne change pas.

**La question qui pèse plus que la taille de bloc : la source.** Il n'existe
aucun bf16 de ce modèle sur disque — seulement le GGUF Q3_K_S et notre NVFP4.
Toute conversion q3n est donc une **seconde quantification** d'une grille
déjà 3 ou 4 bits, et la qualité absolue du format ne peut pas se mesurer
ainsi. La perplexité du dossier en cours en donnera l'effet net contre
llama.cpp qui, lui, sert la source telle quelle ; si elle décroche, la vraie
issue est de retélécharger la source bf16 (~160 Go) et de convertir depuis
elle — décision de ressources qui appartient à l'utilisateur.

Et un défaut d'exécution attrapé en relecture, la classe que les tests
d'équivalence ne voient pas : `q3n_gemv_cuda` lisait l'échelle globale par
`.item()` sur un tenseur CUDA à chaque appel — le piège documenté de NVFP4,
62 % du temps de décodage au profil d'époque, et un noyau incapturable en
graphe CUDA. Corrigé par la même mémorisation côté hôte ; deux tests
verrouillent la classe entière, dont un qui capture réellement un graphe.

## 8 septembre 2026 — v0.4.93-94 : la panne URT disséquée, deux dossiers restants

Le modèle tout-q3n produisait « URTURTURT » (perplexité 5,6 M, pire que
l'uniforme). La chasse, dans l'ordre des innocentements : noyau et
répartiteur (équivalence sur tenseurs réels), poids sur disque (cosinus
0,95–0,98 contre le NVFP4 sain, classe par classe), noyau encore à
151 936 lignes (argmax identique à la référence sur le lm_head réel),
transport des experts (aller-retour bit à bit, audit du 8/09 sur la branche
manon). L'outil qui a tranché : la trace de cosinus des activations contre
le modèle NVFP4 sain, même invite, couche par couche.

* v0.4.93 : plancher int8 sur `linear_attn.*` — insuffisant, URT encore.
* La trace montre alors 0,99 sur les couches GDN et un effondrement à 0,57
  dès la couche 3, première attention pleine (q/k/v/o en q3n). v0.4.94
  étend le plancher à `self_attn.*` et `lm_head`. Coût : +229 tenseurs
  int8, dossier stable à 32 Go (contre 296,8 équivalent bf16).
* Après reconversion : prefill sain («  Paris. »), plus de falaise — mais
  une décroissance PROGRESSIVE du cosinus (0,97 à la couche 7, 0,48 à la
  13, bruit dès la 15) : un bruit de format qui se compose, réparti sur les
  experts q3n.

Deux dossiers ouverts, instruits en parallèle :

* **(A) divergence des chemins de décodage** — les deux chemins dégénèrent
  différemment (« loi,loi » direct t==1, « URT » masqué) alors que le
  prefill est sain et que la copie des experts est innocentée. Défaut de
  code. L'audit du pool a par ailleurs trouvé un débordement silencieux
  possible (curseur modulo sans état par emplacement, repli top_k
  incohérent 2 vs 8) : garde-fou et correctif fusionnés (9cf1a15), sans
  preuve que ce fût LE bug.
* **(B) bruit de prefill** — perplexité acvram 858 contre 137 pour le
  NVFP4 sain sur le même harnais (harnais lui-même douteux : corpus
  interne de 282 jetons, fenêtres de 512 qui coupent l'état de la
  récurrence GDN — les deux chiffres ne valent que par leur écart,
  ~1,8 nat/jeton). Les SNR du manifeste sont un PLANCHER (90 % des
  73 872 tenseurs dans 0,10 dB autour de 13,33) : ils mesurent le couple
  de grilles Q3_K_S→q3n, pas les poids. 67 des 68 tenseurs < 10 dB sont
  des down_proj d'experts des couches hautes. Deux mécanismes candidats en
  cours de mesure float64 : échelle de bloc FP8 e4m3 subnormale ou nulle
  sur tenseur à aberrant global ; table sans zéro qui reconstruit les
  blocs creux à ±0,1025 × amax. Correctifs esquissés selon le verdict :
  clamp d'échelle (une ligne) ou table/bloc à rediscuter.

Rappel de méthode qui a coûté une reconversion : la première hypothèse
(GDN seule) était plausible, confirmée par un indice réel (l'ancienne
conversion), et fausse quand même — c'est la trace par couche, pas
l'indice, qui a montré la vraie frontière.

## 8 septembre 2026 — dossier B instruit : la table sans zéro crée de l'énergie sur les blocs creux

Mesure float64 sur les vingt pires tenseurs du manifeste (session de mesure,
8/09, processeur seul). Les deux mécanismes candidats départagés :

* **Échelle FP8 subnormale ou nulle : réfuté deux fois.** Les échelles à
  zéro (jusqu'à 44,4 % des blocs) recopient des blocs source déjà vides —
  énergie perdue 0,00 % partout, SNR inchangé à 0,01 dB en les ignorant.
  Les subnormaux portent au plus 8,8 % de l'énergie, SNR propre pas pire
  que les blocs normaux. Un clamp ne gagnerait rien de mesurable.
* **Table sans zéro : confirmé, c'est tout l'écart.** Le plus petit niveau
  (0,1025 × échelle) reconstruit chaque poids nul à ±0,1025 × amax du
  bloc : sur un bloc creux c'est de l'énergie créée, qui se compose couche
  après couche — la décroissance progressive du cosinus depuis la
  couche 7. Vingt pires : 85–95 % de creux ; tenseurs à 13,3 dB : 21 %.
  Dans un même tenseur sain : blocs denses 13,49 dB, blocs creux 8,60.
* **Fait inattendu : les tenseurs cassés sont des experts quasi morts.**
  97,37 % de zéros exacts dans la source pour le pire (couche 46), 90–95 %
  pour les suivants, contre 20,73 % pour un expert sain. Les 67 down_proj
  sous 10 dB sont des experts vides bruités — propriété du modèle
  exploitable un jour par le planificateur.
* **Remède mesuré, pas prédit** : table à niveau zéro, même bloc 32, même
  échelle, 3,25 bpw inchangés — +9,6 à +16,7 dB sur les cassés, et bloc 32
  à zéro bat bloc 16 sans zéro (3,50 bpw) sur toute la colonne. Contrôles :
  le gain tient quand on neutralise les zéros exacts de la source
  (+16,0 au lieu de +16,7) ; sur du synthétique dense il PERD ~0,8 dB
  (gaussienne, laplace) et gagne fort sur queue lourde (+8,4 student t=2,
  +12,4 avec aberrants). Sur les tenseurs sains réels, +0,95 dB dont
  +0,41 seulement une fois la source neutralisée.
* **Décision qui en découle — et sa limite** : pas de changement global de
  table ; une table PAR TENSEUR choisie sur le creux mesuré à la
  conversion (deux tables, un bit d'en-tête, zéro coût de débit). La
  variante symétrique du niveau zéro reste à mesurer avant de graver — la
  table gagnante actuelle est asymétrique et la spécification du 8/09 a
  banni l'asymétrie pour une raison qui ne s'applique pas ici (±1,0
  présent des deux côtés), à confirmer par la mesure, pas par l'argument.

Ordre maintenu : le dossier A (divergence des chemins de décodage, en
cours) passe avant tout changement de table — tant qu'il est ouvert,
aucun SNR ne prédit une perplexité. Étalon de perplexité en cours :
llama.cpp sur le GGUF Q3_K_S source, wiki.test.raw, contexte 512.

## 8 septembre 2026 — le dossier A se referme, l'étalon de perplexité est posé

* **La « divergence des chemins de décodage » n'était pas un bug** : mêmes
  poids, même routage, seul l'ordre d'accumulation bf16 différait —
  5,7e-3 d'écart relatif par couche, 0 exact en fp32 (mesure du 8/09,
  branche manon). Sur un modèle sain cet écart ne change pas le mot ; à
  perplexité dégradée, l'argmax n'est plus porté par le signal et deux
  ordres de sommation donnent deux charabias différents — « loi,loi » et
  « URT » étaient deux tirages du même bruit. Les deux chemins accumulent
  désormais en float32 (écart à la référence 6,58e-3 → 5,13e-3). Le seul
  dossier restant est le format : la table sans zéro sur les blocs creux.
* **Le harnais de perplexité mentait d'un facteur quinze.** llama.cpp sur
  le GGUF Q3_K_S source : **9,1831 ± 0,073** (wiki.test.raw, sha256
  173c87a5…, 1 290 590 octets ; 584 fenêtres de 512 disjointes, 256
  positions notées par fenêtre, binaire e34f042). Le « 137 » d'acvram sur
  le NVFP4 sain venait du corpus interne de 283 jetons noté depuis la
  position zéro — les premières positions coûtent une dizaine de nats à
  n'importe quel modèle. Correctif : --min-context (fusionné), éval
  comparable = wiki.test.raw, fenêtre 512, stride 512, min_context 256,
  après vérification du compte de jetons des deux tokeniseurs.
* Au passage : CUDA_VISIBLE_DEVICES réparé (detect_rig interrogeait
  nvidia-smi qui l'ignore ; renumérotation à la manière de CUDA), la
  machine se partage désormais carte par carte ; et sept échecs de tests
  graphes/MoE observés pendant qu'une mesure occupait la 5090 étaient de
  la contention GPU, pas une régression — leçon : la suite complète ne se
  juge que carte libre.
* Mesures en attente : perplexité q3n comparable (à l'étalon 9,18, en
  séquentiel après le contrôle NVFP4), Lloyd-Max stratifié 288+288
  tenseurs disjoints (table unique ou par tenseur).

## 8 septembre 2026, midi — l'étalonnage inter-instruments attrape un bug de harnais

Protocole (session de mesure) : même corpus wiki.test.raw (sha256
173c87a5…), mêmes fenêtres de 512 disjointes, positions notées à partir de
256 jetons de contexte, llama.cpp d'un côté, acvram eval de l'autre, sur
deux témoins convertis en int8 partout (--autoriser-grossissement — la
garde anti-grossissement avait silencieusement basculé le premier témoin
en q3n : dossier nommé « int8 », MLP à 14,9 dB ; attrapé au manifeste,
règle : diff des manifestes TENSEUR PAR TENSEUR avant d'attribuer un écart
à un format).

* phi-4 (dense) : acvram 6,911 contre llama.cpp 6,5988 ± 0,041 — écart de
  +4,7 %, soit sept fois et demie l'incertitude d'échantillonnage (±0,63 %)
  et plus du double du seuil de 2 % : il est réel et NON EXPLIQUÉ. Une part
  inconnue revient à la requantification Q4_K_M→int8, le reste à
  l'instrument ; une mesure de séparation est en cours (Q8_0 par
  llama-quantize, même instrument des deux côtés — l'attente écrite
  d'avance : un int8 à ~44 dB coûte d'ordinaire bien moins de 1 % de
  perplexité, il resterait alors 3-4 % de biais d'instrument à documenter,
  petit devant le facteur 4,1 des hybrides mais dangereux le jour où il
  s'additionnera à un gain de format de 5 %).
* Qwen3.6-12B (hybride GDN) : acvram 129,185 contre 31,6919 ± 0,265 —
  facteur 4,1. Or les tenseurs sont à ~44 dB (vérifiés) et le chemin de
  service int8 est innocenté sur tenseurs réels (écart 1,7e-3 contre un
  produit dense, l'arrondi bf16). Le défaut est lié à l'architecture
  HYBRIDE, pas au format ni au noyau.
* Piste désignée (en cours de confirmation) : le magasin d'états
  récurrents des couches linear_attn n'est pas réinitialisé entre les
  fenêtres d'évaluation — le cache KV repart à zéro à chaque fenêtre, pas
  l'état GDN ; l'état de la fenêtre N fuit dans la N+1. Test : après
  correctif, l'éval 12B doit tomber de 129 vers ~32.
* Conséquence : toutes les perplexités mesurées sur des hybrides par
  acvram eval (858 q3n, 137 NVFP4 de Coder-Next) sont invalidées jusqu'au
  correctif. L'écart q3n/NVFP4 devra être remesuré après : correctif
  d'état + égalisation du filet (reconversion q3n avec snr_floor 25 —
  l'ancienne avait snr_floor 0, AUCUNE promotion, 143 shared_expert sans
  filet sur le chemin de 100 % des jetons).

Leçon de méthode, payée trois fois aujourd'hui : ce qui n'est pas imprimé
à côté du chiffre finit par être supposé faux — le cadrage, le snr_floor,
les formats réels. acvram eval imprimera les formats du modèle évalué.

## 8 septembre 2026, 12:10 — verdict : la récurrence GDN d'acvram diverge avec la longueur

Balayage à trois fenêtres, deux modèles int8 partout, mêmes corpus/cadrage
des deux côtés (étalons llama.cpp de la session de mesure, cibles écrites
d'avance) :

  fenêtre   phi-4 dense (acvram/llama)     12B hybride (acvram/llama)
    128      9,592 / 9,1320  = +5,0 %      113,42 / 52,577 = ×2,16
    512      6,911 / 6,5988  = +4,7 %      129,19 / 31,692 = ×4,08
   2048      5,984 / 5,8406  = +2,45 %     221,91 / 24,080 = ×9,21

* Le dense porte un biais de forward de +2,5 à +5 % (les bornes de
  notation, fenêtres et agrégation sont IDENTIQUES à llama.cpp, vérifié
  dans perplexity.cpp e34f042 — audit de la nuit ; le +4,7 % n'est pas de
  la comptabilité). À séparer du coût int8 par un phi-4 bf16 pur (en file).
* L'hybride DIVERGE : l'écart double à chaque quadruplement de fenêtre, et
  la perplexité acvram EMPIRE avec le contexte (113→129→222) là où le
  modèle réel s'améliore (53→32→24). Le défaut est cumulatif au fil des
  pas de la récurrence linear_attn — pas le format (int8 44 dB), pas le
  chemin de service (1,7e-3), pas le magasin d'états entre fenêtres
  (réinitialisé), pas la cohérence bloc/pas-à-pas (4,2e-4) : la DYNAMIQUE
  elle-même (décroissance, portes, convolution) s'écarte de la référence
  et l'erreur se compose.
* Conséquence maintenue : toutes les perplexités acvram sur hybrides sont
  invalidées (858 q3n, 137 NVFP4, 7,233 nemo). Et l'URT/13,5 t/s de
  Coder-Next tournait sur cette même récurrence : la correction peut
  changer aussi la qualité de génération, pas seulement l'éval.

## 8 septembre 2026, 12:30 — le témoin hybride était une architecture servie faux en silence

Le « défaut hybride » du balayage n'était pas dans le harnais ni dans la
récurrence : le témoin Qwen3.6-12B est un **qwen35**, dont le GGUF déclare
`rope.dimension_sections = [11, 11, 10, 0]` et que llama.cpp sert en
LLAMA_ROPE_TYPE_**IMROPE** (M-RoPE entrelacé) — acvram lui appliquait un
NEOX standard. RECTIFICATION une heure plus tard, à la lecture de
ggml (ops.cpp:5639) : l'entrelacement IMROPE choisit l'AXE de position
par paire mais les fréquences avancent uniformément — en texte pur, si
les quatre flux de positions sont égaux, IMROPE ≡ NEOX et ce diagnostic
ne tient QUE si llama.cpp alimente des positions non égales pour qwen35
en texte (vérification en cours sur le graphe, avec un candidat de
repli : attention.key_length = 256 là où hidden/têtes donnerait 213 —
le head_dim ne se déduit pas, il se lit). Les signatures du balayage
(×2,16 → ×4,08 → ×9,21 ; GDN exacte jusqu'à t=2048 ; dense propre)
restent des faits ; leur cause au sein de l'attention pleine du qwen35
reste À PROUVER, l'outil de dump par couche est prêt si l'hypothèse
RoPE tombe. Vérifié dans les DEUX GGUF (session
de mesure) : Coder-Next (`qwen3next`) ne porte AUCUNE section, freq_base
5e6 et rotary_dim 64 correctement lus — son RoPE NEOX est le bon
traitement sur les trois axes qui peuvent mentir (type, theta, dims).

* Ce qui se referme : le harnais est utilisable sur qwen3next ; le 858
  q3n / 137 NVFP4 de Coder-Next ne sont plus suspects d'un défaut de
  moteur — ils restent incomparables entre eux par le FILET asymétrique
  (snr_floor 0 contre 25). Reconversion snr_floor 25 + éval 584 fenêtres
  en cours ; l'écart à l'étalon 9,1831 sera ENFIN le coût du format.
* Ce qui s'ouvre : implémenter l'IMROPE (les qwen35 du parc sont
  aujourd'hui servis faux en silence) avec validation par juge extérieur ;
  et le garde-fou immédiat — une architecture dont le type de RoPE n'est
  pas reconnu se REFUSE à la conversion et au chargement. La leçon de la
  journée, formulée par la session de mesure : le défaut n'était pas le
  calcul mais le silence du calcul sur ce qu'il ne savait pas faire.
* Biais d'instrument résiduel sur dense : +2,5 à +5 %, décroissant avec la
  fenêtre, non expliqué (décomposition noyaux/précision infaisable en OOM
  sur modèle entier) — borné, documenté, à ne jamais soustraire d'un
  chiffre mesuré à une autre fenêtre.

## 8 septembre 2026, 13:10 — fausse cause racine, rétractée en vingt minutes

Une alerte « le facteur de décroissance GDN est transformé deux fois »
(le GGUF stocke −exp(A_log), gdn.py réappliquerait −exp) a été démontrée
FAUSSE par son autrice avant qu'aucun code ne change : le drapeau
`gdn_a_log_negexp` est lu, branché (loader.py:595-598 inverse la valeur
stockée quand il est vrai, aller-retour vérifié au bit près) et True dans
les config des modèles convertis — c'était le montage du test isolé qui
sautait l'inversion du chargeur. Vérifié indépendamment ici avant de
rétracter. Ce qui reste VRAI de cette séquence : la couche 0 réelle
d'acvram est à 1,1 % de llama.cpp, PLATE en position (le coût attendu de
deux quantifications différentes) — la couche linéaire est saine sur
toute la ligne ; un bruit de poids ne croît pas en position (mesuré,
0,95×) ; et le cumul par blocs égale le pas-à-pas (6,9e-05). Le défaut du
12B témoin reste SANS cause identifiée (les cosinus par couche d'un run
réel se dégradent progressivement entre couches GDN quand la ligne de
base Q5↔Q8 reste plate — fait mesuré, inexpliqué). Le dossier Coder-Next,
lui, n'a jamais dépendu de cette alerte : son éval « avant » (q3n sans
filet, cadrage propre) est relancée telle quelle.

Leçon, la troisième du même jour pour les montages : un montage à moitié
réparé donne un chiffre PLAUSIBLE — la bonne forme, le bon ordre de
grandeur, la signature attendue — et le plausible est ce qui échappe au
contrôle. Chaque montage s'éprouve après CHAQUE réparation, pas
seulement à la première.

## 8 septembre 2026, 15:15 — BILAN de la série Q3N : le format est innocenté, le moteur est désigné

Quatre chiffres, même corpus (wiki.test.raw, sha256 173c87a5…), même
cadrage (584 fenêtres de 512 disjointes, 148 920 positions notées à
min_context 256), même harnais :

  étalon llama.cpp, GGUF Q3_K_S source     9,1831 ± 0,073
  q3n v0.4.94, SANS filet (snr_floor 0)    1005,838
  q3n v0.4.95, filet réel (144 promus)     209,036
  NVFP4 (snr_floor 25, ancien dossier)     193,621

* **Le filet vaut 1,57 nats** (×4,8) : les 144 shared_expert promus int8
  étaient le premier poste. PROMOTE ignorait q3n en silence (corrigé
  v0.4.95, test au registre : tout format quantifié a une issue).
* **Le format ne coûte que 0,077 nats** (+8 % relatif) : q3n à 3,25 bits
  contre NVFP4 à 4,5, à filet égal, mêmes conditions — 25 % d'octets en
  moins pour 8 % de perplexité relative en plus. C'est le chiffre que la
  journée cherchait, et il est favorable au format.
* **L'excès dominant (3,05 nats) est COMMUN aux deux formats** : le
  « fait sans cause » des hybrides est partagé par qwen3next, mesuré sur
  le dossier — au moteur ou au harnais sur cette architecture, pas au
  format. Paradoxe directeur pour la suite : le NVFP4 à ppl 193 au
  harnais génère du texte cohérent en usage réel — le défaut penche vers
  le chemin d'évaluation/prefill des hybrides, pas la génération.
* Prédictions écrites d'avance et leurs verdicts (session OnePlus) :
  « après » prédit 25-70 centre 40, mesuré 209 — modèle du bruit blanc
  des routés réfuté par son autrice : l'erreur de la table sans zéro est
  un BIAIS CORRÉLÉ entre experts (énergie créée, même géométrie partout),
  qui s'additionne au lieu de se moyenner ; sa prédiction dérivée pour la
  table à niveau zéro (30-90) est bornée par le mode commun (~179 tant
  qu'il n'est pas expliqué). « Ça ne descendra pas à 9,18 » : tenu.

Suite, dans l'ordre convenu à trois : (1) expliquer le mode commun des
hybrides (dump par couche sur Coder-Next, chemin d'éval vs génération) ;
(2) table à niveau zéro par manifeste (mécanique v0.4.95 prête et
scellée, valeurs d'Océane en attente de ce préalable) ; (3) le programme
performance sur le créneau du modèle exilé — distribution du routage
(l'outil de trace n'a jamais pris de trace réelle), banc comparatif
reproductible (la dispersion 21-26 % est un préalable), cache d'experts
(promis par tiering.py, absent du moteur, VRAM réservée perdue),
recouvrement transfert/calcul (bus à 5,7/18,7 Go/s, cartes muettes une
seconde sur deux), transfert en format compact, et l'axe jetons/kJ que
personne ne publie (104 j/kJ net déjà mesurés au compteur, cartes
bridées 400/275 W).

### Addendum du soir — la piste n°1 du mode commun est nommée et testable

La fourchette « 30-90 après table » est RETIRÉE par son autrice : la table
ne peut agir que sur les 0,077 nats qui séparent les formats, pas sur les
3,05 communs. Et le paradoxe (ppl 193 au harnais, texte cohérent en
génération) est requalifié en MESURE : deux faits incompatibles sur le
même modèle décrivent deux CHEMINS — la génération ne consomme que la
dernière position de chaque passe, l'évaluation les consomme toutes
(`logits_positions=all_token_indices`). Test décisif en cours (session
OnePlus) : logits d'un prefill unique multi-positions (A) contre décodage
pas-à-pas par le chemin réel de génération (B), 32 jetons, témoin dense
phi-4 DANS le montage (A et B doivent y coïncider, sinon c'est le montage
qui diverge). Si A[-1]=B[-1] mais A[i]≠B[i] : le chemin d'évaluation
multi-positions des hybrides est le défaut, toutes les perplexités
hybrides sont à refaire après correctif, et le vrai q3n est peut-être
bien sous 209.

### Addendum, 16 h — l'évaluation est innocentée, le service est touché

Perplexité par le chemin de GÉNÉRATION (décodage pas à pas, cache KV et
magasin d'états persistants, jetons du corpus forcés) contre le chemin
d'ÉVALUATION (préremplissage multi-positions), MÊMES 512 jetons, même
min_context 256, une seule exécution, q3n avec filet :

  A préremplissage  108,159   B décodage pas à pas  121,666   (255 positions)

Les deux chemins donnent le même ORDRE, ce qui suffit à écarter
l'évaluation comme cause du facteur vingt. Mais **B est 12,5 % pire que
A, et ce n'est pas un arrondi** : les écarts de logits entre chemins
valent 1,4e-02, l'incertitude d'échantillonnage ≈6 % par chiffre. Le
service reste donc dégradé PAR RAPPORT à l'évaluation, d'un montant
mesuré et sans cause — fait supplémentaire, à ne pas ranger sous
« les deux concordent ». **Le paradoxe « perplexité 193 mais
texte cohérent » n'est donc pas un défaut d'instrument : le service
calcule bien ce que l'évaluation mesure.** La priorité du projet est
confirmée : la dégradation est réelle en service.

Contrôle de représentativité, exigé et payant : A vaut **108 sur ce
sous-ensemble contre 209 sur les 584 fenêtres** — le début du corpus
n'est pas représentatif (facteur ~2), tout essai court reste interne à
lui-même et ne se relie pas au chiffre global.

Suspects éliminés ce soir, tous par la mesure : prefill multi-positions
(causalement correct, 0,000e+00), divergence éval/génération (concordance
aux arrondis, témoin dense DANS le montage — et l'hybride diverge MOINS
que le dense), amplification récurrente de l'erreur de poids (composition
en √N des deux côtés, rapport constant 2,3), hypersensibilité
architecturale comme explication (mesurée à 6,1× sur perturbation
modérée, mais 26× trop petite ET saturante — les hybrides SE
requantifient, +11,9 % au pire chez llama.cpp : la demande de 160 Go bf16
à l'utilisateur est épargnée).

Sous-produit réutilisable : **une couche récurrente est 2,3 fois plus
sensible à une perturbation de poids qu'un perceptron de même largeur** —
critère de placement pour le planificateur, et justification rétrospective
du plancher int8 de la v0.4.94.

Question restante, unique et nette : pourquoi acvram dégrade-t-il d'un
ordre de grandeur par rapport à llama.cpp sur le même modèle, en service
comme en évaluation, alors que chaque composant testé isolément est sain.
Voie retenue : comparer les LOGITS (toutes nos comparaisons opposent
acvram à acvram, sauf la perplexité qui est l'agrégat final).

### Addendum, 18 h — le composant est nommé : les couches d'attention linéaire

Deux mesures indépendantes convergent et ferment la chasse.

**Par couche** (session OnePlus), chaque couche alimentée avec l'entrée
EXACTE de llama.cpp — donc sans propagation d'erreur — sur les neuf
premières couches du 12B :

  attention pleine (couches 3, 7)      1,0 à 1,3 %   cos 1,00048
  attention linéaire (0-2, 4-6, 8)     10,9 à 40,2 % cos 0,918 à 0,996

Un facteur 10 à 40 entre les deux familles, mêmes poids, même passe, même
perceptron à experts — ce qui innocente le MoE, l'attention pleine, le
RoPE et le harnais d'un seul coup, et explique la constance du facteur
global : la proportion de couches linéaires est elle-même constante.

**Par position** (ici), NLL d'acvram contre celles de llama.cpp sur les
mêmes 255 positions notées, ventilées par difficulté de la référence :

  quartile de difficulté   référence   acvram   écart
    1 (trivial)              0,001      1,662   +1,66
    2                        0,085      3,197   +3,11
    3                        0,978      4,333   +3,36
    4 (difficile)            5,410      8,479   +3,07

Le fait décisif est le premier quartile : **là où la référence est
certaine à 99,9 %, acvram tombe à 19 %**. L'écart est ensuite à peu près
constant (~3 nats) — signature d'un bruit ajouté aux logits, pas d'une
erreur sélective. Corrélation position par position 0,59 : nos positions
difficiles ne sont qu'à moitié les siennes.

Deux lectures antérieures sont donc RENVERSÉES, et il faut le dire :
la queue d'acvram (29,6 % du coût sur 10 % des positions) est plus
LÉGÈRE que celle du moteur sain (48,4 %) — ce n'était pas un symptôme
mais l'inverse ; et le top-1 sain vaut 63,2 %, pas les 35-40 % supposés,
donc notre 32,2 % est bien plus loin d'un modèle sain qu'estimé.
Enfin la corrélation NLL/identifiant vaut 0,434 chez nous contre 0,163 en
référence — à surveiller, mais probablement un effet indirect de la
difficulté.

**Chaîne causale complète** : les couches d'attention linéaire produisent
11 à 40 % d'erreur → les activations finales sont bruitées → les logits
le sont → même les prédictions triviales deviennent incertaines →
perplexité ×20 alors que l'argmax reste souvent correct (32 % de top-1),
d'où du texte plausible. Le paradoxe de la journée est résolu.

Point structurant pour la suite : `gdn.py` n'utilise QUE les fonctions de
référence de `transformers`. Valider notre couche contre `transformers`
revenait donc à la comparer à elle-même — le test ne pouvait pas échouer.
L'écart mesuré oppose en réalité **transformers à llama.cpp** sur cette
architecture, et la question devient : laquelle des deux implémentations
est juste. Candidat précis, à vérifier en premier : llama.cpp normalise
explicitement q et k après la convolution causale (`ggml_l2_norm` avec
l'epsilon du modèle, qwen35.cpp:318-321) là où acvram délègue au noyau
par `use_qk_l2norm_in_kernel=True` (gdn.py:120).

### Addendum, 18 h 30 — CAUSE RACINE confirmée : le drapeau `gdn_a_log_negexp` n'atteint pas le moteur

La « fausse cause racine » rétractée à 13 h était en réalité vraie, et
c'est la rétractation qui était fausse. Preuve prise dans le moteur
chargé, pas dans le source (session OnePlus), et vérifiée
indépendamment ici sur les deux dossiers convertis :

  `config.json`                     gdn_a_log_negexp = **True**
  `acvram_manifest.json` → model    **ABSENT**

Or `load_model` reconstruit la spec depuis le MANIFESTE (`loader.py:155`)
et `ModelSpec.to_dict()` ne sérialise pas `raw` — la branche
d'inversion de `loader.py:595` ne s'exécute donc jamais en service. Le
facteur de décroissance est pris tel quel puis retransformé par
`−exp()` dans gdn.py : deux à cent fois trop fort. Mesuré au chargement :
`a_log` min −0,3379 / max −0,0038, identique au disque, là où la valeur
inversée vaudrait −5,5609 / −1,0850.

Effet sur la couche 0, entrée exacte de llama.cpp : sortie de la GDN
1,109e−01 → **1,124e−02** (÷10), sortie de couche 1,089e−01 →
**2,177e−02** (÷5), et les normes concordent enfin (291,4 contre 292,5,
il manquait 8 % d'énergie). Cela explique exactement le profil mesuré
une heure plus tôt — 1 % sur les attentions pleines, 11-40 % sur les
linéaires, seules porteuses d'un `a_log` — et la constance du facteur
global.

**Toutes les perplexités hybrides de la journée sont à refaire**, et le
vrai q3n est probablement très en dessous de 209. Correctif : `load_model`
complète `spec.raw` depuis `config.json` quand le manifeste ne le porte
pas — répare les modèles DÉJÀ convertis, sans reconversion.

Leçon, symétrique de celle du matin : **vérifier un objet ne dit rien
d'un autre**. La rétractation reposait sur `load_model_spec(dossier).raw`,
que le chargeur n'utilise pas ; lire le code ne remplace pas mesurer ce
qu'il produit. Le relevé qui contenait déjà la réponse — « clés gdn dans
manifest[model] : aucune » — était sous nos yeux depuis le matin.

### 8 septembre, 17 h 20 — après correctif : le coût réel du format, enfin mesuré

Mêmes 584 fenêtres, 148 920 positions notées, wiki.test.raw, cadrage
512/512/min_context 256 pour tous.

  étalon llama.cpp, Q3_K_S source            9,1831 ± 0,073
  acvram NVFP4  (4,5 bpw)                   **13,692**   +0,398 nat
  acvram q3n    (3,25 bpw)                  **15,361**   +0,514 nat
  → **coût propre du format q3n : +0,115 nat, +12,2 % pour −25 % d'octets**

Avant le correctif, les mêmes dossiers rendaient 193,6 et 209,0 : le
facteur 21-24 qui masquait tout est tombé à 1,5-1,7. Et l'essai court
rend du français correct là où il produisait « URTURTURT ».

**Contrôle par l'immobilité, plus convaincant que le gain** : phi-4
dense rend **6,911 avant ET après** le correctif, au millième près — un
modèle sans `a_log` ne devait pas bouger, il n'a pas bougé ; de même les
couches d'attention pleine sont inchangées au chiffre près dans la mesure
par couche. Un correctif se prouve autant par ce qu'il laisse intact.

Biais d'instrument, MESURÉ et non estimé : phi-4 int8 6,911 contre 6,5988
= **+0,0461 nat (+4,73 %)** sur dense, au même cadrage.

Décomposition en cours des 0,398 nat du NVFP4 :
  0,398 = coût de l'opération de requantification + biais d'instrument
          (0,0461) + coût de grille propre au NVFP4 + résidu
Le troisième terme avait été posé à zéro par erreur — 4,5 bits depuis 3,4
reste un changement de grille et coûte quelque chose. Témoins en cours
(session de mesure) : Q3_K_S → Q3_K_S (opération seule) et Q3_K_S →
Q4_K_S (opération + grille plus fine). Réserve posée d'avance : rien ne
garantit que ces termes s'additionnent linéairement en log-vraisemblance ;
une somme juste à 2 % conclurait, une somme fausse de 20 % mettrait
peut-être en cause l'additivité plutôt qu'un terme manquant.

**Comment lire le +12,2 %** : c'est un arbitrage, pas une victoire. Sur
un modèle qui tient en mémoire, personne ne paiera 12 % de qualité pour
un quart d'octets ; sur un modèle exilé, ce quart décide de ce qui reste
en VRAM et de ce qui traverse le bus — poste qui gouverne le débit
(5,7 Go/s effectifs sur 18,7, 479-541 Mio par jeton). **q3n se justifie
par le tiering, pas dans l'absolu.** La mesure qui trancherait vraiment,
et que personne n'a faite : à mémoire constante, q3n avec moins de
couches exilées contre NVFP4 avec davantage d'exil — les 12,2 % de
perplexité contre les jetons par seconde et les joules que l'exil évité
fait gagner. Réserve qui voyage avec le chiffre : il vaut pour une
requantification depuis une grille contenant un zéro, et ne se transporte
pas à des poids bf16.

### 8 septembre, 18 h — décomposition des 0,399 nat : 81 % restent inexpliqués

Témoins mesurés par la session de mesure, mêmes 584 fenêtres, même
cadrage, chaîne llama.cpp validée idempotente au passage :

  (a) opération de requantification (Q3_K_S → Q3_K_S)   **0,0000 nat**
      identité à la quatrième décimale : déquantifier puis requantifier
      sur la même grille ne coûte rien
  (b) changement de grille vers plus fin (→ Q4_K_S)     **0,0291 nat** (+3,0 %)
  (c) biais d'instrument, mesuré sur DENSE (phi-4)      **0,0461 nat** (+4,73 %)
  ————————————————————————————————————————————————————————————————
  écart NVFP4 − étalon                                   0,3994 nat
  **(d) résidu = 0,324 nat, soit +38,3 % — 81 % de l'écart**

Le seuil de réalité était 0,01 nat : on est trente fois au-dessus. Le
résidu est réel.

**Ce que (d) contient, et qui n'est pas forcément un défaut** : tout ce
que notre chaîne de conversion fait EN PLUS d'un changement de grille —
recherche d'échelles par canal (AWQ), rotation de Hadamard, filet de
promotions, mélange de formats dans un même dossier — plus le biais
d'instrument sur architecture HYBRIDE, jamais mesuré (le (c) ci-dessus
vaut pour du dense). Pour absorber les 0,324 à lui seul, ce biais devrait
valoir sept fois le biais dense : possible mais peu probable, il restera
sans doute quelque chose.

Réserves du témoin (b), à garder avec le chiffre : il mesure ce que
llama.cpp paie pour changer de grille, pas ce que NVFP4 paie chez nous
(grilles et structures différentes) ; c'est une borne. Mais elle joue
CONTRE l'hypothèse d'un gros résidu — si le vrai coût de format était
plus élevé, le résidu diminuerait d'autant. Qu'il reste 0,324 malgré une
borne basse est donc un argument solide.

**Suite décidée** : (1) mesure du biais sur hybride — le 12B témoin
contre les étalons 31,6919 (c=512) et 24,0799 (c=2048), seul terme
pouvant encore absorber une part importante sans hypothèse ; (2) si le
résidu survit, le décomposer par ce que fait notre conversion —
convertir le même modèle SANS AWQ, SANS Hadamard, SANS promotions, et
mesurer chaque variante. Chacune est censée AMÉLIORER la qualité ; si
l'une la dégrade, le fil est tenu. Quatre conversions, quatre mesures,
chaque terme isolé au lieu d'être deviné.

### 8 septembre, 19 h — le témoin hybride est guéri, le résidu survit entier

Après correctif, 12B témoin int8, corpus entier, contrôle d'`a_log`
chargé conforme (−5,5609 / −1,0850) :

  fenêtre 512, contexte 256    acvram **31,933** contre 31,6919 → **+0,76 %**
  fenêtre 2048, contexte 1024  acvram **24,005** contre 24,0799 → **−0,31 %**

Ce modèle rendait **129,185 et 221,914** le matin : les facteurs 4,08 et
9,21 ont entièrement disparu, et le second écart est **négatif** —
acvram fait très légèrement mieux que llama.cpp. Le « fait sans cause »
qui a lancé la chasse n'existe plus.

**Avertissement retiré par son autrice** : la « dépendance du biais à la
longueur de contexte » venait de seize fenêtres non comparables ; sur le
corpus entier l'écart décroît légèrement, comme le biais dense, et les
deux valeurs sont trop petites pour en tirer une loi. La décomposition
n'a pas besoin de cette limite. Le cumul intermédiaire montre pourquoi :
24,05 après 16 fenêtres, 23,82 après 32, 22,79 après 64, 24,14 après
128 — **6 % d'oscillation**, largement de quoi fabriquer une fausse loi.

**Conséquence sur la décomposition, et elle ne va pas dans le sens
espéré** : le biais d'instrument sur HYBRIDE vaut environ zéro. Il ne
peut absorber aucune part des 0,324 nat de résidu, qui survit entier.

**Et la question qui se retourne** : si le biais est nul sur hybride,
pourquoi vaut-il **+4,73 % sur DENSE** (phi-4 : 6,911 contre 6,5988) ?
Une architecture plus simple s'écarte six fois plus qu'une architecture
compliquée. Cette anomalie semblait négligeable à côté du facteur vingt ;
elle ne l'est plus, et c'est peut-être elle qui porte le résidu. Le plan
de décomposition (convertir sans calibration, sans Hadamard, sans
promotions, mesurer chaque variante) devrait donc s'appliquer **à phi-4
d'abord** : modèle plus petit, mesure plus rapide, et l'écart inexpliqué
y est proportionnellement plus grand.

### 8 septembre, 18 h 05 — le biais d'instrument pur est nul

Témoin choisi pour ne rien mêler : **Qwen3-0.6B-Q8_0**, dense, minuscule,
source déjà en huit bits — converti en **bf16 PUR** (310 tenseurs, tous
bf16, aucune quantification), donc ni perte de format ni changement de
grille à démêler.

  llama.cpp sur le GGUF Q8_0    **21,9438 ± 0,192**
  acvram bf16 pur               **22,0040**   (148 920 positions)
  écart                         **+0,0027 nat, +0,27 %**

Sous le seuil de réalité de 0,01 nat : **le calcul de perplexité d'acvram
et celui de llama.cpp donnent le même nombre** quand aucune
quantification n'intervient. L'hypothèse d'un défaut d'instrument, ouverte
le matin, est fermée.

**Portée exacte, à ne pas dépasser** : ce qui est établi, c'est que le
calcul de perplexité lui-même est juste sur un DENSE non quantifié de
0,6 milliard de paramètres. Ce n'est pas « acvram n'a pas de biais » : un
MoE de 80 milliards servi en tiering emprunte d'autres chemins (routage,
exil, transport d'experts) qui ne sont pas couverts par ce témoin.

**Deux conséquences.**
1. **Le terme (c) était surestimé d'un facteur dix-sept** : le biais pur
   vaut 5,9 % du 0,0461 nat retenu. Les +4,73 % de phi-4 ne sont donc pas
   un biais mais le **coût de la requantification Q4_K_M → int8**, ce qui
   recoupe les +3,0 % mesurés pour Q3_K_S → Q4_K_S par une voie
   totalement indépendante — deux chemins, même ordre de grandeur.
   Note de méthode : la conversion de référence de phi-4 n'avait NI
   calibration, NI Hadamard, NI promotions (lu au manifeste avant de
   lancer quoi que ce soit) — quatre conversions épargnées et une
   prédiction réfutée sans mesure.
2. **Le résidu de Coder-Next s'aggrave** :
   0,3994 = (a) 0,000 + (c) **0,003** + coût de grille ≈ 0,029 +
   **(d) ≈ 0,367 nat, soit +44 % et 92 % de l'écart total.**
   Mais il s'aggrave dans la bonne direction : ce n'est ni l'instrument,
   ni l'opération de requantification, ni le changement de grille. Restent
   deux différences avec ce témoin — l'**architecture hybride** (le 12B de
   Manon la départage : +0,76 % et −0,31 %, donc elle n'explique rien) et
   **ce que la chaîne de conversion fait en plus** sur les gros modèles :
   échelles par canal, Hadamard, promotions, formats mélangés. C'est la
   seule piste qui reste, et elle se décompose étape par étape.

### 8 septembre, 18 h 15 — un second défaut de convention : la normalisation des poids d'experts

Même mécanisme que l'`a_log` du matin : une convention absente du
fichier, un défaut choisi par nous, la vraie valeur codée en dur chez
llama.cpp.

  `lfm2.cpp:32` et `qwen3next.cpp:483` : `build_moe_ffn(..., norm_w = **true**, ...)`
  GGUF LFM2.5 et Coder-Next : **aucune clé `expert_weights_norm`**
  `gguf.py:528` (branche lfm2moe) : `bool(g("expert_weights_norm", **False**))`

Avec un routage à quatre experts, des poids non normalisés somment à une
valeur arbitraire au lieu de 1 : la sortie du bloc part à une échelle
fausse ET variable selon le jeton.

**Mesuré** sur LFM2.5-8B-A1B converti en bf16 PUR (aucune quantification,
modèle résident, 146 717 positions) :

  llama.cpp Q4_K_M                    **33,148 ± 0,321**
  acvram, normalisation absente       **110,350**   (×3,33)
  acvram, normalisation forcée        **58,815**   (×1,77)
  → le défaut vaut **0,629 nat, facteur 1,88** — et il reste **+77 %**

**Audit complet du routage contre llama.cpp**, fait champ par champ
plutôt que supposé :
* **scoring** : `qwen3moe.cpp:89` et `qwen3next.cpp:476` codent
  `GATING_FUNC_TYPE_SOFTMAX` **en dur**, sans lire le fichier — notre
  défaut `softmax` est juste. La règle « clé absente donc sigmoïde » que
  j'allais généraliser est fausse pour nos familles : llama.cpp ne force
  le sigmoïde que pour AFMOE, MISTRAL4, GLM4_MOE, GLM_DSA et STEP35.
* **échelle** : `llama-graph.cpp:1413` — `if (w_scale != 0.0f && w_scale
  != 1.0f)`, donc leur défaut 0,0 et le nôtre 1,0 sont tous deux neutres.
* **normalisation** : le seul écart réel, confirmé pour lfm2moe ET
  qwen3next.

**Ce qui reste** : les +77 % de LFM2.5 après correction. Le modèle est
hybride (`shortconv.l_cache = 3`), et la table des témoins garde sa
forme — les quatre modèles fautifs sont hybrides, le seul propre
(Qwen3-0.6B, transformeur classique) sort à +0,27 %. Le défaut de
normalisation était réel mais partiel : il masquait la piste hybride, il
ne la remplace pas. Témoin décisif en cours : `Qwen3-Coder-30B-A3B`, un
MoE en architecture Qwen3 classique — ni convolution, ni récurrence, et
routage vérifié identique des deux côtés.

Contrôle de plausibilité qui a servi : `gemma-4-31B` rend **4452** de
perplexité chez llama.cpp — modèle inutilisable comme témoin, écarté
avant d'avoir mesuré quoi que ce soit avec.

### 8 septembre, 18 h 30 — deux faits d'instrumentation qui qualifient tous nos débits

**1. Le régime de puissance a changé ce soir, et la colonne du banc le
date.** `plafond_W` (somme des deux cartes, `energie.py:213`) donne
l'horodatage : lignes de 18h15-18h20 à **775 W** (500 + 275), lignes
après 18h26 à **875 W** (500 + 375), comparatif du 3 septembre à
**675 W** (400 + 275). La 5090 était donc déjà à 500 W avant la série du
soir, et la 3080 Ti est passée de 275 à 375 W pendant ou juste après —
mon constat « changé avant aujourd'hui » était faux, ma vérification
était simplement postérieure au changement.
Règle : lire `power.limit` AU MOMENT de la mesure et l'écrire à côté du
chiffre ; deux séries de régimes différents ne se comparent pas en
énergie. **Le comparatif du 3 septembre est à refaire pour deux raisons
indépendantes** : son régime (400/275) n'existe plus, et il tournait en
spéculation `ngram` sans que ce soit un choix — le lanceur ne passe
jamais `--speculative` et le défaut de `cli.py:585` est `ngram`, donc le
mode `mtp` n'a jamais été mesuré, y compris sur les modèles dont nous
chargeons et quantifions la tête MTP.

**2. Le débit était TIRÉ AU SORT à chaque changement de modèle.**
`arreter()` n'attendait que la fermeture du port plus trois secondes ; le
serveur précédent rendait son port **sans avoir rendu sa mémoire**, et le
planificateur du suivant calculait son placement sur ce qui restait.
Preuve involontaire (session OnePlus) : deux entrées de parc pointant le
même dossier sous le même alias ont rendu **15,8 puis 152,2 t/s** —
facteur 9,6, textes différents ; au journal, le premier chargement exilait
5 puis 6 MLP de plus en RAM hôte, le second aucun.
Correctif : `attendre_memoire()` (attend que la mémoire cesse de bouger,
écrit la mémoire libre à côté de la ligne) et `t_s_passages` (les débits
dans l'ordre — min et max ne disent pas LEQUEL s'écarte, et un passage
froid a la même signature qu'une bimodalité).
**Portée** : toute mesure de débit prise juste après un changement de
modèle a pu l'être sur un plan dégradé, sans aucun signal. Cela peut
expliquer une part de la dispersion de 21-26 % qui a motivé toute la
série de déterminisme.

À faire après la série en cours : activer le **mode persistant**
(`nvidia-smi -pm 1`, actuellement désactivé sur les deux cartes), avec
mesure de dispersion avant et après sur le même modèle — il évite le
déchargement du contexte GPU entre deux processus et devrait réduire la
latence de démarrage et stabiliser les horloges.

### 8 septembre, 18 h 40 — un dévoreur de mémoire expliquait nos morts au hasard

`tokensave sync` avalait la mémoire vive à **25 Mo par seconde** et pesait
**14,25 Go** à son arrêt ; la machine est passée de 67 à **81 Go
disponibles**. Il saturait les 93 Go et le système tuait tout ce qui
demandait de la mémoire — évaluations, étalons, suites de tests, chez les
trois sessions, y compris quand la machine paraissait calme.

**Ce que ça réécrit** : plusieurs échecs attribués à la contention entre
nos mesures, ou au cache de pages laissé par une lecture massive, avaient
en réalité cette cause unique. L'hypothèse du cache de pages était
plausible et donnée comme telle ; elle est probablement fausse. Le
processus fautif était invisible parce qu'il ne portait aucun nom
évocateur et ne figurait dans aucune liste de suspects — chercher un
coupable parmi ceux qu'on surveille laisse passer celui qu'on ne
surveille pas.

**Mode persistant activé** sur les deux cartes (il était désactivé). Il
supprime le déchargement du contexte GPU entre deux processus, donc une
source de latence variable qu'on soupçonnait sans pouvoir la nommer.

**Trois régimes de mesure désormais**, à écrire à côté de tout chiffre :
675 W · 875 W · 875 W avec persistance. Et une distinction qui sauve la
moitié du travail :
* **invalidé** — tout ce qui se mesure en watts, joules ou secondes :
  17,4 J/jeton, 104 jetons/kJ, les débits, les temps de première réponse,
  le profil du bus ;
* **intact** — **toutes les perplexités** et tous les SNR de format. Une
  perplexité est un calcul déterministe sur des poids fixes : le plafond
  de puissance et la persistance changent le temps qu'elle met à sortir,
  pas sa valeur. L'étalon 9,7380 et la table des témoins restent valides.

Le dossier **qualité** est donc préservé, le dossier **performance** est
à refaire. La série de déterminisme, écrite le matin et jamais exécutée
faute de machine calme, devient possible pour la première fois : les
trois obstacles (dévoreur de mémoire, contexte GPU déchargé, construction
de ROM concurrente) sont tombés ensemble. Prédiction écrite d'avance : si
la persistance était la cause principale, la dispersion passe sous 10 %
et le premier passage cesse d'être aberrant ; si elle reste à 20 %, il
faut chercher dans l'ordonnancement ou l'allocateur.


## 15 septembre 2026 — le MoE décode par la MMA groupée (v0.6.1)

À b=12, les GEMV MoE dépensaient 1,4-2,6 instructions par octet DRAM
(71 % des instructions du pas, chaque noyau au plafond de 400 W) là où la
MMA block-scaled de CUTLASS en fait 0,18 (`revue/instr-par-octet-14-09.md`,
×8,5 sur le pas contre vLLM). Le chemin MMA du prefill est rendu capturable
au décodage (`MoEBlock._forward_grouped_mma` : grille de tuiles fixe,
`scatter_add_` au lieu de `bincount`, fantômes à poids nul ; trois noyaux
groupés corrigés d'un accès à `t0 − 1` sur tuile vide). Mesuré : régime
court 22 s, −13,6 % de temps et −19 % de J/jeton, le pas sort du plafond ;
chiffre officiel mode moyen (Laure, rondes ctx 2048) : 17,37 → 16,25 ms,
0,620 → 0,538 J/jeton. Qualité : PPL par le chemin de décodage MMA=1/MMA=0
= 0,9995 (Manon). Défaut `ACVRAM_MOE_DECODE_MMA=1` ; `0` = témoin.

### v0.6.2 (15/09) — garde de lot sur le chemin MMA

Laure, b=1, ABAB 20 s : MMA=0 4,48 ms/pas, 223 t/s, 1,51 J ; MMA=1 7,00 ms,
143 t/s, 1,72 J — une tuile m16 pour un jeton coûte 36 % de débit. Garde
`ACVRAM_MOE_DECODE_MMA_MIN_T` (défaut 6, provisoire, Sage
`revue/sage-mma-lot-15-09.md`) : GEMV sous le seuil, MMA au-dessus ; `t` = lot
réel en eager, godet sous graphes. Seuil définitif par la courbe b=2/3/4/6
(MMA dès que J(MMA) ≤ J(GEMV) et ms ≤ 1,02 × GEMV), à remesurer après
route+pack. Test `tests/test_moe_decode_mma_lot.py`.

### v0.6.3 (15/09) — seuil de lot à 9

Courbe de Laure (16 cellules, une carte) MMA/GEMV : b=2 +36 % ms / −3,5 % J,
b=3 +27 / −6,3, b=4 +24 / +2,2, b=6 +18 / +2,7, b=12 −6,4 / −13,2. Aucun lot
de 2 à 6 ne satisfait le critère (J ≤ et ms ≤ 1,02×) : le coût fixe du
chemin MMA est ≈ 2,0-2,3 ms/pas et ne s'amortit qu'à 12. Défaut
`ACVRAM_MOE_DECODE_MMA_MIN_T = 9` (godets 12 et 16) ; le 6 provisoire servait
b=5-8 à −18 %. À remesurer après route+pack.

### v0.6.5 (15/09) — routage + rassemblement du MoE en un noyau (route+pack)

Le frontend torch du chemin MMA au décodage (argsort, scatter_add_, `_tuiles`,
gather, pad, inv : ~45 lancements par couche, 2 160 par pas) est remplacé par
`moe_route_pack` (un lancement ; tri stable par comptage, tuiles de grille
fixe, fantômes à poids nul), bit à bit égal au torch. Chiffre officiel b=12
(Sage, régime ≥ 20 s au compteur, une carte, Coder-30B) : **12,80 ms/pas,
0,425 J/jeton, 398 W** — contre 14,46 ms / 0,435 J / 361 W sans :
**« −12 % de temps, −2 % de joules : le pas est revenu au plafond »**.
Lancements par pas 3 677 → 1 517 (cliquet ≤ 1 700), jetons identiques.
Défaut `ACVRAM_MOE_ROUTE_PACK=1` ; `0` = témoin torch.

### v0.6.6 (16/09) — seuil de lot à 5 ; échelle AWQ par expert dans la pile

Cellule b=5 de Laure (`revue/verdict-cellule-b5-16-09.md`, ABAB 34 s, energie.py
corrigé) : MIN_T=5 contre 9, ms −0,3 % (égalité), J −4,1 % (SM 2 937 MHz au
lieu de 2 727 : la GEMM sort du plafond 400 W). Critère de Sage tenu → défaut
`ACVRAM_MOE_DECODE_MMA_MIN_T = 5` (godets 8, 12, 16 en MMA ; 2 et 4 en GEMV).

Échelle AWQ par expert dans la pile MoE (`revue/awq-pile-15-09.md`,
`revue/laurine-awq-gate-up-15-09.md`) : tables [E, K] bf16 au chargeur, ligne
(jeton, expert) divisée par s[e] dans `moe_route_pack` avant la quantification,
activation divisée par s_down[e] dans `moe_act`, GEMV et prefill groupé
alignés ; gate et up à échelles distinctes = seconde ligne et seconde
`nvfp4_quant_act` (8 → 9 lancements par couche). Table d'unité sautée. La
pile n'exile plus une couche pour une échelle AWQ.
