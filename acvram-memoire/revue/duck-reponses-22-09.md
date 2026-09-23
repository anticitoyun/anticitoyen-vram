# Réponses brutes duck.ai — Q(13) — 22/09 ~19 h 30

**Question de base** : GEMM étroites M=12 sur RTX 5090 sm_120 (qkv N=5120 K=2048 ; o N=2048 K=4096). TRT-LLM y lit en FP8 W8A8 à 1,81 To/s, nous à 0,71-0,82 To/s, pour les mêmes octets. Quelles techniques de noyau expliquent ce ×2,2 ? Split-K et occupation déjà réfutés. Pour chaque piste, quelle mesure de 5 min au plus la réfuterait ?

**Donnée additionnelle (Jerome, reçue pendant génération)** : notre noyau nvfp4 étroit tourne à 0,44-0,46 To/s à M=12 (10,2-10,8 µs), donc pas plus vite que notre int8 qui lit deux fois plus d'octets.

**Q2 ajoutée** : qu'est-ce qui plafonne un GEMV/GEMM FP4 bloc-16 à M=12 sous 0,5 To/s sur sm_120 (dé-quantification, échelles, forme des accès) ? Et quelle technique le rapproche des 1,8 To/s ?

---

## GPT-5.6 Luna — Q1 initiale (mode raisonnement, 12 s)

L'écart est de 2,2 à 2,55×. vous : 0,71–0,82 To/s ; TRT-LLM : 1,81 To/s ; rendement relatif : 39–45 %.

Avec M=12, le problème est presque certainement le GEMM étroit, pas le débit mémoire brut. QKV : [12,2048]×[2048,5120] ; O : [12,4096]×[4096,2048]. ≈ 252 MFLOP pour QKV et 201 MFLOP pour O par exécution. Le débit affiché en To/s peut être identique côté octets, tout en variant fortement selon l'efficacité des Tensor Cores.

**Causes les plus probables :**

1. **Mauvais mapping des tuiles pour M=12** : une tuile typique 128 lignes laisse une grande partie des lanes inactives. TRT-LLM dispose probablement d'un chemin spécialisé pour petits M alors que notre kernel paie une sous-utilisation importante.
   - Mesure : comparer TFLOP/s effectifs (pas seulement To/s) entre notre kernel et un appel cuBLASLt M=12 identique.

2. **Accumulation FP8 mal exploitée** : vérifier que nous utilisons bien le chemin FP8 × FP8 → FP16/BF16 ou FP32 sur Tensor Cores, non une conversion FP8→FP16 suivie d'un GEMM classique.
   - Mesure : regarder dans le profil SASS si `mma` reçoit des opérandes en FP8 directement.

3. **Layout des poids** : une transposition/reformatage implicite peut effacer une grosse partie du gain.
   - Mesure : comparer temps avec poids déjà packés vs runtime.

4. **Fusions manquantes** : à M=12 les coûts fixes (scales, conversion, biais, activation, stores intermédiaires, lancements) deviennent importants. TRT-LLM peut employer un kernel persistent.
   - Mesure : comparer temps GEMM seul vs GEMM+scales+biais.

5. **Trop de petits lancements** : si les kernels ont chacun un bon débit mais que le graphe global est lent, le problème vient des lancements, synchronisations ou allocations.
   - Mesure : comparer temps QKV seul vs QKV+O+annexes.

**Premier test recommandé :** pour chaque matrice, comparer 4 variantes :
| Test | QKV | O |
|---|---|---|
| GEMM FP8 brut, poids déjà packés | mesurer | mesurer |
| GEMM FP8 avec scales | mesurer | mesurer |
| chemin complet avec conversions/layout | mesurer | mesurer |
| graphe fusionné | mesurer | mesurer |

Sous Nsight Compute : utilisation des Tensor Cores ; occupation et nombre de warps actifs ; temps passé en conversion FP8 ; transactions globales et L2 ; présence de kernels transpose/reformat ; temps entre les kernels ; dimensions et layouts réellement transmis à cuBLASLt/CUTLASS.

---

## GPT-5.6 Luna — Q2 additionnelle (même fil, mode raisonnement, 21 s)

Le plafond à 0,44–0,46 To/s vient probablement moins de la déquantification arithmétique que de la combinaison : **petit M + faible réutilisation des poids + chargement des scales mal structuré**.

À M=12, un MMA NVFP4 SM120 travaille naturellement sur un atome d'environ **16×8×64**. Les deux lignes manquantes ne suppriment pas le coût de la tuile : elles se traduisent par des lanes inactives, de la prédication et une mauvaise amortisation des transferts. Le chemin matériel SM120 expose bien un MMA block-scaled NVFP4 natif, mais son débit théorique ne s'applique pas directement à une petite tuile M=12.

Le format NVFP4 ajoute deux flux :
- valeurs FP4 = 0,5 octet par valeur ;
- scale FP8 par bloc de 16 = 1/16 octet par valeur + scale globale.

Surcoût brut des scales ≈ **12,5 %** par rapport aux données FP4. Si on décompacte les nibbles, convertit en FP16/BF16 et charge les scales séparément : coût dominant, plus d'instructions, plus de registres, plus de trafic SMEM, disparition du bénéfice du MMA block-scaled natif.

Le deuxième problème est la **forme des accès aux scales** : elles ne doivent pas être lues comme une matrice FP8 linéaire [N, K/16]. Pour SM120, l'atome MMA est `m16n8k64` et consomme des fragments de scales de forme **16×4** pour A et **4×8** pour B. Une disposition naïve produit des chargements strided, des duplications entre threads et des transactions de petite taille.

**Symptôme typique** : beaucoup de temps dans les instructions de chargement/permutation ; peu d'utilisation des Tensor Cores ; registres élevés ; consommation mémoire globale étonnamment faible ; aucune accélération FP4 vs INT8.

**5 causes dans notre cas (classement) :**
1. **M sous-dimensionné** : M=12 < granularité naturelle M=16 du MMA. Tuile payée presque entièrement pour 12 lignes utiles.
2. **Absence de réutilisation inter-tuile de A** : pour chaque tuile de N, les 12 lignes d'activation peuvent être rechargées. À faible M, le coût des poids reste élevé par rapport au nombre de FMA utiles.
3. **Scales B non swizzlées selon le format matériel** : layouts block-scaled utilisent des blocs physiques de type 128 éléments dans la dimension M/N et 4 groupes de scales dans K.
4. **Déquantification hors MMA** : si profil SASS montre des boucles d'unpack, conversions FP4→FP16 ou multiplications explicites par les scales avant `mma.sync`, c'est le principal plafond.
5. **Pipeline trop court** : à 10 µs, un ou deux petits stages de chargement non chevauchés avec le calcul suffisent à laisser les Tensor Cores inactifs. Il faut un pipeline GMEM→SMEM→registre→MMA double-bufferisé.

**La technique qui donne le plus de chances d'atteindre le niveau TRT :**

> MMA NVFP4 natif + poids et scales pré-packés dans le layout SM120 + kernel persistent/grouped avec staging TMA ou équivalent, et regroupement de plusieurs problèmes pour augmenter M effectif.

Concrètement :
1. **Utiliser directement l'instruction block-scaled NVFP4** : SASS doit contenir `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec` et non un GEMM FP16 précédé d'une déquantification logicielle.
2. **Pré-packer les poids une seule fois** : FP4 et scales arrangés dans le layout attendu par la MMA. Scales swizzlées en blocs physiques adaptés, structure 128×4 décrite par CUTLASS.
3. **Charger poids et scales dans le même pipeline** : `GMEM weights/scales → TMA ou chargement vectorisé → SMEM swizzled → copie vers fragments de registres → native block-scaled MMA`.
4. **Traiter M=12 comme M=16** : padder logiquement à 16 lignes, masque uniquement lors du store final. Tests de validité à l'intérieur de la boucle K très coûteux.
5. **Augmenter le M effectif par regroupement** : regrouper plusieurs GEMM indépendants dans un kernel persistent ou grouped GEMM afin que le scheduler voie un M effectif de 32, 64 ou 128.
6. **Conserver les activations dans SMEM** : les 12 lignes de A sont réutilisées sur de nombreuses tuiles N.

**4 variantes de mesure** (≤ 5 min chacune) :
| Variante | Résultat attendu |
|---|---|
| FP4 déquantifié en FP16 puis GEMM | référence lente |
| MMA NVFP4 natif, scales linéaires | amélioration partielle |
| MMA NVFP4 natif, scales swizzlées SM120 | gros gain |
| même kernel avec M effectif 64/128 | rapprochement du plafond TRT |

Si la version native + swizzle reste à 0,45 To/s → limitant = petit-M/persistent scheduling. Si elle monte nettement mais reste sous TRT → réutilisation de A, nombre de stages du pipeline ou regroupement des tuiles.

**Conclusion Luna** : "charger deux fois moins d'octets" ne garantit pas deux fois plus de débit. En FP4, la réduction du payload est obtenue au prix d'un format sous-octet et de metadata à layout contraint. À M=12, le coût fixe du mouvement des tuiles et des scales peut dépasser le coût du calcul FP4 lui-même. La technique la plus réaliste : **kernel persistent, grouped GEMM, poids prépackés, MMA NVFP4 natif et plusieurs problèmes fusionnés pour augmenter le parallélisme et le M effectif**.

---

## Gemma 4 31B — Q1+Q2 (mode raisonnement, 16 s)

**1. Le paradoxe du NVFP4 (4-bit vs 8-bit)**

Le fait que le noyau FP4 soit plus lent (en To/s) que le noyau Int8, alors qu'il déplace deux fois moins de données, est le signal majeur : on n'est plus limité par la bande passante mémoire brute, mais par le **débit d'instructions (Instruction Throughput) et la latence de dépaquetage**.

- **Coût de dépaquetage (Unpacking)** : en FP4, les données sont stockées de manière compacte. Pour les utiliser dans les Tensor Cores, elles doivent être extraites et converties. Si le dépaquetage est fait via des opérations scalaires (shifts, masks) au lieu d'instructions vectorisées, le temps passé à "préparer" la donnée dépasse le temps gagné lors du chargement depuis la VRAM.
- **Efficacité du pipeline** : à M=12, le ratio calcul/chargement est très faible. Le pipeline de dépaquetage crée des bulles (stalls) dans l'exécution des instructions MMA. Le GPU passe son temps à attendre les registres, faisant chuter le To/s effectif.

**2. Pourquoi TRT-LLM est à 1,81 To/s (FP8)**

TRT-LLM utilise des optimisations spécifiques à l'architecture Blackwell (sm_120) que nos noyaux actuels ne semblent pas exploiter pleinement :

- **cp.async + TMA (Tensor Memory Accelerator)** : TRT-LLM utilise probablement le TMA de Blackwell pour déplacer les données directement de la mémoire globale vers la mémoire partagée (SMem) sans passer par les registres, minimisant l'utilisation des instructions de load.
- **Alignement et Vectorisation** : pour M=12, TRT-LLM optimise likely le "padding" ou utilise des kernels spécialisés où les accès mémoire sont parfaitement alignés sur 128 octets.
- **Utilisation native FP8** : FP8 est supporté nativement par les Tensor Cores sm_120 sans dépaquetage complexe. Le flux est `Load → MMA`. Pour FP4 : `Load → Unpack → MMA`.

**3. Pistes d'optimisation**

Optimiser le dépaquetage FP4 :
- Instructions de chargement vectorisées (`ld.global.v4.u32` ou similaire).
- Instructions de manipulation bits intrinsèques pour dépaqueter plusieurs éléments FP4 simultanément dans un seul registre.
- Si possible, déporter le dépaquetage dans SMem pour masquer la latence.

Gestion "Narrow" (M=12) :
- **Tiling** : vérifier qu'on utilise un mapping des colonnes N pour saturer les unités de calcul.
- **Occupancy** : optimiser le nombre de blocs par SM pour masquer la latence mémoire sans saturer les registres.

Exploitation sm_120 :
- **Instructions MMA natives** : s'assurer d'utiliser les instructions `mma` natives pour FP4/FP8 de l'architecture Blackwell. Si on simule FP4 via FP16, on perd tout l'avantage matériel.
- **Cache** : à M=12, les activations sont petites et tiennent en L1/L2. Le goulot d'étranglement est le flux des poids. Assurer que les accès aux poids sont strictement séquentiels et coalescés.

**Résumé Gemma** : notre noyau FP4 souffre d'un "unpacking penalty" qui annule le gain de bande passante. Pour atteindre les performances TRT-LLM, il faut réduire le nombre d'instructions nécessaires pour amener une donnée du bus mémoire jusqu'à l'unité MMA.

---

## gpt-oss 120B — Q1+Q2 (mode raisonnement, 1 s)

Le fait que l'implémentation FP8 (W8A8) atteigne seulement 0,71–0,82 To/s alors que TRT-LLM obtient 1,81 To/s (avec les mêmes octets) peut s'expliquer par plusieurs facteurs techniques :

1. **Utilisation des unités Tensor FP8 (nvfp4)** : le kernel nvfp4 mesuré à 0,44–0,46 To/s (10,2–10,8 µs) montre que la capacité de débit des unités Tensor core est bien inférieure à son potentiel théorique. Le kernel ne semble pas effectuer les opérations de manière alignée avec le mode de fonctionnement naturel des Tensor Cores SM 12.

2. **Overhead de conversion/quantisation** : en FP8, chaque opération implique souvent une conversion de la représentation interne (par ex. FP8 → FP16/FP32) pour les calculs intermédiaires, et une recompression après. Ces conversions ajoutent des cycles CPU/GPU supplémentaires et réduisent le débit effectif.

3. **Alignement et charge de données** : les charges de 8 bits (int8) sont naturellement alignées sur les bus de mémoire de 32 bits, ce qui maximise l'utilisation du bus de données. Pour FP4 ou FP8 spéciaux, si les données ne sont pas correctement alignées ou regroupées, les transactions mémoire peuvent être moins efficaces.

4. **Optimisation du kernel** : TRT-LLM utilise probablement des kernels spécialement tunés pour le tensor core FP8 (SM 12) avec un tirage de micro-tuiles qui maximise le remplissage des registres et minimise les accès mémoire. Votre kernel nvfp4 ne pas exploiter pleinement ces micro-architectures (par ex. taille des blocs, vectorisation), d'où la différence de performance.

**Points d'amélioration possibles :**
| Aspect | Pourquoi c'est limitant | Action suggérée |
|---|---|---|
| Taille des blocs | Des blocs trop petits laissent les tensor cores sous-utilisés | Expérimentez avec des blocs (64×64, 128×128) adaptés aux capacités de SM 12 |
| Mise à l'échelle FP8 | Conversions supplémentaires augmentent la latence | Intégrez les facteurs de mise à l'échelle directement dans les poids (pré-quantifiés) ou utilisez les nouvelles instructions `FP8Scale` |
| Utilisation des registres | Registres sous-utilisés → plus d'accès mémoire | Réduisez la taille du shared memory ou réarrangez les chargements pour garder davantage de valeurs en registre |
| Pré-fetching / double-buffering | Le pipeline de chargement peut être bloqué par des latences de mémoire | Implémentez un double-buffering explicite pour masquer les temps d'attente |
| Profiling détaillé | Sans profil détaillé, il est difficile d'isoler le goulot | Utilisez Nsight Compute avec les métriques `sm__warps_active.avg.pct_of_peak_sustained_active` et `dram__bytes_read.sum` |

**Résumé gpt-oss** : le kernel FP8 est limité davantage par la **computational throughput** et les **overheads de quantisation** que par la bande passante mémoire brute. Aligner la conception du kernel sur les meilleures pratiques de Tensor-core FP8 (taille de bloc, faisceaux de mise à l'échelle, utilisation maximale des registres) devrait rapprocher le débit de celui observé par TRT-LLM.
