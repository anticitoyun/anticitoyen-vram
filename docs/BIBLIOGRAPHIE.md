# Bibliographie — inférence LLM sur Blackwell grand public

Quatorze articles relevés le 3 septembre 2026, identifiants et titres vérifiés
auprès de l'API arXiv. Chaque fiche dit **ce que l'article apporte** et **ce
qu'on en fait ici** : lecture à faire, expérience à monter, ou simple point de
comparaison.

## Les quatre à lire en premier

| # | article | pourquoi d'abord |
|---|---|---|
| 5 | H-Scale | affine le choix des échelles NVFP4 **sans coût à l'inférence** — c'est exactement notre format et notre contrainte |
| 6 | SharQ | parcimonie d'activation + FP4, chiffres annoncés **sur RTX 5090** |
| 7 | Diagnostic FP4 couche par couche | explique pourquoi certaines couches encaissent mal le FP4 : notre plancher de SNR est empirique, celui-ci est raisonné |
| 3 | FlashInfer | moteur d'attention paginée, graphes CUDA, ordonnancement : notre `paged_attn_*` en est la version artisanale |

---

## 1. Private LLM Inference on Consumer Blackwell GPUs
*A Practical Guide for Cost-Effective Local Deployment in SMEs* —
[arXiv:2601.09527](https://arxiv.org/abs/2601.09527), 14 janvier 2026.

79 configurations sur RTX 5090 / 5070 Ti / 5060 Ti, avec Qwen3-8B, Gemma3-12B,
Gemma3-27B et GPT-OSS-20B : BF16, W4A16, **NVFP4**, MXFP4, contextes de 8 K à
64 K, RAG, multi-LoRA, API en forte concurrence. Résultat mis en avant :
**NVFP4 ≈ 1,6× le débit de BF16 pour ≈ 41 % d'énergie en moins**.

**Chez nous** : c'est la mesure indépendante la plus proche de notre poste.
Leur protocole (mêmes modèles, mêmes formats, une seule carte) est
directement comparable à notre banc ; à confronter à nos chiffres du
3 septembre. Le volet énergie manque complètement chez nous — nos bancs ne
relèvent que les t/s, jamais les joules par jeton.

## 2. Silicon Showdown
*Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM
Inference* — [arXiv:2605.00519](https://arxiv.org/abs/2605.00519), 1er mai 2026.

Comparaison d'architectures pour l'inférence grand public : Blackwell, MoE,
modèles au-delà de 70B, quantification, mémoire, débit, efficacité énergétique,
et les **barrières d'écosystème** (ce que le logiciel empêche, indépendamment
du silicium).

**Chez nous** : le chapitre « barrières » recoupe notre expérience — la 5090
était inutilisable par la plupart des moteurs jusqu'à ce que `sm_120` soit
géré, et c'est l'origine du projet.

## 3. FlashInfer
*Efficient and Customizable Attention Engine for LLM Inference Serving* —
[arXiv:2501.01005](https://arxiv.org/abs/2501.01005), 2 janvier 2025.
Code : <https://github.com/flashinfer-ai/flashinfer>.

Attention, cache KV, blocs creux, graphes CUDA, ordonnancement, long contexte.
Intégré dans vLLM, SGLang et MLC. Réduction annoncée de **29 à 69 % de la
latence entre jetons** selon les scénarios.

**Chez nous** : notre `paged_attn_partial_kernel` / `paged_attn_reduce_kernel`
couvre le même terrain à la main, et v0.4.37 vient d'en supprimer le second
lancement quand une seule tranche suffit. Leur découpage en blocs creux et
leur ordonnanceur sont la suite logique. À lire avant d'écrire quoi que ce
soit de plus dans l'attention.

## 4. Pretraining Large Language Models with NVFP4
NVIDIA — [arXiv:2509.25149](https://arxiv.org/abs/2509.25149), 29 septembre 2025.

Transformée de Hadamard aléatoire, quantification **2D**, arrondi stochastique,
précision mixte. Démonstration : un 12B entraîné sur 10 000 milliards de jetons
en NVFP4.

**Chez nous** : l'entraînement ne nous concerne pas, mais deux outils s'y
transposent tels quels. La **transformée de Hadamard** avant quantification
étale les valeurs aberrantes sur le bloc — c'est précisément ce qui fait
plonger le SNR de certaines couches et nous force à les promouvoir en INT8.
La **quantification 2D** (échelles par ligne *et* par colonne) est une piste
pour les couches que notre plancher rejette aujourd'hui.

## 5. H-Scale
*Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference* —
[arXiv:2608.28113](https://arxiv.org/abs/2608.28113), 28 août 2026.

Choix des échelles NVFP4 guidé par la hessienne, **sans surcoût à
l'inférence** : tout se passe à la conversion, le format servi reste le NVFP4
standard.

**Chez nous, la piste la plus directement applicable de la liste.** Notre
convertisseur choisit l'échelle globale par tenseur au min-max (ou proche), et
promeut en INT8 dès que le SNR passe sous le plancher. Une sélection guidée par
la sensibilité pourrait sauver une partie des couches promues — donc rendre des
modèles plus compacts, donc éviter la RAM hôte. À implémenter dans
`acvram/quant/convert.py` et à mesurer contre le plancher de SNR actuel.

## 6. SharQ
*Bridging Activation Sparsity and FP4 Quantization for LLM Inference* —
[arXiv:2606.26587](https://arxiv.org/abs/2606.26587), 25 juin 2026.

Parcimonie d'activation combinée à NVFP4, HiF4 et MXFP4. Chiffres annoncés
**sur RTX 5090** : latence divisée par ≈ 2,2 à 2,4 face au FP16, débit ≈ 1,2 à
1,4× celui du FP8.

**Chez nous** : la parcimonie d'activation est un angle que nous n'avons pas du
tout exploré. Au décodage nous lisons **tous** les poids d'un expert
sélectionné ; si une part des activations est nulle, une partie de cette
lecture est gaspillée. C'est le seul levier de la liste qui attaque les octets
lus plutôt que le nombre de lancements — et les octets lus sont notre mur
(`nvfp4_gemv` plafonne à 767 Go/s, `int8_gemv` à 1780).

## 7. Diagnosing FP4 inference
*A layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4* —
[arXiv:2603.08747](https://arxiv.org/abs/2603.08747), 5 mars 2026.

Analyse couche par couche et bloc par bloc de la sensibilité au FP4 : où
l'erreur naît, comment elle se propage, quel compromis précision/vitesse.

**Chez nous** : notre promotion en INT8 est déclenchée par un plancher de SNR
mesuré, uniforme et sans théorie derrière. Cet article donne la carte de
sensibilité qui manque, et dit lesquelles de nos promotions sont justifiées.
À lire avec le 5.

## 8. ZipServ
*Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless
Compression* — [arXiv:2603.17435](https://arxiv.org/abs/2603.17435),
18 mars 2026.

Compression **sans perte** des poids, décompression consciente du matériel,
moins de trafic mémoire.

**Chez nous** : intéressant exactement là où nous butons — le décodage est lié
à la bande passante. Une compression sans perte au-dessus du NVFP4
réduirait encore les octets lus, à condition que la décompression tienne dans
le noyau sans consommer les registres qui nous ont déjà coûté 30 % sur le
noyau gate-up « large ». À évaluer, sans illusion sur la marge.

## 9. InferCept
*Efficient Intercept Support for Augmented Large Language Model Inference* —
[arXiv:2402.01869](https://arxiv.org/abs/2402.01869), 2 février 2024.

Interruptions d'inférence (appels d'outils, RAG) : préemption, mémoire GPU,
recalcul ou transfert du cache KV. Annoncé : **1,6 à 2× de requêtes servies**,
latence normalisée **1,3 à 12×** meilleure.

**Chez nous** : notre serveur ne gère pas la préemption ; une requête
interrompue par un appel d'outil garde son cache KV occupé. Pertinent le jour
où acvram servira des agents, pas avant.

## 10. P/D-Serve
*Serving Disaggregated Large Language Model at Scale* —
[arXiv:2408.08147](https://arxiv.org/abs/2408.08147), 15 août 2024.

Séparation **prefill / decode** sur des ressources distinctes, transfert du
cache KV entre elles, ordonnancement, décodage spéculatif.

**Chez nous** : nous avons deux cartes très dissemblables (5090 32 Gio,
3080 Ti 12 Gio) et un pipeline bi-GPU déjà essayé et écarté. La séparation
prefill/decode est l'autre façon d'utiliser deux cartes — le prefill est lié au
calcul, le décodage à la bande passante ; ils ne veulent pas la même carte.
Piste à rouvrir avec cet article en main.

## 11. Preble
*Efficient Distributed Prompt Scheduling for LLM Serving* —
[arXiv:2407.00023](https://arxiv.org/abs/2407.00023), 8 mai 2024.

Ordonnancement des invites, réutilisation du cache KV entre requêtes partageant
un préfixe, appels d'outils, charges de travail d'agents.

**Chez nous** : la mise en cache de préfixe **existe déjà**
(`BlockAllocator(n_blocks, enable_prefix_cache)`, active par défaut) — elle est
seulement coupée sur les hybrides à récurrence linéaire, où les blocs KV ne
suffisent pas à restaurer l'état GDN. Ce qui manque en revanche, c'est
l'ordonnancement *entre requêtes* décrit ici : nous réutilisons un préfixe,
nous n'ordonnançons pas les requêtes pour maximiser cette réutilisation.

## 12. KernelSight-LM
*A Kernel-Level LLM Inference Simulator* —
[arXiv:2606.28565](https://arxiv.org/abs/2606.28565), 26 juin 2026.

Simulateur au niveau des noyaux : explique pourquoi deux runtimes servant le
même modèle sur le même matériel n'ont pas les mêmes performances.

**Chez nous** : c'est la question centrale de notre comparatif des quatre
moteurs. Un simulateur dirait où passent nos 6,2 ms par jeton sans monter un
banc à chaque hypothèse — et six de nos huit dernières pistes ont été écartées
*après* implémentation. Complémentaire de Nsight Systems.

## 13. Puro-2B
*Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090* —
[arXiv:2608.27370](https://arxiv.org/abs/2608.27370), 27 août 2026.

Entraînement complet sur une seule RTX 5090 : FP8, 1 400 milliards de jetons,
architecture de type Qwen, coût contenu.

**Chez nous** : hors périmètre du moteur, mais c'est la référence si l'on veut
un jour affiner un modèle sur cette carte plutôt que seulement le servir.

## 14. Quartet
*Native FP4 Training Can Be Optimal for Large Language Models* —
[arXiv:2505.14669](https://arxiv.org/abs/2505.14669), 20 mai 2025.

FP4 natif à l'entraînement sur Blackwell : débit, efficacité énergétique.

**Chez nous** : même statut que le 13 et le 4 — l'entraînement n'est pas notre
sujet, mais tout ce qui concerne la **stabilité numérique du FP4** se
transpose à la conversion.

---

## Ce que la liste dit de nos angles morts

1. **L'énergie.** Trois de ces articles mesurent des joules par jeton ; nous ne
   mesurons que des jetons par seconde. Ce poste tourne déjà bridé — 5090 à
   **400 W** (au lieu de 600), 3080 Ti à **275 W** (au lieu de 350) — donc le
   compromis puissance/débit est déjà choisi, mais jamais mesuré : nous ignorons
   ce que ces 200 W en moins coûtent réellement en jetons par seconde, et où se
   trouve le point d'inflexion.
2. **La parcimonie d'activation** (6) : le seul levier proposé qui réduise les
   **octets lus**, notre mur réel.
3. **Le choix des échelles** (5, 7) : notre plancher de SNR est empirique et
   uniforme ; deux articles donnent de quoi le raisonner et probablement de
   quoi convertir plus de couches en NVFP4 — donc éviter la RAM hôte.
4. **L'ordonnancement des requêtes** (11) : le cache de préfixe existe, mais
   rien n'ordonne les requêtes pour en tirer parti.
5. **La séparation prefill/decode** (10) : la bonne façon d'exploiter deux
   cartes dissemblables, là où le pipeline bi-GPU a échoué.

---

# Deuxième relevé — 7 septembre 2026

Fait après la nuit du 6 au 7, qui a mis au jour six défauts du chemin
« poids en RAM hôte, calculé sur la carte » et un planificateur optimiste
d'un facteur 2,4 à 3,4 sur tout le parc. Axes retenus ici : FP4 et NVFP4,
choix des échelles, cache de clés-valeurs et de préfixe.

Règle appliquée : un article ne compte que si l'on peut nommer le fichier
qu'il toucherait et la mesure qui dirait s'il marche. Les chiffres viennent
de la section expérimentale, jamais du résumé. Ce qui n'a pas été lu est dit
comme tel.

**Contraintes qui rendent un chiffre comparable ou non.** Les cartes sont
bridées, 400 W et 275 W ; aucun débit publié à pleine puissance ne se compare
à nos références. Lot de un, une requête à la fois. Deux cartes dissemblables,
la 5090 avec les instructions FP4 natives, la 3080 Ti sans. Modèle de travail :
80 milliards de paramètres dont 3,2 actifs, 512 experts, 10 routés — rapport
octets stockés sur octets lus par jeton de 45, contre quelques unités dans la
littérature sur 8 ou 64 experts.

## 15. Microbenchmarking NVIDIA's Blackwell Architecture

`arXiv:2512.02189` — retenu, avec une réserve de transposition.

Mesure les cœurs tensoriels de Blackwell précision par précision. FP4 atteint
7702,5 TFLOPS, soit **96,3 % du pic théorique** ; FP8 3851,4 TFLOPS, 96,3 %
aussi. La latence ne varie que de 1,27 fois entre FP64 et FP4, 11,2 à 14,2
cycles, alors que le débit varie de 177 fois : le débit vient de la largeur
du chemin de données, pas d'un pipeline plus profond.

Sa phrase utile est une conclusion négative : « avec 96 à 99 % du pic
théorique sur toutes les précisions, les cœurs tensoriels ne sont pas le
goulot ; **la bande passante mémoire et le coût de lancement des noyaux** le
sont ». C'est exactement la forme de l'erreur de notre planificateur, mesurée
la nuit dernière : un plancher fixe d'environ 3 ms par jeton plus une
composante croissante avec la taille.

**Réserve** : tout est mesuré sur B200, carte de centre de données, pas sur
une 5090. Aucun chiffre de bande passante mémoire atteinte contre plaque n'y
figure pour une carte grand public, et c'est précisément le chiffre qui nous
manque. L'article ne remplace donc pas la mesure ; il dit seulement que
chercher du côté des lancements et de la mémoire est la bonne direction.

**Ce qu'on en fait** : rien dans le code. Il justifie de mesurer nous-mêmes
la bande passante effective de la 5090 bridée à 400 W, au lieu des 1792 Go/s
de plaque que `acvram/memory/tiering.py` retient encore.

## 16. ScaleSweep — initialisation des échelles de bloc NVFP4

`arXiv:2606.07618` — retenu, et c'est le seul des trois qui touche notre code.

NVFP4 associe un format E2M1 à une échelle FP8 par bloc de 16 et une échelle
globale par tenseur. ScaleSweep dérive des bornes inférieure et supérieure
pour l'échelle de bloc optimale sous deux objectifs, puis balaie l'espace des
motifs de bits FP8 dans ce voisinage seulement.

Résultats de la section expérimentale : sous quantification poids et
activations, le taux de récupération monte à **99,50 % sur Qwen3-8B** ; avec
le cache de clés-valeurs quantifié, **1 à 2 points de récupération** gagnés
selon les modèles, sur RTN comme sur GPTQ. L'écart aux échelles FP32
optimales reste **sous 10 %** dans presque tous les cas, contre les méthodes
AbsMax et 4/6.

La conclusion dit que la méthode « n'introduit qu'un coût négligeable à
l'inférence » : c'est une méthode de **quantification**, pas d'exécution.

**Ce qu'on en fait** : `acvram/quant/nvfp4.py`, au choix de l'échelle de bloc
à la conversion. **Mesure qui dirait si ça marche** : reconvertir un modèle
témoin et comparer la perplexité par `acvram eval`, à débit inchangé
puisqu'aucun noyau ne bouge. C'est l'angle mort « choix des échelles » du
premier relevé, avec cette fois une méthode chiffrée.

**Non lu** : le corps de la démonstration des bornes, et l'annexe E.

## 17. MixFP4 — blocs FP4 ou INT4 selon leur distribution

`arXiv:2605.31035` — **écarté**, malgré des résultats de précision réels.

Constat de départ juste, et qui vaut d'être noté : dans un même tenseur,
quelques valeurs aberrantes coexistent avec de larges régions plates, et un
seul dictionnaire 4 bits ne convient pas aux deux. D'où un choix de format
par bloc, exponentiel pour les blocs à aberrations, uniforme pour les blocs
plats.

Perplexité WikiText mesurée, utile comme point de comparaison même si l'on
n'adopte rien : sur Qwen3-8B, BF16 12,21, **NVFP4 12,74**, NVINT4 12,73,
4/6 12,56. Sur Llama-3.1-8B, BF16 7,33, NVFP4 8,26. Voilà ce que coûte NVFP4
sur des modèles publics, à comparer un jour à nos propres écarts.

**Pourquoi écarté**, et c'est dit par les auteurs : le prototype est une
simulation PyTorch, et « la latence et le débit d'un noyau natif pour MixFP4
dépendent d'un support matériel et d'un travail de noyau hors du champ de cet
article ». Les 3,1 % de surcoût en surface et 1,5 % en puissance sont une
**synthèse en 28 nm**, pas une mesure sur silicium existant. Rien de tout
cela ne s'exécute sur une 5090.

## Vérification de matériel, faite avant de transposer ces chiffres

`nvidia-smi` et `/sys/bus/pci` donnent, sur ce poste : les deux cartes sont
négociées en **x8**, pas en x16. La 5090 monte à 32 GT/s (PCIe 5.0), la
3080 Ti à 16 GT/s (PCIe 4.0). La génération lue au repos vaut 1 : c'est
l'économie d'énergie, pas le lien réel.

La topologie mesurée par `acvram bench --what topology`, dans
`~/.config/acvram/acvram-topology.json`, fait autorité et donne **par carte** :

| carte | hôte → carte | carte → hôte | lien physique |
|---|---|---|---|
| RTX 5090 | 18,7 Go/s | 21,2 Go/s | PCIe 5.0 x8, 31,5 Go/s théoriques |
| RTX 3080 Ti | 11,4 Go/s | 12,6 Go/s | PCIe 4.0 x8, 15,8 Go/s théoriques |

Soit 59 et 72 % du théorique. Ce point est vérifié et non estimé, contrairement
aux 1792 Go/s de bande passante mémoire, qui restent une valeur de plaque.

## 18. SPICE — préchargement spéculatif d'experts

`arXiv:2608.21240` — retenu, et c'est la réponse à la question que je ne savais
pas chiffrer.

Mesuré sur **RTX 5090 avec 128 Go de DDR5**, plus une 4060 et une troisième
carte. Le chiffre : **le chargement des experts occupe 73 à 88 % de la latence
par couche, le calcul 12 à 27 %**, sur DeepSeek-V2-Lite et Qwen2-57B-A14B.
Jusqu'à 3,12× sur le temps par jeton de sortie. Ils dérivent la profondeur
d'anticipation minimale pour cacher le transfert et la rendent adaptative,
la confiance de prédiction variant d'une couche à l'autre.

**Transposition, deux réserves.** Leur banc est en PCIe 4.0 x8 ; le nôtre est
en 5.0 x8, mesuré à 18,7 Go/s contre environ 13 chez eux, donc la part du
transfert serait un peu moindre ici. Et ils routent 6 experts sur 64 ou 160,
nous 10 sur 512 : le rapport octets stockés sur octets lus n'est pas le même.
Leur lot n'est pas indiqué et n'a pas été vérifié comme unitaire.

**Ce qu'on en fait** : `acvram/memory/tiering.py` et le pool par couche. La
courbe 0, 9, 18 couches exilées reste à tracer chez nous, mais on sait
désormais quoi attendre.

## 19. Évaluation reproductible des caches d'experts MoE

`arXiv:2608.07911` — retenu, et **à lire avant d'écrire le cache d'experts**
que la feuille de route promet depuis longtemps.

Ce n'est pas une méthode, c'est le banc qui explique pourquoi les taux publiés
mentent. Trois résultats. Une relecture de trace incohérente **gonfle les
politiques de récence de 27 à 29 %** et laisse celles de fréquence à 4 %, ce
qui inverse leur classement. Sur Qwen3-30B-A3B, un taux annoncé de 37,99 %
s'explique à **96 % par la fragmentation de capacité** : l'expert le plus
routé ne pèse que 2,7 fois le moins routé, la distribution est plate. Enfin,
permuter l'ordre temporel d'un même flux fait passer l'écart à l'optimum hors
ligne de 44,9 à 30,8 %.

**Ce qu'on en fait** : la méthode de mesure, avant tout code. `tiering.py`
réserve déjà `expert_cache_bytes` et annonce un taux de succès que rien ne
mesure — sur Coder-Next il vaut 2,9 %, calculé comme un simple rapport de
capacité majoré de 1,3. Cet article dit que ce genre de chiffre ne veut rien
dire tant que la trace n'est pas rejouée dans l'ordre.

## 20. Cache-Aware Joint Router Adaptation

`arXiv:2609.04895` — **écarté pour la méthode, retenu pour l'ordre de grandeur.**

Taux de succès dur de **93,04 %** contre 90,62 %, soit 1,15 à 18 points de
mieux que le meilleur préchargement, et 4,6 à 53 % de trafic en moins. Mais
c'est obtenu en post-entraînant le routeur, quatre époques : hors périmètre,
nous ne réentraînons rien.

Il donne quand même le plafond à viser : un cache d'experts bien fait atteint
environ 90 % sur un modèle à 128 experts routés 8. **Personne dans le relevé
ne donne le chiffre pour 512 experts routés 10**, ce qui est notre cas.

## 21. FreeToken — service MoE sur matériel grand public

`arXiv:2608.16157` — retenu, **code public**, et c'est notre problème exact.

Sur DeepSeek-V4-Flash, 6 experts sur 256 par couche, 13 milliards de
paramètres actifs sur 284, les auteurs écrivent que le tout « tient dans les
32 Go d'une RTX 5090 ». Leur politique partage chaque défaut de cache entre le
remplissage du cache GPU et **l'exécution directe sur processeur**, selon la
bande passante réellement soutenue — c'est notre chemin « stocké en RAM,
calculé là où ça coûte le moins », mais piloté par la mesure au lieu d'un plan
figé et d'une constante fausse.

**Réserve** : le débit exact sur 5090 est dans une figure qui n'a pas été lue
en texte.

## 22. SAEM — gestion d'experts par étape de raisonnement

`arXiv:2608.21614` — retenu pour une raison de régime.

1,33× de débit moyen, 1,54× quand la calibration correspond, sur
Qwen3-30B-A3B, et surtout **en lot unitaire explicitement mesuré**, avec une
section dédiée. C'est le seul du relevé à mesurer notre régime au lieu de le
supposer.

## 23. DraftExpert — auto-spéculation et préchargement

`arXiv:2607.24434` — retenu pour un chiffre, pas pour la méthode.

1,45× sur le décodage, mesuré sur 4090. Le chiffre à garder pour notre pool :
le gain du brouillon tombe de **4,7× en top-1 à 2,7× en top-2 et 2,0× en
top-3**. Élargir la prédiction coûte vite — argument direct contre un pool
généreusement dimensionné.

## 24 et 25. Énergie par jeton

`arXiv:2608.28044` (H100/H200) : de 7,46 à 0,72 J par jeton selon la longueur
de sortie, et le gain du lot 16 sur le lot 1 tombe de 6,31× à 1,17× quand le
contexte passe de 512 à 4096 jetons. Les mélanges d'experts amplifient l'effet.

`arXiv:2608.00008` : le seul sur du matériel grand public, une 4060 Ti sous
Ollama, modèles de 1 à 7 milliards ; le 7B consomme 4,4 fois plus par jeton
que le plus sobre. Travail préliminaire, et loin de nos 80 milliards.

**Conclusion de l'axe** : personne ne mesure notre régime. Une mesure d'énergie
par jeton sur deux cartes bridées à 400 et 275 W serait, à la connaissance du
relevé, la première publiée dans ces conditions. Nous avons déjà la colonne
dans le banc.

## 26. Parcimonie contextuelle par SVD

`arXiv:2603.14110` — retenu sous réserve de nature.

Prédicteurs de parcimonie sans entraînement dédié, mesurés sur **RTX 3090** :
à 90 % de parcimonie des perceptrons, 1,8× sur le décodage de bout en bout, et
**jusqu'à 11,69× en configuration processeur/carte déchargée** — c'est le
chiffre qui nous concerne, puisqu'un exil n'aurait plus à ramener que les
neurones actifs.

**Réserve dirimante pour nous** : ce sont des modèles denses rendus
parcimonieux, pas des mélanges d'experts. Chez nous la parcimonie est déjà
dans le routage, et le gain ne s'additionne pas.

## Écartés du second relevé

* `arXiv:2608.01536` Celty — noyau creux 2,8× sur cuBLAS, mais l'essentiel du
  gain vient d'un cœur SIMT co-conçu : matériel hypothétique.
* `arXiv:2609.03079` LeanStream — mobile, modèles de 7 milliards.
* `arXiv:2608.08081` RotaryQuant — 120 milliards dans 32 Go à 9-19 jetons par
  seconde, mais par quantification du cache à la volée : hors périmètre.

## Ce que les deux relevés disent ensemble

Deux articles, deux angles opposés, une même réponse. Le microbenchmark
Blackwell mesure les cœurs tensoriels à 96-99 % du pic et conclut que le goulot
est ailleurs. SPICE mesure l'autre côté et trouve 73 à 88 % du temps dans le
transfert des experts. **Ce n'est pas le calcul.** C'est cohérent avec la nuit
du 6 au 7 : le planificateur d'acvram se trompait sur un débit et sur un
plancher, jamais sur un noyau.

## Troisième relevé — 8 septembre 2026, après la série Q3N

Recherche ciblée sur les chantiers ouverts par la journée : quantification
des architectures hybrides, état récurrent, recouvrement transfert/calcul,
tables de quantiles sous 4 bits.

### 27. Why Gated DeltaNet Survives 4-Bit Quantization (arXiv:2609.04098)

**Le plus directement actionnable, et il contredit notre prudence.**
Qwen3.8-27B — 48 couches GDN, 16 couches d'attention, c'est-à-dire la
famille exacte de notre parc. Les quantifications 4 bits communautaires
laissaient le bloc GDN en 8 ou 16 bits, « surtout ses portes de
décroissance et d'écriture », sur l'intuition qu'une récurrence accumule
les erreurs — **exactement le raisonnement qui nous a fait poser les
planchers int8 des v0.4.93-94**. Les auteurs testent l'intuition et la
réfutent : NVFP4 W4A4 sur les **496 couches linéaires, GDN comprise**,
égale BF16 dans le bruit de graine (moyenne −0,52 sur cinq tâches),
pour le plus petit modèle (17,5 Gio) et le préremplissage le plus rapide
(+14 à 19 %) de leur comparatif ; l'écart de perplexité à 32K **se
réduit** avec la position. Mécanisme invoqué : le facteur d'échelle par
bloc de **16 éléments** de NVFP4 localise les aberrants du flux
résiduel.
**Ce que ça change pour nous** : nos planchers int8 sur `linear_attn.*`,
`self_attn.*` et `lm_head` ont été posés le matin du 8/09 contre un
symptôme dont la vraie cause (le drapeau `gdn_a_log_negexp` perdu au
transport) a été trouvée le soir. Ils coûtent donc probablement des bits
pour rien. Mesure à faire : reconvertir sans plancher, sur le moteur
corrigé, et comparer. Réserve : notre q3n utilise des blocs de **32**,
pas 16 — leur mécanisme d'absorption des aberrants ne se transpose pas
tel quel, et c'est peut-être une raison de mesurer le bloc 16 malgré son
surcoût de 0,25 bit.

### 28. DAMP — Decay-Aware Mixed-Precision Recurrent-State Quantization (arXiv:2608.27513)

Premiers à étudier la quantification post-entraînement de l'**état
récurrent** (et non des poids) des modèles à GDN et KDA. Constat qui nous
concerne directement : ces états sont couramment stockés en **FP32**,
consomment beaucoup de mémoire, et leurs mises à jour sont **bornées par
la bande passante mémoire** — elles pèsent donc sur la latence de
décodage. La quantification uniforme échoue ; il faut tenir compte de la
décroissance. **C'est notre cas exact** : `kda.py` tient l'état, les
convolutions causales, les portes et la décroissance en float32.
Gain attendu double, mémoire et débit, sur le poste qui gouverne le
décodage.

### 29. DAK — Direct-Access-Enabled GPU Memory Offloading (arXiv:2604.26074)

**Contredit frontalement notre stratégie de préchargement.** Les cadres
d'exil existants préchargent vers la HBM ; les auteurs montrent que
donner au GPU un **accès direct** à la mémoire distante fait mieux, en
atteignant la bande passante agrégée optimale — le préchargement crée de
la contention HBM, gaspille de la capacité et fabrique des bulles de
pipeline. Mécanisme : détourner le Tensor Memory Accelerator pour aller
chercher poids et cache KV directement en mémoire partagée, plus un
algorithme glouton qui fixe le taux d'exil par opération.
À confronter à notre mesure : bus à 5,7 Go/s effectifs sur 18,7, cartes
muettes une seconde sur deux. Le TMA existe sur Blackwell (5090) mais pas
sur la 3080 Ti — l'asymétrie de notre parc est ici un sujet.

### 30. ChunkFlow — préchargement chunké conscient de la communication (arXiv:2605.11335)

Recouvrement transfert/calcul par morceaux pour l'exil par couches.
Complémentaire de DAK : ce que l'on peut faire sans TMA.

### 31. Scaled Outer Product (arXiv:2605.14929)

Quantification par couche avec **paires de dictionnaires fixes et
dynamiques sélectionnées par un bit par bloc**, échelles signées par
bloc, sélection par cosinus pondéré par les activations, promotion des
couches sensibles par sac à dos multi-choix. 4,5-6 bits par poids quasi
sans perte sur matériel à décodage par table.
Directement parlant pour q3n : nous venons d'implémenter la **table par
manifeste** ; ce papier fait un cran plus fin, une table par bloc choisie
par un bit — et il formalise la promotion des couches sensibles, notre
`snr_floor` réparé du jour.

### 32. Optimal Post-Training Quantization Scales and Where to Find Them (arXiv:2606.10890)

Notre angle mort déclaré du 7/09 (« choix des échelles ») traité de
front. À lire avant de retoucher la table Lloyd.

**Ce que ce relevé dit de nos priorités** : le premier article remet en
cause un choix que nous venons de faire par prudence, le deuxième ouvre
un gisement mémoire et débit que nous n'avions pas vu (l'état récurrent
en FP32), le troisième conteste la stratégie même du préchargement sur
laquelle repose notre chantier « recouvrement ». Aucun ne se transpose
sans mesure : tous portent sur du matériel ou des tailles de bloc
différents des nôtres.

## Relevés croisés des trois sessions — 8 septembre 2026, soir

Domaines répartis : hybrides et récurrence (session OnePlus, fiches dans
`docs/VEILLE-HYBRIDES.md`), quantification et mesure (session de mesure),
exil et matériel (ici).

### 33. AAAC — dictionnaires adaptatifs conscients des activations (arXiv:2605.08692)

Deux petites tables apprises par couche (64 octets), chaque groupe
choisissant celle qui minimise l'erreur **pondérée par les activations**,
le choix encodé dans le **bit de signe inutilisé** de l'échelle du
groupe : **zéro octet de surcoût**. Calibration 3 à 30 minutes.
Ce que ça dit de notre table par manifeste : bonne direction, mauvais
grain (le modèle au lieu du groupe) et mauvais critère (statistique des
poids au lieu de l'erreur en sortie) — la même différence qu'entre un SNR
de poids et une perplexité. Et le code sur huit inutilisé de notre table
à sept niveaux est en fait un **canal libre** : deux tables sélectionnables
par groupe y tiendraient, à 3,25 bits par poids inchangés.

### 34. ActQuant — l'échelle domine la table (arXiv:2605.24011)

« Le choix de l'ÉCHELLE, et non la conception de la table, domine sous
quatre bits. » Ablation à ~2,6 bits : naïf 4,5 % de réussite, **+ échelle
pondérée par la magnitude : 57,7 %** — l'échelle vaut treize fois le
reste de leur pile.
**Nous avons optimisé le mauvais paramètre toute la journée.** Notre
échelle est `amax(bloc) / amax(tenseur)` arrondie en FP8 : la plus naïve
possible, fixée par la valeur la moins représentative du bloc. Réserve :
leur mesure porte sur un modèle vision-langage-action à 2,6 bits, le
mécanisme se transpose, l'amplitude non. Mesure qui tranche, en float64
sans carte : à table identique, échelle actuelle contre échelle
optimisée sur une grille autour d'amax, sur les 288 tenseurs
d'évaluation disjoints. **Si le gain dépasse les +3,43 dB de la table,
l'ordre des priorités s'inverse et rien ne se grave avant.**

### 35. L'axe énergie n'est plus vide — correction (arXiv:2605.11733 et suivants)

Un article de position demande désormais que les travaux d'inférence
rapportent joules/jeton, la contrainte active, la puissance corrigée du
PUE et le débit corrigé de l'utilisation ; plus arXiv:2601.22076 « Where
Do the Joules Go? », 2607.26571 (estimation analytique), 2608.25096
(énergie selon la longueur de contexte). L'axe reste bon mais il a des
**conventions naissantes** : s'y conformer plutôt qu'inventer un format.
**Alerte qui conditionne toute communication** : la littérature situe le
coût typique autour de **1,8 J par jeton** ; nous mesurons **17,4 J brut,
9,6 net** — dix fois plus. Un modèle de 80 milliards de paramètres exilé
sur des cartes grand public bridées contre des services optimisés en
centre de données n'est pas comparable tel quel, mais **tant que l'écart
n'est pas décomposé, publier nos jetons/kJ nous exposerait à être
comparés à des chiffres qui ne mesurent pas la même chose** — le piège
inter-instruments transposé à la publication.

### 36. La divergence entre moteurs est documentée (arXiv:2605.19537)

Cinq moteurs contre `transformers` : jusqu'à **16,3 points d'écart** sur
GSM8K, DeepSeek R1 à 78,1 % chez transformers contre 61,7 % chez Ollama,
écart persistant en échantillonnage stochastique — donc au niveau des
logits. **Un chiffre de qualité mesuré sur un seul moteur décrit le
couple modèle-moteur, pas le modèle.** Valide la journée entière et la
généralise : rejouer l'étalon externe à chaque changement de MOTEUR, pas
seulement de format.

### 37. Contradiction non tranchée sur la propagation de l'erreur

arXiv:2504.09629 annonce une croissance **exponentielle** de l'erreur de
quantification avec la profondeur ; notre mesure du soir donne **√N**
(1,87e−03 à 5,50e−03 de 1 à 8 couches, ×2,93 pour ×8, √8 = 2,83), témoin
dense identique. Les deux ne parlent probablement pas du même objet — eux
quantifient séquentiellement, chaque couche absorbant l'erreur de la
précédente ; nous perturbons indépendamment avec résiduel. **Non
vérifié**, et la mesure qui trancherait est écrite plutôt que
l'explication qui arrange.

### Convergence des trois relevés sur une décision

arXiv:2609.04098 (4 bits sur GDN égale BF16) et notre mesure du 2,3×
disent la même chose par deux voies : la sensibilité des couches
récurrentes existe mais **ne s'amplifie pas avec la profondeur**. Les
planchers int8 des v0.4.93-94, posés le matin contre un symptôme dont la
cause réelle a été trouvée le soir, sont donc à retirer — et c'est
mesurable maintenant que le moteur est sain. Aucun travail publié ne
compare une couche récurrente à un perceptron de même largeur : notre
2,3× semble sans équivalent.
