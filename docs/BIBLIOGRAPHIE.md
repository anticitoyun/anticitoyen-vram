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
