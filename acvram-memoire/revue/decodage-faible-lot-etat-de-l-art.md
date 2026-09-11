# Décodage à faible lot sur Blackwell : ce que font les autres

Revue du 10/09/2026, sans carte. Quatre questions posées par chef après nos
gains sur l'attention paginée. **La colonne qui compte est la dernière : ce que
notre montage rend applicable, et pourquoi le reste ne l'est pas.**

Critère de lecture appliqué : *un travail qui annonce un gain sans dire son
régime, son lot et son contexte est classé « non comparable »*.

## Nos chiffres de référence, pour comparer à quelque chose de mesuré

    grille lancee            80 blocs de 4 warps pour 170 SM (releve du matin)
    tranche optimale         64, mesuree sur sept valeurs (16 -> 2048)
    noyau a chunk 64         25,66 us dont 15,36 de plancher, contexte 3007
    debit moteur             +88 % (350 mots) et +104 % (3000 mots)
    borne memoire            le noyau reste a 8,6 fois sa borne apres le gain

---

## 1. FlashAttention-3 / FlashInfer : le découpage par tranche, fait avant nous

**Ce que c'est.** *Split-K*, alias *FlashDecoding* : découper le KV sur la
dimension séquence pour créer du parallélisme quand il n'y a pas assez de
requêtes pour remplir la carte. C'est **exactement notre découpage par
tranche**, sous un autre nom.

**Le défaut qu'ils ont eu, et qui est le nôtre.** L'heuristique par défaut de
FlashAttention-3 *coupe* le découpage sur le seul critère de la longueur de
séquence. En régime peu de têtes KV (MQA, ou GQA sous parallélisme de tenseurs),
**le noyau ne lance que 8 blocs et laisse plus de 90 % des SM inactifs**
(Llopart Font *et al.*, BSC, arXiv:2604.00028, mars 2026, H100 132 SM).

**Leur correctif et son gain.** Une règle qui autorise le découpage quand
l'occupation est basse : **+21 à 24 %** sur l'efficacité du noyau de décodage —
mais **seulement** par le chemin `get_scheduler_metadata()` avec `num_splits`
passé explicitement. Sans métadonnées précalculées, le gain retombe à
**1,00-1,05×**. Régime déclaré : `Batch=1`, `L_K ≤ 512`, `H_KV=1`, `D=128`,
Llama-3.1-70B.

**Applicable chez nous ? Sans objet : déjà fait, et par une autre voie.** Notre
découpage n'a jamais été conditionné à la longueur de séquence, donc le garde-fou
qu'ils lèvent n'existe pas chez nous. Ce qui manquait était le **choix de la
valeur**, que nous avons mesurée au lieu de la déduire.

**Ce qui est comparable, et c'est troublant.** Leur balayage à `Batch=1`,
`L_K=512`, `H_KV=1` donne un **plateau large** avec minima locaux, meilleur à
`s=64` (~11,14 µs), et *moins de 2 % d'écart* entre `s=3` et le meilleur. Chez
nous, à contexte 3007, **32 coûte +23,9 % et 16 coûte +63,1 %** : pas de
plateau, une dégradation franche. Deux explications possibles, non tranchées :
leur contexte est six fois plus court (512 contre 3007), et leur carte a
132 SM contre 170. **La valeur 64 sort des deux côtés — coïncidence à ne pas
sur-interpréter tant qu'on n'a pas mesuré à leur contexte.**

---

## 2. Le coût de la réduction, et jusqu'où découper

**Ce que c'est.** Découper crée des résultats partiels à recombiner : un second
noyau, et une mémoire de travail proportionnelle au nombre de tranches. Chez
nous `part[BQ, HQ, C, D]` plus `pm`/`pl`, et `paged_attn_reduce_kernel` lancé
seulement si `C > 1`.

**Ce que les autres paient.** FlashInfer place les partiels dans un *workspace
buffer* explicite, **taille recommandée 128 Mio**, alloué par l'appelant et
réutilisé — le même tampon sert au plan de l'ordonnanceur dynamique. Le
découpage n'est donc pas gratuit chez eux non plus : il est **budgété d'avance**
plutôt que dimensionné par appel.

**Savent-ils pourquoi trop de découpage dégrade ?** Pas dans ce qui est publié.
Le travail du BSC observe le plateau et **ne l'explique pas** ; il choisit
`s=3` par prudence (« le plus petit qui entre dans le régime »), pas par un
modèle. Notre mesure va plus loin sur ce point précis : **le travail par appel
sature en dessous de 64** — 10,30 µs à 64, 10,31 à 32, identique au centième —
parce qu'une tranche de 32 positions ne remplit plus un bloc, pendant que le
plancher, lui, double. **Nous avons le mécanisme, ils ont le plateau.**

**Applicable chez nous ? Oui, et je l'ai chiffré — la tranche adaptative a un
coût mémoire que personne n'avait regardé.** Notre cache `part/pm/pl` est un
`static std::map` indexé par `(BQ, HQ, C, D, device)`, **jamais vidé**. Chaque
valeur de `C` jamais vue ajoute une entrée définitive de `BQ·HQ·C·D` flottants,
soit 16 384 octets par unité de `C` à `HQ=32, D=128, BQ=1`.

Avant, `chunk` valait 512 : `C = ceil(N/32)`, donc **16 valeurs possibles** pour
une table de 512 blocs. Avec la tranche adaptative, `chunk` vaut 64 jusqu'à
1024 blocs : `C = ceil(N/4)`, donc **128 valeurs**.

    chunk 512 (avant)     16 entrees      2,2 Mo cumules    C max  16
    adaptatif (apres)    128 entrees    135,3 Mo cumules    C max 128

**135 Mo de VRAM au lieu de 2,2**, atteints seulement si le serveur voit assez
de longueurs de contexte différentes — donc **invisible sur un banc court et
progressif en service réel**. Ce n'est pas rédhibitoire à 33,6 Gio, et cela ne
remet pas en cause le gain ; mais c'est une conséquence non annoncée de mon
propre correctif, et elle appartient au dossier. Le remède est celui de
FlashInfer : **un tampon unique dimensionné au pire cas** (`C max`) et
sous-découpé, au lieu d'un cache par forme. Coût : une allocation de 2,1 Mo
fixe. À faire avant que quiconque déploie.

---

## 3. Cache KV en FP8 : le gain existe, mais pas là où on le croit

**Ce que c'est.** vLLM propose le cache KV en FP8 ; Blackwell l'accélère
matériellement. Le nôtre est en **int8 avec une échelle fp16 par vecteur de
128** — même nombre d'octets.

**Le gain mesuré.** Sur Blackwell, la pente d'ITL passe de 4,37e-05 à
2,37e-05 ms/jeton (**54 % de la pente BF16**), l'ordonnée à l'origine ne
bougeant quasiment pas (6,44 → 6,58 ms) — vLLM, billet du 22/04/2026.

**Où il se situe.** Deux causes, pas une : moitié moins de trafic mémoire, **et**
deux fois plus de FLOPS FP8 que BF16 sur Hopper/Blackwell, avec un chemin
FlashInfer qui calcule en FP8 **sans déquantification**.

**Applicable chez nous ? Partiellement, et pas pour la raison attendue.** Le
gain de trafic est **déjà acquis** : notre int8 lit autant d'octets que leur
FP8. Reste la seconde moitié — calculer *dans* le format sans déquantifier. Or
notre noyau déquantifie vers `float` avant l'accumulation. **C'est la piste
réelle, et elle est cohérente avec notre mesure : à 8,6 fois la borne mémoire,
notre coût n'est pas dans les octets lus.** À chiffrer avant d'y toucher : la
comparaison honnête n'est pas « int8 contre FP8 » mais « déquantifier contre
calculer en place ».

---

## 4. MoE à petit lot : la GEMM groupée, et son seuil de rentabilité

**Ce que c'est.** Nous relançons un noyau par couple (expert, jeton). La *grouped
GEMM* exécute plusieurs petites GEMM en **un seul lancement**, en regroupant les
jetons par expert : tri par identifiant d'expert pour rendre contigus les jetons
d'un même expert, puis vues par expert **sans copie dispersée**.

**Ce que ça coûte.** Le tri et la préparation. Et c'est là que se trouve la
réponse à la question posée : **à très petit lot (BS = 1 à 16), le coût de
préparation dégrade les performances** — c'est le régime documenté comme
défavorable, pas comme favorable.

**Applicable chez nous ? Non en l'état, et la raison est chiffrable.** À un jeton
par pas, chaque expert reçoit **un** jeton : le tri n'a rien à trier et le
regroupement ne regroupe rien. La GEMM groupée est rentable quand plusieurs
jetons partagent un expert, donc à partir d'un lot que notre régime de décodage
mono-séquence n'atteint jamais. **La technique qui s'applique à notre cas est
l'autre :** les graphes CUDA, que nous utilisons déjà, et qui suppriment le coût
de lancement sans rien regrouper — ×2,17 mesuré sur un MoE le 9/09.

**Ce qui resterait à examiner** : `cutlass` « Grouped GEMM + Quant » sur
Blackwell (SM100) fusionne la quantification de sortie et le *gating* par ligne
dans la GEMM groupée. Intéressant **si** un jour nous servons plusieurs
séquences ; sans objet à lot 1.

---

## Ce que cette revue change pour nous

1. **Notre découpage n'est pas une nouveauté** — c'est FlashDecoding, connu
   depuis 2023. Ce qui est à nous est la **valeur mesurée** et le **mécanisme de
   sa dégradation**, que la littérature consultée observe sans l'expliquer.
2. **Un défaut de mon propre correctif, chiffré ici** : le cache `part/pm/pl`
   passe de 16 à 128 entrées jamais libérées, soit 135 Mo au lieu de 2,2. À
   remplacer par un tampon unique au pire cas avant tout déploiement.
3. **La piste FP8 est mal posée si on la pose en octets.** Notre int8 lit déjà
   autant. La question est de calculer sans déquantifier.
4. **La GEMM groupée est un contresens à lot 1** et doit être écartée
   explicitement, pour qu'on ne la repropose pas dans trois semaines.

## Sources

- Llopart Font, Hernando, España-Bonet, *Sequence-Aware Split Heuristic to
  Mitigate SM Underutilization in FlashAttention-3 Low-Head-Count Decoding*,
  arXiv:2604.00028v1, 19/03/2026 — https://arxiv.org/html/2604.00028
- *The State of FP8 KV-Cache and Attention Quantization in vLLM*, 22/04/2026 —
  https://vllm.ai/blog/2026-04-22-fp8-kvcache
- FlashInfer, documentation des noyaux d'attention et du workspace —
  https://docs.flashinfer.ai/api/attention.html
- Ye *et al.*, *FlashInfer: Efficient and Customizable Attention Engine*,
  arXiv:2501.01005 — https://arxiv.org/pdf/2501.01005
- *Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM Kernel*,
  PyTorch —
  https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- NVIDIA cuDNN, *Grouped GEMM + Quant – Unified (SM100)* —
  https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/gemm_fusions/grouped_gemm_quant_unified.html
