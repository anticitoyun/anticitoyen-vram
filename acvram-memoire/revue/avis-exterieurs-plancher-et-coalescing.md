# Deux questions posées à des IA extérieures — 10/09/2026

Interrogé : **duck.ai**, deux modèles par question (Gemma 4 31B, puis gpt-oss
via « 2e avis », ce dernier avec recherche web activée pour la question 2).

**Deux des trois sites prévus sont inutilisables** : `perplexity.ai` refuse de
répondre sans compte (« Sign up and repeat your request »), et `phind.com`
rend un **404 DEPLOYMENT_NOT_FOUND**. Le site qui « cite ses sources » est
donc précisément celui auquel nous n'avons pas accès.

**Correction apportée avant de poser la question 1** : l'énoncé transmis
disait « de 384 à 1504 blocs ». Mes mesures vont de **1504 à 6016** blocs
(15,36 → 21,47 → 33,73 µs). Poser des chiffres faux aurait rendu une réponse
construite sur eux.

---

# Question 1 — pourquoi le plancher croît avec le nombre de blocs

## Affirmé

**Gemma 4 31B** — cinq noms : *GigaThread Scheduler (GTS) Dispatch Rate*,
*Grid Management Unit (GMU) Throughput*, *SM Resource Arbitration*, *Hardware
Work Queue (HWQ) Latency*, *Wavefront Scheduling Overhead*.

**gpt-oss** — neuf noms, dont *Grid-Level Dispatch Engine (GLDE)*,
*Resident-Block Table Update Logic*, *Launch-Time Resource Allocation Engine
(Rae)*, *GPC-to-SM Work-Queue Arbitration*, plus une liste de métriques à
mesurer.

## Vérifié — et c'est là que la moisson est maigre

**Le seul mécanisme que les deux nomment est le GigaThread Scheduler / taux de
distribution des blocs.** C'est aussi le seul dont le nom corresponde à un
bloc matériel documenté par NVIDIA. Le reste diverge entièrement d'un modèle à
l'autre : « GLDE », « Rae », « Block-Issuing Unit » ne sont pas des noms
NVIDIA — ils sont plausibles et inventés.

**Les métriques proposées par gpt-oss n'existent pas.**
`sm_block_dispatch_latency`, `warp_dispatch_rate`, `giga_thread_sched_rate`,
`gpc_sm_queue_stalls`, `sm_block_swap_latency`, `kernel_completion_latency` :
aucune n'appartient au jeu de compteurs de Nsight Compute, dont les noms
suivent la forme `unité__quantité_qualificateurs.rollup`. **Cela malgré la
consigne explicite de dire « je ne sais pas » plutôt qu'inventer.**

**Une erreur de lecture de nos propres données, commise par les deux** :
Gemma écrit que le plancher « scales linearly with the number of blocks », et
gpt-oss « grows roughly linearly ». **C'est faux et l'énoncé le disait** :
×4 sur les blocs donne ×2,2 sur le temps. Un modèle linéaire prédirait
61 µs à 6016 blocs, nous en mesurons 33,7.

## À vérifier

Une seule piste survit, et elle vaut d'être poursuivie : **le taux de
distribution des blocs par le GigaThread Scheduler**. Ce qu'il faut pour la
tester est un test à nous, pas une métrique fournie : faire varier le nombre
de blocs **à travail total constant** et vérifier si le plancher suit le
nombre de blocs ou le nombre de vagues (`ceil(blocs / (SM × blocs par SM))`).
Un plancher qui suit les **vagues** et non les blocs expliquerait la
croissance sous-linéaire que les deux modèles ont niée.

---

# Question 2 — l'agencement du cache KV et le coalescing

## Affirmé — et les deux se contredisent

**Gemma 4 31B** : vLLM, TensorRT-LLM et FlashInfer utilisent « presque
universellement » `[num_blocks, num_kv_heads, block_size, head_dim]`, avec
`head_dim` **en dimension la plus interne**, faute de quoi le débit est
détruit.

**gpt-oss** (avec recherche web) : vLLM utilise
`(num_blocks, 2, block_size, num_kv_heads, head_size)`, FlashInfer
`(num_blocks, num_kv_heads, block_size, head_dim/x, x)`, TensorRT-LLM
`(num_blocks, block_size, num_kv_heads, head_size)` dit « NHD ».

**Les deux ne peuvent pas être vrais pour le même moteur.** La divergence
porte sur la question même qui nous intéresse : où se trouve la dimension
séquence.

## Vérifié

**La contradiction est instructive, parce qu'elle recouvre un fait réel** :
vLLM n'a pas *un* agencement mais plusieurs selon le backend. Son noyau
PagedAttention historique sépare K et V et **découpe `head_size`** —
`[num_blocks, num_kv_heads, head_size/x, block_size, x]` pour K — tandis que
le backend FlashInfer utilise une forme « NHD » avec K et V ensemble. Le
`head_dim/x, …, x` de gpt-oss pour FlashInfer et le découpage que ma question
citait pour vLLM vont dans ce sens.

**Conséquence directe pour nous, et elle contredit Gemma** : dans l'agencement
K de vLLM, `block_size` — la dimension **séquence** — n'est *pas* la plus
externe, et `head_size` est **découpé en deux morceaux** de part et d'autre.
Autrement dit, la règle « `head_dim` doit être la dimension la plus interne,
sinon tout est cassé » est démentie par le moteur de référence lui-même. Le
critère réel n'est pas l'ordre des dimensions mais **la taille du segment
contigu vu par un fil** : `x` est choisi pour que chaque fil charge
16 octets d'un coup.

**Métriques ncu — un modèle donne du réel, l'autre invente en jurant de ne pas
inventer.** Gemma cite `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` et
`smsp__sass_inst_executed_op_global_ld.sum` : **ces deux-là existent**, et
leur rapport est bien une mesure de coalescing. gpt-oss cite
`l1tex__t_requests_per_actual_request_avg`, `dram__sectors_per_request_avg` et
`gld_transactions_per_request_avg` en concluant « **No invented metric names**
» : le troisième est un compteur **nvprof** hérité, absent de `ncu`, et les
deux premiers n'existent sous aucune forme. La métrique directe qui manque aux
deux est `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`.

## VÉRIFIÉ CHEZ NOUS : la piste est réfutée, et sans toucher au GPU

`acvram/memory/kvcache.py:336` :

    shape  = (num_blocks, block_size, num_kv_heads, head_dim)
    sshape = (num_blocks, block_size, num_kv_heads)      # une echelle fp16

**`head_dim` est la dimension la plus interne.** Les 128 valeurs int8 d'une
tête pour un jeton donné sont donc **contiguës en mémoire** — exactement ce que
Gemma décrit comme la condition à respecter, et l'agencement « NHD » que
gpt-oss attribue à TensorRT-LLM. Les échelles suivent le même ordre, une par
(bloc, position, tête).

**La prémisse de la question était fausse pour nous** : nos vecteurs de 128 ne
sont pas rangés le long de la séquence. **Le coalescing du cache KV n'explique
donc pas les 172 Go/s**, et cette piste se ferme ici — pour le coût d'une
lecture de quatre lignes, après deux consultations extérieures qui ne pouvaient
pas y répondre puisque aucune ne connaissait notre code.

*Ce qui reste vrai de l'avis extérieur* : la question était bien posée, et
elle méritait d'être tranchée. Une prémisse fausse qu'on vérifie coûte quatre
lignes ; la même, crue sur parole, aurait coûté une réécriture du cache.

## Ce qui reste à mesurer

1. ~~Notre agencement met-il la séquence à l'intérieur ?~~ **Réfuté ci-dessus.**
2. **Mesurer plutôt que raisonner** :
   `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_inst_executed_op_global_ld.sum`
   sur `paged_attn_partial_kernel`. Le rapport dit combien de secteurs de
   32 octets chaque instruction de chargement coûte réellement ; 4 pour un
   chargement de 16 octets par fil parfaitement coalescé, davantage sinon.
3. **Ne pas conclure du chiffre seul** : notre noyau reste à 8,6 fois sa borne
   mémoire. Si le coalescing était le coupable, il le serait pour la lecture du
   KV — or c'est précisément la partie qui **ne domine pas** le temps.

---

# Ce que cette consultation apprend sur la consultation elle-même

- **La concordance n'a rien valu, la divergence a tout apporté.** Les deux
  modèles concordent sur « GigaThread Scheduler » et sur « votre plancher est
  linéaire » — le premier est banal, le second est **faux**. Leur désaccord
  sur l'agencement du cache, lui, a mené au fait utile : vLLM a plusieurs
  agencements et le critère n'est pas l'ordre des dimensions.
- **Une consigne « n'invente pas » ne suffit pas.** gpt-oss a inventé des
  métriques, puis écrit « aucun nom inventé ». La consigne déplace l'invention,
  elle ne la supprime pas.
- **Deux réponses sur quatre citent nos propres chiffres de travers.** Un avis
  extérieur qui reformule mal l'énoncé ne peut pas répondre à la question posée.
