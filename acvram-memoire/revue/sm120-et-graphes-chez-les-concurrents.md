# sm_120, formats, et politiques de graphes chez vLLM / SGLang

**Consigné par chef le 10/09/2026 à partir des messages de poste8**, parce que
le fichier qu'elle annonçait (`question-1-formats-sm120-v2.md`) n'existait pas et
que `question-1-formats-sm120.md` était resté un plan de recherche vide de
885 octets, daté de 10:59. **Ce qui suit vient de ses messages, pas d'un dossier
qu'elle aurait écrit** : le statut de chaque ligne est donné tel qu'elle l'a
donné, et les réserves sont les miennes.

## Ce qui est utilisable en l'état

**`wgmma` n'existe pas en sm_120.** C'est une instruction Hopper (sm_90).
Conséquence directe : **toutes les techniques « à la Hopper » que la littérature
ou un avis extérieur peuvent nous proposer sont hors sujet pour notre carte.**
Le chemin est celui des `mma` classiques.

**Formats sur sm_120**, annoncés d'après la PTX ISA (table 66) :
- FP8 (e4m3) natif via `.kind::mxf8f6f4`
- FP4 / NVFP4 natif via `.kind::mxf4nvf4`, conteneur 8 bits, dimension K de 32
- INT8 et FP8 au même niveau hiérarchique

## Ce qui reste à trancher — RÉSOLU

**TMA et `tcgen05` sont deux questions indépendantes**, et la première réponse de
poste8 les avait confondues. Résolution par PTX ISA 9.4 :

- **`tcgen05` (5th Generation TensorCore)** : **n'existe PAS en sm_120**  
  Preuve : PTX ISA 9.4 sec 9.7.18.1, ligne 37183 : « On architecture sm_100a/sm_100f, the 5th generation TensorCore's Tensor Memory... »  
  Conclusion : sm_100 uniquement (Blackwell datacenter).

- **TMA (Tensor Memory Accelerator)** : **EXISTE en sm_120**  
  Preuve : `ptxas -arch=sm_120a` accepte `cp.async.bulk.tensor` sans erreur (instruction confirmée supportée).  
  Contrôle positif : même test refuse explicitement `wgmma` en sm_120a (« not supported »), prouvant que l'instrument sait dire non.  
  Conclusion : TMA présent (charge tuiles via moteur dédié, sans threads). **Piste ouverte** : nos noyaux passent temps en lecture poids — TMA peut réduire contention mémoire-thread. À mesurer.

**Conséquence** : les techniques de Hopper (`wgmma`, TMA-basées) sont hors sujet pour notre sm_120.
Le chemin est celui des `tcgen05.mma` classiques quand applicables — mais tcgen05 est absent de sm_120.
Donc `mma.sync` et ses variantes classiques (FP8, FP4) restent la voie.

## Politiques de graphes CUDA (affirmé, non vérifié dans le code)

```
vLLM     ~51 graphes captures, godets de 1 a 512, pas d eviction visible
SGLang   pas de limite a 4 creneaux ; 42 formes MoE = 2,4 Go de graphes
acvram   MAX_GRAPHS = 16, ACVRAM_HYBRID_SLOTS = 4
```

**Nos deux constantes sont spécifiques à acvram.** Mais « personne d'autre ne
l'impose » est un argument pour **réparer**, pas pour **supprimer** : nous avons
mesuré que porter les créneaux à 12 produit du texte faux sur GLM-4.7.

## MLA chez vLLM / SGLang — affirmé par un modèle, PAS vérifié dans le code

Le cache latent MLA y serait **paginé** : pool global
`[num_blocks, block_size, latent_dim]` et une `block_table` par séquence, au lieu
de tampons contigus réservés jusqu'à `max_len`. L'état par séquence passerait par
des **tenseurs d'indexation** (`block_table`, `seq_lens`, `positions`,
`slot_mapping`) remplis par le CPU dans des tampons statiques avant chaque rejeu
de graphe.

Fichiers nommés, **à lire** : `vllm/model_executor/models/deepseek_v2.py`,
`vllm/model_executor/layers/mla.py`, `sglang/srt/layers/radix_attention.py`.

**Réserve de méthode : une réponse d'IA est un nom à chercher, jamais une
conclusion.** GitHub est en lecture pour nous ; trois extraits de code avec
fichier et ligne suffiraient à trancher — la signature du noyau MLA, l'allocation
du cache latent, et la façon dont le graphe reçoit les métadonnées.

## Preuves du code — vLLM v1

**Question 1 : allocation du cache latent (paginé)**

Fichier : `/tmp/vllm/vllm/v1/attention/backends/mla/cutlass_mla.py:189`
```python
_, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape
```

✓ Le cache latent a la forme `[num_blocks, PAGE_SIZE, D_ckv]` — paginé.
`D_ckv = 512 + 64 = 576` (C latent + RoPE).
Chaque bloc couvre `PAGE_SIZE` jetons ; pas de réservation par séquence.

**Question 2 : métadonnées dans le graphe (block_table)**

Fichier : `/tmp/vllm/vllm/model_executor/layers/attention/mla_attention.py:2571`
```python
block_table_tensor=block_table_tensor[:num_decodes, ...],
```

Contexte : cette ligne (dans `_build_decode`) passe un slice du tenseur `block_table_tensor`
global (provenant de `common_attn_metadata.block_table_tensor` à la ligne 2442) à la
construction des métadonnées MLA de décodage. Le tenseur est **rempli par le CPU
avant chaque rejeu** (pas de copie supplémentaire).

Ligne 2405 :
```python
return MLACommonDecodeMetadata(
    block_table=block_table_tensor,
    ...
)
```

✓ La `block_table` MLA est un slice sans-copie du même tenseur utilisé par l'attention classique.
**C'est l'architectural pattern vLLM : une table d'indexation CPU statique, remplie avant chaque rejeu.**

---

## Ce que cette réponse sous-estime, si elle se confirme

poste8 conclut « chantier architectural ». **C'est plus petit que ça, parce que
nous savons déjà le faire** : `paged_attn_partial` et `paged_attn_reduce`
prennent **déjà** une table de blocs et servent tout un lot en deux noyaux. Le
cache latent MLA est `[L, rank + rope]` — un vecteur par jeton, exactement ce
qu'une table de blocs adresse — et son écriture est déjà ponctuelle
(`cache.index_copy_(0, st["len"], k_new)`), donc un `slot_mapping` la remplacerait
comme `kv_write_int8_kernel` le fait déjà.

**L'obstacle n'est pas le noyau, c'est la plomberie** : `mla.py` tient son propre
magasin d'état, séparé de `m.caches` et de l'allocateur, et le chargeur ne pagine
pas les couches MLA — `test_mla_stocke_sans_paginer` le teste explicitement.

Ce n'est donc pas l'importation d'une architecture étrangère, c'est l'application
à MLA de ce que nous faisons déjà à dix lignes de là dans le même fichier.
