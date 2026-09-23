# Revue de la pièce 104 v1 (poste5, e1bc9b03, cache KV k8v4) — à sec — 23/09 (poste1)

Points demandés par le chef, avec fichier:ligne, sur e1bc9b03.

**Défauts (2)**
1. **Sous graphes, la vérification spéculative (q_len > 1) n'est jamais capturée en k8v4, sans le dire.**
   * `engine/graphs.py:432-433` : `paged_ok` exige `c0.cfg.dtype == "int8"` ; en k8v4, il vaut False.
   * `graphs.py:638` rend alors `False` sans compter ni nommer le repli (pas de `_eager(...)`). Or `serve` spécule par défaut.
   * Correctif : `c0.cfg.dtype in ("int8", "k8v4")`. Le noyau `paged_attention_k8v4` porte q_len (`acvram_kernels.cu:5797`, `paged_attention_gen`).
2. **La division `amax / 7.f` de l'écrivain V est approchée (`--use_fast_math`, `kernels/__init__.py:381`) : `acvram_kernels.cu:4322`.**
   * Le jumeau torch divise en IEEE (`kv_k8v4.py`, `quantifier_v`), puis arrondit en half. Près d'une frontière d'arrondi half, l'échelle stockée diffère, et le « au bit avec le jumeau » annoncé n'est pas garanti.
   * La ligne suivante (`:4323`) et tous les autres quantificateurs du fichier (`:4454`, `:5212`, `:6323`) emploient `__fdiv_rn`.
   * Correctif : `__fdiv_rn(amax, 7.f)`.

**Vérifié sans défaut**
* **Défaut int8 intact.**
  * `kvcache.py` : toutes les branches nouvelles sont gardées par `cfg.k8v4`. Pour int8, `storage_dim_v = storage_dim`, `torch_dtype_v = torch_dtype`, `echelles_v = ()`. `_quantize_v` et `_dequantize_v` renvoient aux fonctions int8, et `nbytes` est identique pour une V de la forme de K.
  * Noyau : `paged_attention_gen<CANAL, false>` garde le corps d'avant, avec un seul `TORCH_CHECK(vc int8)` en plus. Le chemin fusionné int8 (`kvcache.py:569`) n'est pas touché.
* **Indexation des groupes de V dans le lecteur** (`acvram_kernels.cu`, branche V4 de `paged_attn_partial_kernel`) :
  * `vs[cell·(D/32) + (lane·PER_LANE)/32]` et `vp4[ch >> 1]`, quartet pair en bas : conforme à l'écrivain (`kv_write_k8v4_kernel`) et à `emballer`.
  * Une voie reste dans un seul groupe, puisque 32 % PER_LANE = 0 pour D ≤ 1 024.
* **Échelle half** : arrondie en half AVANT de quantifier, avec un plancher de 2⁻²⁴, dans les deux jumeaux (sauf le point 2).
* **Chemin q_len > 1 en eager** : la branche k8v4 de `kernels/__init__.py:1415-1423` précède les branches canal et Triton (q_len == 1 seulement), et transmet q_len. Le défaut de ce chemin est sous graphes (point 1).
* **Octets dans tiering** : `tiering.py:463-467` passe `fmt` à `kv_bytes_per_token`, qui compte les octets réels (`kv_k8v4.octets_par_jeton_couche` : 808 o contre 1 040 par jeton et par couche sur Coder). `loader._kv_plancher` lit `plan.kv_bytes_per_token`, et `KVCacheConfig.bytes_per_block` compte aussi les octets réels.
