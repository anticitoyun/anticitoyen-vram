# Q41 — Marlin dans les roues pip vLLM/SGLang, cache JIT (29/09)

**duck.ai indisponible cette session** (extension Chrome non connectée) : pas de croisement
3 modèles, sources primaires seules ci-dessous. À compléter dès l'extension rétablie.

## Précompilation par architecture

- vLLM : `TORCH_CUDA_ARCH_LIST` pilote les architectures compilées dans le fatbinary ;
  liste recommandée dans les docs `'7.0 7.5 8.0 8.9 9.0a+PTX'`. Le `+PTX` global est ignoré
  pour les extensions générales — seuls certains noyaux internes reçoivent du PTX portable
  ([docs.vllm.ai/gpu](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)).
- Roue précompilée téléchargeable indépendamment de la compilation locale via
  `VLLM_PRECOMPILED_WHEEL_COMMIT` (+ `nightly`) et `VLLM_PRECOMPILED_WHEEL_VARIANT`
  (cu129/cu130/cpu) — [PR #27714](https://github.com/vllm-project/vllm/pull/27714).
- gptq_marlin peut se compiler seul (torch suffit, pas de CMake/vLLM) —
  [hankyul2/my-vllm-marlin](https://github.com/hankyul2/my-vllm-marlin).
- Blackwell (sm_120/121) : upstreaming récent et encore fragile. Un cas documenté :
  noyau natif compile silencieusement pour sm_89 au lieu de sm_120, échec seulement à la
  première requête réelle (« no kernel image is available ») —
  [FreeToken#509](https://github.com/FlashML-org/FreeToken/issues/509). SGLang : Marlin/CUTLASS
  FP8 excluait SM121 à cause d'un bug NamedTuple, corrigé par des PR dédiées SM120/SM121
  ([sglang#28125](https://github.com/sgl-project/sglang/pull/28125),
  [#28137](https://github.com/sgl-project/sglang/pull/28137),
  [vllm#43814](https://github.com/vllm-project/vllm/pull/43814)). Contournement source :
  `TORCH_CUDA_ARCH_LIST="sm_120"` explicite ([discussion sglang#9543](https://github.com/sgl-project/sglang/discussions/9543)).
  4 mois de travail amont sur 6 dépôts (SGLang, FlashInfer, flash-attention, vLLM, CUTLASS,
  Triton) pour le Blackwell grand public — [récit Poppy](https://getpoppy.app/blog/four-months-upstreaming-consumer-blackwell).

## Éviter un cache JIT d'empreinte différente

- FlashInfer (utilisé par vLLM et SGLang) a scindé sa roue JIT-cache monolithique en
  roues **par architecture** (`flashinfer-jit-cache-sm121a`, etc.) — un mismatch entre roue
  provider attendue et présente échoue tard, pas immédiatement
  ([eugr/spark-vllm-docker#415](https://github.com/eugr/spark-vllm-docker/issues/415)).
- Cache JIT persistant sur disque (`FLASHINFER_CACHE_DIR`), survit aux redémarrages ;
  compilation à la première requête par forme unique `(batch_size, num_heads, head_dim,
  page_size)`, 1-3 s par forme — [docs FlashInfer](https://docs.flashinfer.ai/installation.html).
- Bug connu : un module JIT (`topk`) a des flags nvcc **codés en dur** pour sm_89, ignorant
  `FLASHINFER_CUDA_ARCH_LIST` → binaire sm_89 chargé même sur une autre arch si le cache
  n'est pas invalidé par arch — [flashinfer#5355](https://github.com/flashinfer-ai/flashinfer/issues/5355).
  Un autre rapport : même avec jit-cache + cubin présents, FlashInfer retente quand même
  le JIT — [flashinfer#2093](https://github.com/flashinfer-ai/flashinfer/issues/2093).
- Mécanisme de fond (déduit, non sourcé littéralement) : la clé de cache torch JIT
  (`torch.utils.cpp_extension`) inclut normalement hash des sources + flags nvcc, donc un
  changement d'archi cible change la clé — mais les bugs ci-dessus montrent que la pratique
  diverge du principe quand des flags sont codés en dur au lieu de dérivés de l'arch détectée.

## Reste

Repasser duck.ai (3 modèles) dès extension Chrome disponible pour croisement/désaccords.
