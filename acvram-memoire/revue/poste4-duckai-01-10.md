# duck.ai 01/10 — préfill par morceaux non au bit (s1), cache de préfixe hybride (g9m)

Sources primaires (docs.vllm.ai, blog Thinking Machines, GitHub vLLM, lues directement) puis
3 modèles duck.ai raisonnement (Gemma 4 31B, gpt-oss 120B, web search activé). Citations exactes,
sans conclusion à notre place — lecture seule pour poste6/chef.

## Q1 — sources de non-associativité, garanties vLLM/llama.cpp, critère de tolérance

**(a) softmax en ligne/par morceaux** : confirmé en général — l'algorithme en ligne (max courant
+ somme partielle rescalée à chaque étape) est équivalent mathématiquement mais change d'ordre de
réduction selon le découpage. Source générique (blog Thinking Machines), pas de chiffre.

**(b) ordre de réduction des noyaux d'attention selon la forme/le nombre de morceaux** : **confirmé
et nommé précisément** par Thinking Machines, « Defeating Nondeterminism in LLM Inference »
(10/09/2025) : « depending on the inference engine's choices, a sequence might get processed in
several parts (such as in chunked prefill) or all at once [...] it's necessary that the reduction
order for a given token does not depend on how many other tokens from its sequence are being
simultaneously processed. **If you reduce over the K/V values in the KV cache separately from the
K/V values in the current tokens being processed (like in vLLM's Triton attention kernel), this
can't be achieved** » — citant précisément
`vllm/attention/ops/prefix_prefill.py#L36` (commit `0ae43dbf8`). Idem pour FlashDecode/split-KV :
un nombre de splits qui dépend de la longueur de KV casse l'invariance ; la correction nécessite
une « fixed split-size » plutôt qu'un nombre de splits fixe.

**(c) requantification du cache KV int8 par morceau** : **mécanisme plausible mais NON documenté
dans les sources externes trouvées** — le blog Thinking Machines traite des noyaux bf16/fp32, pas
de la quantification du cache. C'est l'hypothèse propre d'poste6 (K/V int8 relus et l'erreur se
compose sur 40-60 couches), pas une cause publiée ailleurs à ce jour.

**vLLM — garantie (réelle, vérifiée)** : `VLLM_BATCH_INVARIANT=1` (docs.vllm.ai/usage/
batch_invariance.html, NVIDIA compute capability ≥ 8.0 ou XPU+Triton) active des noyaux
déterministes pour l'attention et les autres opérations, désactive les optimisations non
déterministes (all-reduce personnalisé en TP, chemin reduce-scatter de la parallélisation de
séquence/async TP), et sur CUDA avec tables matmul réglées tourne sans `torch.compile` via des
CUDA graphs « breakable » (`VLLM_USE_BREAKABLE_CUDAGRAPH=0` pour désactiver). La doc dit
explicitement un coût de performance accepté pour la reproductibilité, **mais ne donne aucun
chiffre de tolérance (Δ logprob, KL)**. Existe aussi une RFC ouverte vLLM #55524 « Mamba2:
exact-replay decode so that prefill, chunked prefill and decode produce identical bits » —
preuve que vLLM lui-même traite ce problème comme NON résolu par défaut pour les modèles
hybrides/récurrents.

**llama.cpp** : aucune source officielle trouvée donnant un seuil de tolérance chiffré ou un
engagement documenté sur l'égalité bit à bit selon `n_ubatch`/`n_batch`. Le phénomène (logits
légèrement affectés par les autres jetons du lot, via la mise à jour du cache KV) est connu de la
communauté (issues/discussions) mais pas formalisé en doc officielle avec un chiffre.

**Critère de tolérance** : **aucun seuil numérique officiel trouvé ni chez vLLM ni chez
llama.cpp**. gpt-oss a cité des seuils précis (1e-5, 1e-4) et des lignes de code
(`ggml-attention.c:420`) **non retrouvés aux sources — écartés, confabulation**.

## Q2 — cache de préfixe automatique (APC) sur modèle hybride sliding+full sans récurrence

**Confirmé, source officielle précise** (docs.vllm.ai, FAQ « How to support models with
interleaving sliding windows », 10/08/2025, citant `google/gemma-2-2b-it` et
`mistralai/Ministral-8B-Instruct-2410` — même famille que gemma-4) :

> « For models with interleaving sliding windows [...] the scheduler will treat the model as a
> full-attention model, i.e., kv-cache of all tokens will not be dropped. This is to make sure
> prefix caching works with these models. Sliding window only appears as a parameter to the
> attention kernel computation. »

Implémentation requise côté vLLM : `config.json` doit porter `layer_types` ; le code modèle doit
analyser la fenêtre par couche et la passer à `per_layer_sliding_window` sur la couche
d'attention (exemple de référence cité : `vllm/model_executor/models/llama.py#L171`, commit
`996357e4808c`). Aucune frontière d'alignement sur un multiple de la fenêtre trouvée ; aucun flag
`VLLM_APC_SLIDING` ni PR #26609/#27433 retrouvés — **ces éléments de gpt-oss écartés,
confabulation** (code, lignes, PR inventés).

**Lien direct avec g9m** : cette doc vLLM confirme exactement le sens du correctif proposé par
poste6 (`est_hybride = spec.couches_recurrentes > 0`, pas `bool(layer_types)`) — l'amont vLLM
traite déjà un hybride sliding+full **sans récurrence** comme un modèle plein (KV jamais
droppé) pour que le cache de préfixe fonctionne ; c'est l'attribution erronée de `est_hybride`
chez nous (confondant « a des `layer_types` » et « a un état récurrent ») qui coupe le préfixe à
la frontière d'instantané et sert 0 jeton du cache.

## Désaccord signalé

**gpt-oss écarté sur plusieurs points** (même défaut que d'habitude) : lignes de code et commits
inventés (`vllm/attention.py` commit `a1f7c9e`, `ggml-attention.c:420`), PR GitHub inventées
(#26609, #27433 citées avec des contenus plausibles mais invérifiables), flags inexistants
(`VLLM_APC_SLIDING`), seuils de tolérance chiffrés non sourcés (1e-5, 1e-4). En revanche son
rappel du flag réel `VLLM_BATCH_INVARIANT=1` et de `VLLM_USE_BREAKABLE_CUDAGRAPH=0` est correct
et recoupé avec la doc officielle. Gemma correcte mais générique, sans citation vérifiable
exacte.

**RESTE** : rien en cours après ce lot. Prochaine reprise : ordre de chef ou repos.
