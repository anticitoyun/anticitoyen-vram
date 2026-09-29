# Q38 — duck.ai, 27/09 (chunked prefill vLLM/SGLang, pour poste1 d19)

Sources primaires lues avant duck.ai : doc vLLM "Optimization and Tuning" (chunked prefill),
doc vLLM "Batch Invariance", issue GitHub `vllm-project/vllm#55524` ("RFC Mamba2 exact-replay decode"),
blog PyTorch/SGLang "Hybrid Models Meet SGLang", doc SGLang `chunked_prefill_size`. 3 modèles duck.ai
(GPT-5.6 Luna avec recherche web, gpt-oss 120B avec recherche web, Gemma 4 31B).

## (1) Identique au bit au prefill d'un seul tenant ?

**vLLM : non, par défaut.** `#55524` le démontre par la mesure : avec le cache SSM en bf16 (défaut),
un split `[256|257]` diverge déjà au token 512, et chaque token après le prochain point de coupure
diffère (88/88 comparés). Cause précise : `_state_passing_fwd` porte l'état inter-morceaux en fp32
**à l'intérieur d'un seul appel** et n'arrondit qu'au stockage ; un appel repris démarre depuis la valeur
déjà arrondie. **Identique au bit uniquement** dans deux cas mesurés : (a) découpages alignés sur la
taille de chunk **avec** `mamba_ssm_cache_dtype=float32` ; (b) mode **exact-replay** (repart de l'état
frontière fp32 et rejoue le morceau partiel par le même chemin SSD), y compris combiné à
`VLLM_BATCH_INVARIANT=1` — vérifié par `torch.equal` sur les logits bruts, `mismatch_count == 0`, sur
plusieurs modèles (mamba2-130m, granite-4.0-h-350m avec attention, Nemotron-H-8B) en TP=1 et TP=2, en
eager et avec CUDA graphs + ordonnancement asynchrone, sous préemption réelle du scheduler.
**Batch-invariance seule ne suffit pas** pour Mamba2 — c'est un mécanisme distinct de l'exact-replay.

**SGLang : pas d'équivalence bit-à-bit établie par une source primaire trouvée**, ni de méthodologie de
test précise (tolérance/KL/bit exact) documentée spécifiquement pour Mamba/GDN. Le blog PyTorch/SGLang
décrit `MambaRadixCache` (match/insert/evict, copie/fork de l'état SSM depuis l'arbre radix) comme un
mécanisme de **continuité fonctionnelle et d'isolation entre requêtes concurrentes**, pas comme un
protocole d'équivalence numérique testé. **Désaccord signalé** : gpt-oss ET Gemma affirment tous deux que
SGLang teste par « tolérance numérique / KL-divergence ≈ 0 » — **aucun des deux ne fournit de source
vérifiable pour cette affirmation précise** (le lien GitHub cité par gpt-oss est une discussion sur le
batching fused prefill+decode, pas un test de KL). GPT-5.6 Luna, recherche web activée, dit explicitement
ne pas trouver cette méthodologie sourcée. **Non retenu tel quel** : à vérifier directement dans le code
de test SGLang (`test/srt/...`) plutôt qu'à citer comme fait établi.

## (2) Tailles de morceau par défaut

| Moteur | Paramètre | Défaut documenté | Note |
|---|---|---|---|
| vLLM V1 (LLM standard) | `max_num_batched_tokens` | **2048** | recommandation officielle : > 8192 pour le débit optimal, surtout petits modèles/grandes GPU ; valeurs plus petites (2048) favorisent l'ITL |
| SGLang | `chunked_prefill_size` | **8192** | `-1` désactive le chunked prefill |

Ces deux défauts ne sont **pas comparables tels quels** : ils reflètent des politiques d'ordonnancement
différentes (vLLM priorise les décodes en attente puis remplit le budget prefill ; le point de départ
"2048" est pensé pour l'ITL, pas pour un débit maximal). Pour une pièce qui **compare** les deux moteurs,
utiliser les valeurs par défaut du serveur **sans les préciser** biaiserait la comparaison — reporter au
minimum : version du moteur, dtype du modèle et du cache SSM, paramètre de chunked-prefill effectif,
mix de batch, cache de préfixe actif ou non, mode exact-replay/batch-invariant actif ou non, et le critère
de comparaison utilisé (`torch.equal`, écart absolu max, KL, ou sorties de tâche). En production, les deux
sont fréquemment réglés au-dessus de leur défaut pour le débit — ne pas présenter "2048 vs 8192" comme
une comparaison intrinsèque des moteurs sans le dire.

## (3) Passage de l'état récurrent d'un morceau à l'autre (Mamba2/GDN)

**vLLM (Mamba2, RFC #55524)** : l'état par séquence et par couche est composé de (a) l'état SSM de
frontière en **fp32**, au dernier multiple de `chunk_size`, et (b) un buffer du morceau partiel depuis
cette frontière contenant les entrées nécessaires au replay : **x, dt, B et C** (pas seulement x/dt/B —
correction apportée par Luna sur ma formulation initiale ; en mode batch-invariant spécifiquement, C
n'est pas bufferisé car il n'entre que dans la ligne de sortie du token lui-même et les lignes rejouées
sont écartées). Coût mesuré : chaque pas de décodage tourne alors les kernels de scan chunké sur le
morceau partiel courant au lieu de la mise à jour à un seul jeton (0,80-1,05 ms contre 0,37-0,40 ms par
couche à batch 64 avec des morceaux partiels de 128 jetons) ; l'état grandit d'environ +43 Mio par
séquence pour un modèle 32 têtes/head_dim 64/dstate 128/27 couches à chunk 256.

**SGLang (`MambaRadixCache`)** : mécanisme de snapshot/restauration via un arbre radix hybride. *Match* :
recherche du plus long préfixe dont le nœud a un état SSM valide, cet état est **copié** (pas partagé
directement, car les mises à jour SSM sont in-place) dans un buffer privé de la requête ; le morceau
suivant reprend depuis cette copie. *Insert* : après un chunked prefill ou une phase de décodage, KV-cache
et état SSM sont insérés dans l'arbre — l'état SSM est copié/forké pour créer un nouveau checkpoint
réutilisable. *Evict* : listes LRU séparées pour KV (éviction feuille→racine obligatoire) et pour les
états SSM (éviction possible depuis n'importe quel nœud). La source décrit précisément *où* l'état circule
(checkpoint de l'arbre → buffer privé → nouveau checkpoint) mais ne dit pas que SGLang garantit par ce
mécanisme une identité bit-à-bit avec un prefill monolithique.

**Point de vigilance pour GDN** : les trois modèles s'accordent — ne pas extrapoler automatiquement le
mécanisme Mamba2 (fp32 boundary + buffer x/dt/B/C) à GDN ou aux autres architectures hybrides ; les
détails d'état et de kernels peuvent différer, à vérifier dans le code exact du modèle GDN concerné.

## Ce qui est solidement établi vs signalé comme faible

Établi par mesure : la non-équivalence par défaut de vLLM (cache bf16), les deux voies documentées pour
l'obtenir (fp32 cache ou exact-replay), le mécanisme précis de MambaRadixCache pour SGLang, les deux
valeurs par défaut de taille de chunk.
Non établi / signalé faible : la méthodologie de test d'équivalence de SGLang pour Mamba/GDN (tolérance
numérique ou KL) — affirmée par deux modèles sans source vérifiable, à vérifier dans le code SGLang avant
de la citer.

**RESTE** : rien en attente de ma part sur Q38 — livré à poste1 et chef.
