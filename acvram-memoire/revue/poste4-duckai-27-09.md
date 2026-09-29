# Synthèse duck.ai 27/09 — Q28-Q31

3 modèles interrogés séparément par question (nouvelle conversation à chaque fois, mode raisonnement) :
`Gemma 4 31B`, `gpt-oss 120B`, `GPT-5.6 Luna`. Luna a lancé une recherche web sur les 4 questions ; les
deux autres ont répondu à froid (raisonnement interne, pas de sources externes).


## Q28 — vLLM/SGLang prefix caching hybride GDN/Mamba, chunked prefill (poste1, 284c)

gpt-oss, Gemma4 31B et Luna (GPT-5.6, avec recherche web) s'accordent : le prefill est bien coupé aux
frontières de bloc, l'état récurrent GDN/Mamba étant snapshotté et stocké comme extension du KV-cache.
Luna précise, sourcée (vllm.ai, lmcache.ai) : vLLM utilise `mamba_cache_mode=align` — le scheduler
raccourcit un chunk si sa fin franchit une frontière cacheable, mais ne snapshotte que le DERNIER bloc
atteint dans un pas scheduler (pas chaque bloc physique si `max_num_batched_tokens` dépasse un bloc) ;
pas de flush global du batch à chaque frontière. SGLang utilise `MambaRadixCache` (état copié, pas
partagé, car le calcul suivant le modifie en place) ; son roadmap (issue GitHub #12867) liste encore le
découpage/fusion des kernels GDN/KDA/ShortConv comme travail à venir — moins abouti que vLLM sur
l'alignement strict.

**Désaccord** sur la ré-déquantification/projection des poids entre chunks d'un même prefill. gpt-oss
affirme vaguement une forme de réutilisation ("aucun sous-module ne doit refaire son propre passage de
quantisation/projection") sans preuve. Gemma4 et Luna (sourcée) le contredisent tous deux, indépendamment :
NI vLLM NI SGLang ne cachent la projection entre chunks — les poids restent chargés/prépackés (réutilisés
tels quels) mais les projections des NOUVEAUX tokens sont TOUJOURS recalculées ; seule la déquantification
fusionnée dans le GEMM peut varier selon le backend. Ce qui est réellement réutilisé : l'état récurrent
GDN/Mamba et les KV déjà calculés en cas de hit de préfixe — pas les projections elles-mêmes. Non
vérifiable dans le code acvram (`grep -rl vllm.*sglang` sur le dépôt confirme : vLLM/SGLang ne sont pas
vendorés ici, seuls des scripts clients type `outils/banc_decode_vllm.py` existent) — désaccord tranché
en faveur de Gemma+Luna (2/3, dont une source primaire) mais laissé formellement ouvert faute de code
vendoré à inspecter.

## Q29 — Prétraitement multimodal vLLM/SGLang, thread-safety/déterminisme HF+PIL (poste6, 276c)

**Désaccord** net sur la parallélisation par défaut. gpt-oss et Gemma4 affirment sans réserve que
vLLM/SGLang parallélisent le prétraitement entre requêtes d'un même lot. Luna (sourcée vllm.ai) corrige :
dans le chemin OFFLINE (`LLM.generate([...])`) le prétraitement est SÉQUENTIEL malgré le batchage ; dans
le chemin serveur/async, le `ThreadPoolExecutor` du renderer est contrôlé par `--renderer-num-workers`,
**par défaut à 1** — donc pas parallèle sans configuration explicite. Un rapport de perf vLLM (GitHub)
cité par Luna montre même que trop de workers peut RALENTIR (contention GIL/CPU, pas un problème de
sûreté). SGLang parallélise bien son runtime (scheduler, continuous batching, KV cache) mais rien ne
garantit que le prétraitement HF/PIL lui-même soit parallélisé entre requêtes.

**Désaccord** plus tranché encore sur le déterminisme bit-à-bit. gpt-oss affirme catégoriquement que HF
(Qwen2/3-VL) et PIL sont "bit-wise sûrs". Gemma4 est déjà plus prudente (pas de garantie entre versions
fast/slow ou backends). Luna, sourcée (doc HF officielle), tranche : les versions PIL/slow et
torchvision/fast **ne sont PAS bit-identiques entre elles** — HF documente explicitement que le backend
PIL sert à reproduire l'implémentation originale, le backend torchvision étant "moderne et plus rapide"
mais divergent. Piège Qwen signalé par Luna seule, avec source (doc GitHub Qwen3-VL) : double resize si
`qwen-vl-utils` redimensionne déjà l'image avant le processor — nécessite `do_resize=False` côté
processor. Luna signale aussi un piège de sûreté absent des deux autres réponses, avec source (issues
GitHub Pillow) : ne jamais partager une instance `PIL.Image.Image` mutable entre threads (bugs documentés
sur `crop` concurrent) — utiliser `.copy()` après `Image.open()`. Non vérifiable dans acvram (Qwen-VL et
vLLM/SGLang non vendorés) ; gpt-oss est l'outlier le moins fiable sur cette question (affirmation
catégorique non sourcée contredite par les deux autres, dont une sourcée à la doc officielle HF).

## Q30 — Spéculation n-gram, coupure adaptative et hystérésis (poste5, 277e)

**Désaccord tranché.** gpt-oss décrit `speculative_disable_by_batch_size` (vLLM) comme piloté par le
TAUX D'ACCEPTATION, et invente des noms de paramètres SGLang non confirmés
(`speculative_batch_size`/`min_batch`/`max_batch`, non sourcés). Gemma4 le contredit : le paramètre
désactive la spéculation selon la TAILLE DU BATCH (régime memory-bound → compute-bound), pas
l'acceptation — lecture cohérente avec le nom littéral du paramètre. Luna (sourcée vllm.ai + GitHub)
tranche définitivement en faveur de Gemma : *« speculative_disable_by_batch_size ne regarde pas
l'acceptation : il désactive la spéculation pour les nouvelles requêtes quand le nombre de requêtes en
attente dépasse le seuil »*. Exemple de config réel cité par Luna :
`--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":4,"prompt_lookup_max":8}' --speculative-disable-by-batch-size 8`
— `prompt_lookup_min/max` bornent la taille de correspondance n-gram dans le prompt, pas une profondeur
adaptative ; la profondeur est pilotée séparément par `num_speculative_tokens`. Côté SGLang, Luna précise
qu'il n'existe PAS d'équivalent direct au couple `prompt_lookup_min/max` pour un proposeur n-gram — SGLang
expose surtout la spéculation par modèle (EAGLE/EAGLE3 : `--speculative-algorithm`,
`--speculative-draft-model-path`, `--speculative-num-steps`, `--speculative-eagle-topk`), avec des seuils
d'acceptation séparés (`--speculative-accept-threshold-single/-acc`, source GitHub) non équivalents. La
politique d'hystérésis EMA (seuils bas/haut + cooldown) que Luna propose est une recommandation, pas un
mécanisme documenté du code vLLM/SGLang lui-même. Non vérifiable dans acvram : `grep` confirme vLLM/SGLang
non vendorés dans ce dépôt.

**Retenu pour acvram** : `speculative_disable_by_batch_size` = coupe-circuit de charge (taille du batch),
indépendant du taux d'acceptation ; toute logique d'hystérésis sur l'acceptation doit être implémentée en
périphérie (routeur/scheduler externe), pas en s'appuyant sur ce paramètre.

## Q31 — lm-eval-harness, filtres strict-match/flexible-extract de mmlu_flan_cot_fewshot (poste2, 275)

**Désaccord et vérification au code (acvram, réel).** gpt-oss se contredit lui-même dans sa propre
réponse (décrit d'abord flexible-extract comme prenant la PREMIÈRE lettre après "answer", puis affirme
que la pratique reconnue est de prendre la DERNIÈRE — incohérence non résolue). Gemma4 reste vague sur
premier/dernier. Luna (sourcée GitHub EleutherAI/lm-evaluation-harness + arXiv "Finding Answers in
Thought Matters") cite le filtre stock verbatim : `multi_choice_regex` sur `"(\([A-Z]\))"` avec
`group_select: -1`, `take_first` — cherche des options ENTRE PARENTHÈSES, n'exige ni le mot "answer" ni
une position en fin de texte, ce qui contredit gpt-oss ET Gemma qui centraient tous deux le mécanisme sur
le marqueur "answer".

Mais le dépôt acvram a déjà tranché empiriquement, par le fichier réel
`outils/lm_eval_taches/mmlu_v275_high_school_mathematics.yaml` (l. 1-4, commentaire de poste2, pièce
275d) : *« dérivée de mmlu_flan_cot_fewshot_high_school_mathematics (lm-eval stock), SEUL le filter_list
change — le motif d'extraction officiel (`(?<=answer is )(.*)`) ratait la plupart des conclusions de ce
dépôt ("**Answer: (A)**", `\boxed{(C)}`, etc. — pièce 275b) »*. Ce motif officiel documenté ici
(`(?<=answer is )(.*)`) ne correspond NI à la description de gpt-oss/Gemma NI au regex parenthésé cité par
Luna — preuve que la définition exacte du filtre stock varie selon la variante de tâche
(`mmlu_flan_cot_fewshot` vs générique) ou la version de lm-eval-harness, et qu'aucun des trois modèles n'a
la version exacte utilisée par ce dépôt. Le filtre réellement en production dans acvram (l. 15-21 du même
fichier, nommé `get-answer-v275`) :
```
regex_pattern: "(?i)answer[^A-D]{0,25}([A-D])"
group_select: -1
```
puis `uppercase` puis `map` vers `(A)`-`(D)`. C'est le regex "answer + dernière lettre A-D à proximité"
que gpt-oss et Gemma décrivaient — mais construit sur mesure par poste2 après avoir constaté que le stock
officiel ratait la plupart des conclusions de ce dépôt, pas parce que c'est le comportement stock par
défaut.

**Retenu pour acvram** : ne pas faire confiance à la description par un LLM du filtre stock
`mmlu_flan_cot_fewshot` — le filtre réellement utilisé ici est `get-answer-v275`
(`outils/lm_eval_taches/mmlu_v275_*.yaml`), déjà validé empiriquement contre les sorties réelles du dépôt
(275b/275d). Piège confirmé par Luna, absent des deux autres : une conclusion sans parenthèses ("The
answer is B") n'est PAS capturée par le regex stock parenthésé `(\([A-Z]\))` — argument supplémentaire en
faveur du filtre maison qui, lui, ne dépend pas des parenthèses.
