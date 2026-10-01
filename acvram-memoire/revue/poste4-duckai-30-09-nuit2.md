# poste4 — duck.ai (30/09 nuit2), 2 questions, 3 modèles de raisonnement

Modèles : gpt-oss 120B, GPT-5.6 Luna, Gemma 4 31B (recherche Web activée
sauf mention contraire). Un chiffre n'est retenu que confirmé par deux
sources indépendantes ; sinon marqué « source unique » ou « désaccord ».

## Q1 — vLLM profile_run/_dummy_run : pic d'activations préremplissage (attention comprise ?), marge, rôle de max_num_batched_tokens

- **Luna** (sourcée GitHub, fichier nommé) : **point clé que gpt-oss et
  Gemma manquent** — `profile_run()` appelle `_dummy_run(max_num_tokens,
  skip_attn=True)` : **l'attention est court-circuitée**, ni les scores
  FA/Triton ni la matrice scores+masque SDPA ne sont mesurés au
  profilage. Le pic retenu = `transient_peak_headroom = torch_peak −
  torch_allocated` après le forward dummy (+ estimation CUDA-Graph
  éventuelle). Marge ajoutée ensuite : **150 MiB fixes** (pas un
  pourcentage). `max_num_batched_tokens` borne seulement le nombre de
  tokens du dummy run, réparti entre requêtes — **ne borne pas un pic
  d'attention réel** puisque celui-ci n'est jamais mesuré ; avec SDPA à
  masque matérialisé, le vrai pic peut donc dépasser ce qui a été
  profilé. Sources : `vllm/v1/worker/gpu/model_runner.py`,
  `gpu_worker.py`, commit 58b2012aa2 (refonte accounting).
- **gpt-oss** : décrit un pic mesuré par `torch.cuda.max_memory_allocated()`
  sans mentionner `skip_attn=True` — **implique à tort que l'attention est
  incluse dans la mesure**. Marge citée : ~10 % du pic (contredit les 150
  MiB fixes de Luna). Citations numérotées sans nom de fichier — plus
  faible.
- **Gemma 4 31B** : même angle mort (`skip_attn`), marge = `gpu_memory_utilization`
  (0,9 par défaut, confond avec le budget global KV, pas une marge de
  sécurité post-profilage). Fichier cité (`model_loader.py:
  profile_num_available_blocks`) **non retrouvé, probablement erroné**.
- **Verdict** : retenir Luna — seule à nommer le mécanisme réel
  (`skip_attn=True`) qui répond directement à la question posée (l'attention
  n'est PAS mesurée, quel que soit le backend) ; gpt-oss et Gemma écartés
  sur ce point précis (ils affirment l'inverse, sans le sourcer
  correctement). Sur `max_num_batched_tokens` : les 3 s'accordent à dire
  qu'il borne le nombre de tokens du dummy run, mais seule Luna tire la
  conséquence correcte (il ne borne donc pas le pic réel d'un backend qui
  matérialiserait les scores).

## Q2 — llama.cpp : dimensionnement du compute buffer (n_ubatch, contexte), échec si insuffisant

- **Luna** (sourcée GitHub `llama-context.cpp`, issues #14836) : `n_ubatch`
  borne le nombre de tokens par passe de graphe (préremplissage découpé en
  micro-lots) ; réservation au pire cas avec `n_tokens = min(n_ctx,
  n_ubatch)`, `n_ctx` pèse surtout sur les tenseurs attention/KV (dépend
  aussi du type d'attention, Flash ou non) ; **pas une simple formule
  n_ubatch × n_ctx**, c'est le pic d'allocations temporaires du graphe par
  backend. Si ça ne tient pas : tentative sans parallélisme pipeline,
  sinon échec d'initialisation (« failed to allocate compute buffers »),
  modèle non démarré ; un agrandissement en cours d'exécution peut échouer
  pareil (OOM).
- **gpt-oss** : **réponse vide/non exploitable** — seulement des cartes de
  sources sans texte de synthèse (GitHub « overallocates speculative
  context », forum NVIDIA). Écarté faute de contenu.
- **Gemma 4 31B** : convergent avec Luna sur le principe (buffer dimensionné
  par `n_ubatch`, découpage en segments, échec = OOM au démarrage), version
  courte et moins précise, sans le détail `min(n_ctx, n_ubatch)` ni la
  distinction attente pipeline/échec définitif.
- **Verdict** : retenir Luna (la plus complète et sourcée) ; gpt-oss écarté
  pour absence de réponse ; Gemma confirme le fond sans apporter de
  nuance supplémentaire.
