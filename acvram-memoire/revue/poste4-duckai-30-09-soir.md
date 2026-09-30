# poste4 — duck.ai (30/09 soir), 3 questions, 3 modèles de raisonnement

Modèles : gpt-oss 120B, GPT-5.6 Luna, Gemma 4 31B (recherche Web activée
sauf mention contraire). Un chiffre n'est retenu que confirmé par deux
sources indépendantes ; sinon marqué « source unique » ou « désaccord ».

## Q1 — vLLM chunked prefill : sortie au bit ou à tolérance ?

- **gpt-oss** (source arXiv, Block-Union KV Selection) : pas bit-identical ;
  réduction du softmax par petites requêtes (Q≪KV) puis accumulation entre
  morceaux, contre réduction en un coup côté FlashAttention/Triton ;
  renormalisation entre morceaux = source d'arrondi.
- **Luna** (sources GitHub RFC #3130, vllm.ai, arXiv FLASH-D) : même
  conclusion, plus complète — écarts venant du découpage des tuiles et de
  l'ordre softmax/dénominateur/P·V, du rééchelonnage ; **peuvent franchir
  une frontière de décision et changer un token échantillonné** (rare en
  greedy). Nuance absente chez gpt-oss et Gemma.
- **Gemma 4 31B** (sans recherche Web) : même conclusion générale, non
  sourcée.
- **Verdict** : accord total des 3 sur « non bit-identical, tolérance
  numérique, cause = non-associativité + ordre de réduction du softmax
  en ligne + renormalisation ». Retenir en plus la précision de Luna
  (impact possible sur le token échantillonné, sources RFC #3130 et
  FLASH-D).

## Q2 — KV en anneau (SWA) : recouvrement chunked prefill, cache de préfixe

- **Luna** (sourcée vllm.ai + GitHub) : vLLM garde des blocs SWA distincts,
  libère ceux sortis de la fenêtre ; recherche de préfixe SWA de droite à
  gauche (hit = derniers W−1 tokens), intersectée avec le hit Full quand
  hybride — **pas d'alias circulaire** pour le préfixe SWA (tokens hors
  fenêtre non réutilisables par les couches SWA, mais réutilisables par
  Full). llama.cpp : SWA-cache réellement en anneau, un morceau qui dépasse
  W écrase les positions les plus anciennes (perte, pas de décalage
  partiel) ; le cache de préfixe llama.cpp reste un cache slot/LCP séparé,
  qui retombe sur un préremplissage complet si l'état SWA n'est plus
  reconstructible.
- **gpt-oss** : même mécanique décrite (intersection gauche/droite côté
  vLLM, anneau côté llama.cpp), mais **source mal attribuée** — cite
  vllm.ai pour l'affirmation llama.cpp, ce que Luna source correctement en
  GitHub (llama-server prefix caching, particula.tech). À traiter comme
  moins fiable sur l'attribution, contenu correct par ailleurs.
- **Gemma 4 31B** (sans recherche Web confirmée malgré la mention de
  sources) : cite PagedAttention pour vLLM (correct) et un PR llama.cpp
  #4437 **non vérifié par une recherche Web réelle** — à ne pas citer tel
  quel sans confirmation.
- **Verdict** : retenir la version Luna (sources GitHub/vllm.ai vérifiées) :
  vLLM = SWA sans réutilisation de préfixe hors fenêtre (mais compatible
  Full) ; llama.cpp = anneau strict, écrase et perd, retombe en
  préremplissage complet si besoin. gpt-oss et Gemma écartés sur
  l'attribution des sources, pas sur le fond (convergent).

## Q3 — MTP Qwen3.5/3.6/3.8 : acceptation, gain, vs n-gram, cas nuls

- **Luna** : Qwen3.6-27B/vLLM MTP-6 : +129 % à 2k, +47 % à 8k, **−14 % à
  16k, −51 % à 30,7k** (acceptation décroît avec le contexte, position-1
  0,935→0,721) — source GitHub. Qwen3.8-27B/DGX Spark : MTP ≈3-3,5 tokens
  acceptés/étape contre 3,3-4,9 pour DFlash2 (nvidia.com, non contrôlé).
  Qwen3.5/SGLang : signal négatif/non concluant (bug MI355, gain B200,
  GitHub). Face au n-gram : MTP généralement supérieur sur texte neuf,
  n-gram peut gagner sur code répétitif/boucles d'agents/haut batch.
- **gpt-oss** : Qwen3.5-122B NVFP4 = **0 % acceptation, aucun gain**
  (GitHub, bug report) ; Qwen3.8-27B natif MTP nettement sous DFlash2 ;
  SGLang/DFlash2 sur Qwen3.6-35B : ~2× plus rapide que MTP (modal.com).
  Seuil cité : MTP ne dépasse le n-gram que si l'acceptation par position
  reste > 0,2 en profondeur (vllm.ai/AMD).
- **Gemma 4 31B** (recherche Web confirmée cette fois) : converge sur
  Qwen3.5-122B NVFP4 = 0 % acceptation (même source GitHub que gpt-oss) ;
  Qwen3.8-27B : DFlash2 bat le drafter MTP natif sur toutes les tâches
  testées (5,3-5,5 tokens acceptés, ×3,11-3,43 à concurrence 1) ; à
  concurrence 32, gains de DFlash2 chutent à ×1,01-1,45 et **MTP/DSpark
  repasse sous l'autorégressif (<1,0×)** quand le GPU devient
  compute-bound.
- **Verdict** : les 3 convergent sans contradiction — pas de modèle écarté.
  Cas où la MTP ne rapporte rien (confirmé ≥2 sources) : **contexte long**
  (acceptation qui décroît, Qwen3.6-27B), **haute concurrence/batch
  élevé** (GPU compute-bound, gains <1,0×), **quantification NVFP4 mal
  supportée** (Qwen3.5-122B, 0 % acceptation). DFlash2 (drafter externe)
  bat systématiquement la MTP native sur les mesures publiques trouvées ;
  aucune mesure vLLM/SGLang directe MTP-vs-n-gram en requête unique
  chiffrée n'a été trouvée par aucun des 3 modèles — **absence de
  donnée**, pas un chiffre à inventer.
