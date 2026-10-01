# duck.ai 01/10 — 4 questions pour poste6 (d19, g9m retenu) — chunked prefill, SDPA sm_120, réserve d'activations, n-gram b=12

Sources primaires (PyTorch, vLLM, NVIDIA, lues directement) + 1 modèle duck.ai raisonnement
(GPT-5.6 Luna, web search activé, réponse exceptionnellement précise, distingue à chaque
section ce qui est prouvé de ce qui doit être vérifié localement — pas de 2e/3e avis jugé
nécessaire).

## Q1 — borne publiée sur l'écart chunked/seul tenant (contexte : Δ 0,02-0,04 Devstral-24B)

**Aucune borne chiffrée publiée trouvée** chez vLLM, SGLang ou TensorRT-LLM (ni
`max|Δlogp| ≤ ε`, ni `D_KL ≤ ε`). **vLLM documente explicitement l'ABSENCE de garantie de
stabilité des logprobs** (variations possibles selon la taille du lot, l'ordre des opérations
flottantes, le chemin d'exécution) — distinct de la garantie algorithmique du décodage
spéculatif (celle-là est garantie, la stabilité numérique brute ne l'est pas. TensorRT-LLM
présente le chunked prefill comme une optimisation mémoire/débit uniquement (dimensionnement
dynamique des buffers d'activations), sans rapporter d'écart probabiliste.

**Point de comparaison le plus proche trouvé** : une issue SGLang récente mesure un écart de
logprob allant jusqu'à **0,032391** par jeton selon la concurrence et le batching du préfill —
**du même ordre que notre 0,02-0,04**, mais ce n'est pas une borne publiée ni une preuve que le
chunking seul en est responsable (concurrence, batching et backend d'attention varient
simultanément dans cette issue).

**Lecture pour d19** : notre écart (0,02-0,04 contre 0,004 pour le témoin cache-de-préfixe) est
compatible avec du bruit d'ordre flottant mais **trop grand pour être attribué par défaut à une
simple accumulation de bruit** sans isoler séparément : backend d'attention, nombre de splits
internes, type/alignement du masque, taille des morceaux, batching concurrent, CUDA Graphs,
quantification, précision du GEMM. Protocole suggéré pour trancher : comparer à jetons
strictement identiques (seul tenant b=1 / chunké b=1 mêmes morceaux / avec et sans cache de
préfixe), backend forcé, logits en fp32 avant échantillonnage, Δmax et KL calculés séparément —
et noter si l'écart porte sur le top-1 ou sur un jeton peu probable (pas la même portée).

## Q2 — PyTorch 2.14, `causal_lower_right` + `enable_gqa=True` sur sm_120

**Mécanisme de dispatch confirmé par lecture DIRECTE du code source PyTorch** (`torch/nn/
attention/bias.py`, `CausalBias._dispatch`, vérifié sur la branche 2.3, logique stable depuis) :
- `causal_lower_right(q_len, kv_len)` rend un objet `CausalBias` (variant `LOWER_RIGHT`), PAS un
  tenseur de masque matérialisé — exactement le cas d'un préfill par morceaux où q_len < kv_len
  (morceau courant contre cache + morceau).
- Quand `attn_mask` est un `CausalBias`, `F.scaled_dot_product_attention` délègue à
  `CausalBias._dispatch` (protocole `__torch_function__`), qui **reconnaît spécifiquement**
  le cas `LOWER_RIGHT` : il construit `SDPAParams` et appelle **`can_use_flash_attention`** —
  si éligible, il appelle directement `torch.ops.aten._scaled_dot_product_flash_attention` avec
  **`is_causal=True`** (jamais un masque matérialisé passé au noyau flash). Padding automatique
  de la tête si `head_dim % 8 != 0`.
- **GQA** : annonce officielle PyTorch (dev-discuss) confirme **« GQA is supported only by math
  and flash_attention kernels »** — `enable_gqa=True` exclut d'office le backend
  memory-efficient du jeu de candidats, quel que soit le masque.
- **Ce qui N'EST PAS vérifié ici** : la signature exacte de `SDPAParams`/`CausalBias._dispatch`
  sur PyTorch 2.14 précisément (lu sur 2.3 ; l'intégration `enable_gqa` dans CE chemin de
  dispatch précis — pas seulement le chemin `attn_mask=None, is_causal=True` — n'est pas
  confirmée par une lecture directe de la 2.14). **À vérifier sur l'installation réelle**, pas
  à supposer : `inspect.getsource(torch.nn.attention.bias.CausalBias._dispatch)`.
- **sm_120 compilé** : vérifier `torch.cuda.get_arch_list()` doit contenir `sm_120`/`compute_120`
  (ajouté aux builds CUDA 12.8+/nightly ; les anciennes wheels CUDA ne l'avaient pas). Le
  support général CUDA/sm_120 **ne garantit pas** que CE couple précis (LOWER_RIGHT + GQA +
  dtype + head_dim) soit éligible au flash — à vérifier séparément.
- **Vérification à l'exécution, par ordre de fiabilité croissante** :
  1. `can_use_flash_attention(SDPAParams(...), debug=True)` / `can_use_efficient_attention(...)`
     (donne la raison exacte d'inéligibilité si refusé) ;
  2. forcer chaque backend un par un via `sdpa_kernel(SDPBackend.X)` et observer l'erreur ou le
     succès (si flash forcé échoue plutôt que de basculer silencieusement, c'est le signal le
     plus net) ;
  3. **le profiler reste la preuve d'exécution la plus fiable** :
     `torch.profiler.profile(...)`, chercher `aten::_scaled_dot_product_flash_attention` contre
     `..._efficient_attention` / `..._attention_math` dans la table — un noyau Triton/fusionné
     peut porter un autre nom, auquel cas Nsight Systems/Compute.

## Q3 — réserve d'activations de préfill (pas le cache KV) : passe mesurée, pas une formule

**vLLM** : `determine_num_available_blocks()` exécute `profile_run()` avec des entrées factices,
mesure le **pic mémoire PyTorch réellement atteint**, puis `M_KV = M_budget − M_poids −
M_non-torch − M_activation,pic` → `N_blocs = ⌊M_KV / M_bloc⌋`. **Passe mesurée, pas une formule
théorique générale.** Rappel d'une recherche précédente (`poste4-duckai-30-09-nuit2.md`,
non refaite ici) : `_dummy_run` appelle avec **`skip_attn=True`** — l'attention elle-même
(scores + masque) **n'est jamais mesurée** au profilage, le pic retenu vient du reste du forward
(MLP, projections) + 150 MiB de marge fixe. Un préfill de 64k ne réserve donc pas forcément des
activations pour 64k jetons simultanés : avec chunked prefill, la forme critique est souvent la
taille du MORCEAU, pas la longueur logique totale.

**llama.cpp** : pas de formule universelle « octets/jeton » non plus — le compute buffer est
dimensionné sur le graphe ggml effectivement construit (dépend de `n_batch`/`n_ubatch`, de la
forme du graphe, du backend), réalloué si une nouvelle forme l'exige. Le cache KV, lui, croît
avec `n_ctx` par une formule connue (têtes KV × dim × type).

**Ordre de grandeur donné pour le CACHE KV** (pas les activations transitoires, distinction à
garder — poste6 demandait les deux, seul le KV a une formule simple) : pour un dense
`h=5120`, bf16, MHA pleine (toutes les têtes KV) : `KV/jeton = 2·L·h·b` ≈ 0,625 MiB/jeton
(L=32) à 0,78 MiB/jeton (L=40) → **≈ 40-50 GiB à 64k jetons**. Avec GQA à 8 têtes KV (facteur
4 vs 32) : **≈ 10-12,5 GiB à 64k**. Pour une fenêtre glissante (gemma 3/4) : le KV d'une couche
à fenêtre bornée `W` est borné par `min(T, W)` au lieu de croître avec T — réduit fortement la
réserve KV persistante, **mais ne rend pas les activations transitoires gratuites** : un
morceau de taille C exige toujours des buffers proportionnels à C, pas à T. Gemma 3/4 mélange
couches globales et locales (et MoE pour gemma-4) — une seule valeur octets/jeton ne s'applique
pas à toutes les couches.

## Q4 — n-gram à b=12 sur texte libre, gemma-3/4 ou Mistral 24B

**Aucune mesure publique trouvée** combinant précisément les 6 critères demandés (gemma-3/4 OU
Mistral 24B, n-gram/prompt lookup, texte libre, batch 12, taux d'acceptation, gain/perte de
débit). Ce qui existe, hors cible : vLLM documente la spéculation comme surtout utile à
QPS faible/moyen, n-gram avec des gains plus modestes qu'un drafter modèle ; une discussion
utilisateur HF sur Mistral-Small-3.1-24B rapporte ~40 % de gain mais avec un COUPLE
Qwen-32B/Qwen-0.5B (pas du n-gram) ; un test communautaire (dev.to, 2×RTX 5060 Ti) sur
gemma-4-26B-A4B et Qwen3-32B avec n-gram conclut qualitativement que la méthode n'aide pas
toujours sur GPU grand public et que les modèles MoE réduisent l'avantage du lot de
vérification (pas de tableau de taux d'acceptation exploitable) ; un benchmark Nutanix rapporte
~5,5 % (Gemma 2-2B), ~14,5 % (Llama 4 Scout), ~18 % (Mixtral) de gain ITL — ni gemma-3/4, ni
Mistral-24B, ni batch 12.

**Aucun seuil d'acceptation universel trouvé ou défendable.** Le seuil dépend du rapport entre
le coût de proposition+vérification et le gain espéré ; formule de rentabilité donnée :
`E[N] = 1 + Σᵢ ∏ⱼ rⱼ` (jetons produits par vérification), rentable si
`E[N] / (T_propose + T_verify) > 1/T_decode`. À b=12, la vérification est déjà plus proche du
régime limité par le calcul — le gain marginal du lot spéculatif diminue généralement. Recommandé
par Luna (point de départ expérimental, **pas un seuil publié**) : tester K=2,4,6,8, désactiver
n-gram si le débit net baisse sur plusieurs fenêtres — un seuil fixe (40 %, 60 %) n'est pas
défendable sans mesurer les temps de proposition/vérification réels de notre moteur.

## Désaccord signalé

Aucun — un seul modèle interrogé, réponse déjà très structurée en « prouvé / à vérifier / non
trouvé » à chaque section, cohérente avec mes propres lectures directes de code (PyTorch bias.py,
vLLM skip_attn, dev-discuss GQA).

**RESTE** : rien en cours après ce lot. Prochaine reprise : ordre de chef, ou file de carte
pour la validation e50.3.
