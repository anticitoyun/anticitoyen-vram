# duck.ai 01/10 — MLA chunked prefill (poste5), MTP sur GDN hybride (poste3) — ordre chef

Sources primaires (vLLM, FlashMLA, FlashInfer, SGLang — lues directement) + 1 modèle duck.ai
raisonnement (GPT-5.6 Luna, web search activé, réponse exceptionnellement précise et
auto-critique — deux issues citées vérifiées directement, titres exacts confirmés ; pas de
2e/3e avis jugé nécessaire cette fois, aucune confabulation détectée).

## Q1 — MLA chunked prefill : troncature causale par morceau, pas masque après coup (poste5)

**Confirmé** : vLLM ne calcule pas l'attention MLA contre toutes les clés puis ne masque pas
après coup. Le backend MLA (`vllm/model_executor/layers/attention/mla_attention.py`) découpe
explicitement le CONTEXTE en morceaux bornés par un espace de travail fixe
(`build_mla_chunked_context_metadata`, `chunked_prefill_workspace`) pour contenir la mémoire
(`k_nope = (kv_c @ W_UK).view(Skv, N, P)` exploserait sinon si Skv est grand). Chaque morceau est
attribué exclusivement aux requêtes qu'il couvre (« no chunk contains an empty context span »).

**Fusion en ligne confirmée** : oui, mécanisme log-sum-exp (LSE) courant, même principe que le
softmax en ligne de FlashAttention — `accumulate_mla_context_chunk` « fold one chunk's partial
into the running context partial », avec `attn_softmax_lse` porté et fusionné explicitement entre
morceaux (code source lu directement par nous, confirmé indépendamment par duck.ai avec la
formule exacte : m=max(m1,m2), LSE=m+log(exp(LSE1−m)+exp(LSE2−m)), O pondéré par ces poids).

**Preuve que c'est bien causal/séparé, pas une concaténation dense suivie d'un masque** :
**vLLM issue #27491 « NaN's in MLA with chunked-prefill »** (titre vérifié directement) — un
premier morceau sans clé valide pour certaines positions rend une sortie nulle et LSE=−∞ ;
`merge_attn_states` fusionne ensuite cet état avec l'autre morceau ; la fusion produit NaN. Ce
bug n'aurait aucun sens si tout était concaténé en une seule passe dense avant masquage — il
prouve que des morceaux sont évalués séparément puis combinés par statistiques. **Fermée
« closed as not planned »** — le contournement rapporté (borner `attn_softmax_lse` à −10000) n'est
pas un correctif officiel validé.

**FlashMLA** : confirme le même modèle pour son noyau sparse (`flash_mla_sparse_fwd`) — un
`gather` explicite des seules clés sélectionnées (`focused_kv = kv[indices]`), jamais un calcul
dense suivi d'un masque ; rend `(out, max_logits, lse)`. **PR FlashMLA #177 « fix sign of online
softmax rescaling factors »** : preuve directe que le code porte une logique de rescaling du
softmax en ligne, assez délicate pour provoquer des bogues de signe.

**Gain publié pour le chunking spécifiquement : AUCUN chiffre fiable trouvé.** Ce qui existe :
TFlops bruts de noyau (FlashMLA : jusqu'à 1 350 TFlops sparse prefill B200, 1 460 TFlops noyau
fusionné norm/RoPE/attention/cast) — ne mesure pas le bénéfice système du chunking ; des tableaux
TTFT/latence vLLM/SGLang sans isolation propre de la contribution du chunking seul ; le RFC vLLM
#3130 (objectif système : mélanger prefill coûteux en calcul et decode coûteux en bande passante)
sans tableau avant/après chiffré. **Ne pas citer un pourcentage de gain inventé.**

**Pièges de précision** : réels et documentés, mais **pas de comparaison FP32/TF32 systématique
trouvée**. Ce qui est établi : l'ordre de réduction change entre une passe unique et une fusion
de morceaux ; un morceau sans clé valide produit LSE=−∞ ; la fusion naïve de ce cas avec un autre
morceau produit NaN (#27491) ; le PR FlashMLA #177 montre que les conventions de rescaling sont
suffisamment délicates pour provoquer des erreurs. **Rien trouvé** attribuant un bogue
explicitement à TF32, ni de spécification garantissant que tous les backends accumulent le LSE
de la même façon.

## Q2 — MTP sur modèle hybride GDN sous CUDA graphs (poste3)

**Mécanisme** : pas un seul « cache KV » — les couches pleines gardent un cache K/V classique,
les couches Gated DeltaNet gardent un état récurrent (`state`/`conv_state`/`ssm_state` selon le
backend). Le vecteur caché fourni à MTP doit provenir du buffer de sortie du forward cible au
rang du DERNIER JETON ACCEPTÉ, avec l'état GDN de cette même position synchronisé AVANT le bloc
MTP suivant. vLLM confirme l'architecture hybride (gestionnaire de cache hybride, noyaux Triton
FLA, CUDA graphs activés par défaut pour réduire le surcoût CPU des lancements Triton).

**Aucun bug trouvé qui prouve littéralement « hidden state NUL »** à la tête MTP — mais
**plusieurs bugs réels et documentés de la même famille** (titres de 2 issues vérifiés
directement) :
- **vLLM #24660 « [Qwen3-next] MPT+CG fail »** (titre vérifié) : `pad_for_cudagraph` reçoit le
  nombre total de tokens spéculatifs (num_spec_decodes × 3) qui dépasse la table des tailles de
  graphe capturées → `IndexError`. Bug de SÉLECTION de taille de graphe, pas un hidden nul.
- **vLLM #40880 (suivi #40914)** : MTP + CUDA graphs + TurboQuant produit une **sortie
  dégénérée** sur Qwen3-Next/Qwen3.6 hybride ; chemin `EagleProposer(method="mtp")`/
  `Qwen3_5MTP` ; redevient correct avec `cudagraph_mode="NONE"` ou sans TurboQuant ; l'auteur
  soupçonne une mauvaise lecture de l'état SSM/conv après acceptation (`block[num_accepted-1]`,
  absence d'un équivalent MTP de `spec_decode_src_indices`). **Le cas le plus proche de la
  question posée** — désynchronisation d'état suspectée, pas prouvée nulle.
- **vLLM #34948 « Qwen3.5 CUDA Illegal Memory Access in GDN Kernel »** : plante dans
  `chunk_gated_delta_rule`/`forward_cuda` avec MTP + kernel GDN FlashInfer — avant même qu'une
  sortie MTP valide puisse être consommée.
- **FlashInfer #3329** : `chunk_gated_delta_rule` sous capture/relecture CUDA graph peut se
  bloquer indéfiniment sur un mbarrier avec Qwen3.5-A3B MTP=5 sur H100/SM90 ; ne se reproduit pas
  hors graphe ; le chemin Triton/FLA contourne.
- **SGLang #18590** (suivi) : liste des PR/issues sur les noyaux GDN, le changement de layout
  d'état `[N,HV,K,V]`→`[N,HV,V,K]`, le support MTP/spec-v2, les problèmes de copie d'état SSM/conv
  pendant la spéculation (#12892, conflit avec le radix cache) — traite l'état GDN comme un objet
  distinct à synchroniser, sans bug « hidden nul » explicite dans le texte accessible.
- **SGLang #21696** : capture séparée des graphes CUDA draft/draft-extend sur Qwen3.5 hybride,
  régression de qualité rapportée, pas d'identification claire d'un hidden absent.

**Test de diagnostic recommandé** (pour g9m si cette classe de bug est un jour suspectée) :
comparer à forme/batch identiques CUDA graphs on/off, backend GDN FlashInfer/Triton-FLA, MTP
on/off, et vérifier le hidden state juste avant `Qwen3_5MTP.forward` à l'index `num_accepted-1` —
une différence entre on/off incrimine la capture/relecture des états, entre backends incrimine le
noyau GDN lui-même.

## Désaccord signalé

Aucun — un seul modèle interrogé cette fois, réponse déjà très auto-critique (distingue
explicitement ce qui est prouvé de ce qui est suspecté/non établi à chaque section), deux
citations vérifiées indépendamment (titres exacts confirmés par lecture directe des pages
GitHub).

**RESTE** : rien en cours après ce lot. Prochaine reprise : ordre de chef, ou file de carte
pour la validation e50.3.
