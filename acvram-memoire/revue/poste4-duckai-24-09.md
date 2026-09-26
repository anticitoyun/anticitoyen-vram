# Revue duck.ai — six questions moteur d'inférence (24/09)

Consultation demandée par le chef d'équipe : six questions techniques indépendantes posées via duck.ai (mode Raisonnement) à trois modèles — Gemma 4 31B, gpt-oss 120B, GPT-5.6 Luna — chacun dans une conversation neuve, même message pour les trois.

---

## Question 1 — PPL de gemma-4-31B-it (bf16, BOS en tête, attention eager) : 344805 sur 512 jetons de prose brute

### Gemma 4 31B
Juge la PPL anormalement élevée mais pas totalement hors de propos pour un modèle "it" évalué hors de son gabarit d'entraînement (formats `<start_of_turn>...`). Suggère que le modèle "hallucine structurellement" ou qu'il y a une incohérence BOS/cache KV. Recommande : modèle base (non-it) pour la prose brute, ou gabarit de conversation strict pour la version it ; corpus WikiText-2/C4.

### gpt-oss 120B
Situe les valeurs typiques Gemma-4-it à 10–30 (voire 50–100 en décodage extrême) et attribue l'écart au prompt "bare-bones" (pas de pré-prompt, BOS en tête, attention eager). Recommande gabarit system+user, corpus standard avec stride. **Affirmation à signaler comme probablement erronée** : gpt-oss avance qu'il faut "évaluer le modèle de base plutôt que la version it (qui a été fine-tuned pour l'italien)" — confusion apparente entre "it" = instruction-tuned et "it" = italien.

### GPT-5.6 Luna
Rejette l'idée que ce niveau de PPL soit un ordre de grandeur seulement dégradé : une PPL de 344805 correspond à une cross-entropie ≈ 12,75, alors qu'un modèle correctement exécuté produit une loss de quelques unités même sans gabarit. Pointe en priorité un **bug d'implémentation probable** (mauvais checkpoint/classe AutoModel, chemin text-only de Gemma 4 défectueux, décalage input_ids/labels, BOS dupliqué, attention_mask incorrect, mauvaise dimension de vocabulaire dans les logits) plutôt qu'un simple effet de gabarit manquant. Cite un problème connu du chemin text-only Gemma 4 dans transformers produisant des losses dégénérées (~17,9) même en CPU/FP32/attention eager. Propose un test minimal de diagnostic ("The capital of France is") avant toute mesure de PPL sur corpus.

### Synthèse et contradiction à ne pas lisser
**Désaccord net entre modèles** : Gemma et gpt-oss traitent la valeur mesurée comme une conséquence plausible (bien qu'extrême) de l'absence de gabarit de conversation et du BOS en tête. Luna conteste frontalement cette lecture : un simple défaut de gabarit dégraderait la PPL sans produire une distribution quasi aléatoire, et pointe vers un bug d'implémentation (chemin text-only Gemma 4, alignement logits/labels) comme cause plus probable, avec une référence externe à l'appui. Par ailleurs l'affirmation de gpt-oss sur un modèle "it fine-tuned pour l'italien" est une confusion factuelle à ne pas retenir.

---

## Question 2 — Ordre de réveil d'un verrou flock() Linux, attente de plus d'une heure

### Gemma 4 31B
Pas garanti FIFO. Le phénomène est du "barging"/famine : un processus qui vient de relâcher le verrou est déjà "chaud" (running) et peut le reprendre avant qu'un processus réveillé par le noyau (encore sleeping→running) n'ait le temps de réagir.

### gpt-oss 120B
Pas strictement FIFO ; dépend du scheduler (nice, cgroup), signaux, poll()/select(). Attente >1h possible si le processus est sorti du CPU (cgroup memory-pressure, OOM-killer, migration), en TASK_UNINTERRUPTIBLE (I/O/NFS), priorité basse, ou si le détenteur du verrou est suspendu (SIGSTOP, cgroup.freeze).

### GPT-5.6 Luna
Confirme l'absence de garantie FIFO (référence man7.org : flock() bloque jusqu'à disponibilité mais ne promet pas de servir dans l'ordre d'arrivée). Détaille un ensemble de causes plus large : ordonnanceur, priorités SCHED_*, affinité CPU, préemption, charge, réveils concurrents, sémantique NFS/SMB différente, descripteur dupliqué conservé après fork(), signal interrompant flock() sans retry/timeout, mélange verrous partagés/exclusifs. Cite un test empirique externe (jjb.cc) montrant un ordre d'acquisition très éloigné de l'ordre d'arrivée. Recommande, si l'équité est nécessaire, de ne pas utiliser flock() comme file d'attente mais un arbitre à tickets, une file FIFO explicite (eventfd/pipe/socket) ou LOCK_NB + backoff + timeout + instrumentation (PID, timestamps, durée de détention).

### Synthèse
Convergence des trois modèles : flock() n'est pas FIFO, le "barging" par un processus qui vient de relâcher le verrou est le mécanisme central cité par Gemma et implicitement par les deux autres. Aucune contradiction de fond ; Luna et gpt-oss ajoutent des causes systémiques (cgroup, NFS, signaux) absentes chez Gemma, et Luna seule propose des alternatives d'architecture (arbitre à tickets, file explicite) pour obtenir une équité réelle.

---

## Question 3 — NVFP4 RTN pur (+11% PPL) → recette pour revenir à ~3%

### Gemma 4 31B
Calibration (type GPTQ/AWQ), attention+GDN+tête en FP8 ou INT8 per-channel, échelle "4 sur 6" plutôt que max/6. Coût : FP4→FP8/INT8 double le trafic mémoire des couches concernées ; impact global <1% si limité à tête et normes.

### gpt-oss 120B
Calibration des activations (~200k tokens, min-max ou KL), échelle par groupe "4 bits/6 canaux", têtes d'attention et GDN en FP8/INT8 per-channel (MLP restant en FP4) → typiquement +3% PPL. Coût mémoire chiffré : FP4 ≈0,5 octet/poids, FP8 ≈1 octet/poids, INT8 per-channel ≈1 octet/poids + ≈0,125 octet/échelle par canal ; passage des têtes d'attention en FP8 double leur débit mémoire, impact global limité (~+0,08 Go sur 27B, ~15% des paramètres concernés).

### GPT-5.6 Luna
Recette la plus détaillée : NVFP4 pour MLP/projections non sensibles, attention entière et GDN (projections) en FP8/INT8 per-channel mais **état récurrent et accumulateurs de GDN en FP32** (point absent chez Gemma et gpt-oss, appuyé par une référence à la littérature Gated DeltaNet), tête LM/embeddings en BF16 ou FP8 (jamais NVFP4 en première expérience), RMSNorm/RoPE/gates en BF16/FP32. Nuance le "4 sur 6" : utile mais pas une règle universelle — recommande une recherche de scale (ex. {4/6, 5/6, 1, 1,25}×max) minimisant une erreur pondérée par les activations plutôt qu'une constante fixe. Table de coût mémoire chiffrée : BF16 2,00 o/poids (54 Go pour 27B), FP8/INT8 ≈1,00 o/poids (27 Go), NVFP4 bloc-16+échelle FP8 0,5625 o/poids (≈15,2 Go), NVFP4+couches FP8 mixtes ≈0,60–0,75 o/poids (≈16–20 Go). Insiste : la cible +3% n'est pas garantie par le format seul, dépend du corpus de calibration et du protocole de PPL.

### Synthèse
Les trois convergent sur la recette générale (calibration + attention/GDN/tête en précision plus haute + assouplissement de l'échelle max/6). Pas de contradiction frontale, mais Luna ajoute une précision technique importante que les deux autres omettent : l'état récurrent et les accumulateurs de Gated DeltaNet doivent rester en FP32 indépendamment du reste, sous peine d'instabilité numérique. Les chiffres de coût mémoire (octets/poids) sont cohérents entre gpt-oss et Luna (~0,5–0,56 pour NVFP4, ~1 pour FP8/INT8) ; Gemma reste qualitatif sur ce point.

---

## Question 4 — RTX 5090, décodage batch=8, hybride GDN 27B : 13ms/pas hors lecture des poids vs 16ms total chez un concurrent

### Gemma 4 31B
Surcoût de 3ms attribué au kernel launch overhead et aux lectures/écritures VRAM intermédiaires. Trois fusions : récurrence GDN (état caché + projection linéaire en un noyau), épilogue (norme + activation fusionnées dans le GEMV), glue (somme résiduelle intégrée pour éviter un aller-retour mémoire).

### gpt-oss 120B
Même diagnostic (kernel de récurrence GDN + opérations "glue" lancées séparément). Propose fusion GDN+attention, fusion LayerNorm+residual+projection, kernel "Packed-QKV", Tensor-Core/WMMA bien dimensionné. Note de lecture : gpt-oss formule sa conclusion comme "réduire nos 13ms à 10ms pour éliminer l'écart de 3ms vs le concurrent" — reformulation un peu confuse du sens de la question (nous sommes déjà plus rapides à 13ms), mais la liste de fusions proposée reste pertinente.

### GPT-5.6 Luna
Interprétation similaire (le concurrent a probablement une chaîne mieux fusionnée). Détaille trois priorités : fusion avant GDN (projection/layout, convolution causale, SiLU, calcul Q/K/V, expansion des têtes, decay, normalisation L2 — en un seul noyau, alignée avec les fusions GDN de Megatron Core citées), noyau de récurrence optimisé (état en registres/shared memory, fusion decay/erase/write/update rank-one, vectorisation, réduction des spills de registres), fusion post-GDN (RMSNorm de sortie + activation de gate + reshape). Conclut que le gain viendra de la suppression de plusieurs petits kernels plutôt que d'un seul GEMM plus rapide, et recommande un profilage détaillé (lancements par token, lectures/écritures globales, occupancy, spills).

### Synthèse
Convergence forte sur la nature du problème (surcoût de petits lancements de kernels et de trafic VRAM intermédiaire autour de la récurrence GDN) et sur le type de fusions à appliquer (récurrence GDN, épilogue/normes, glue résiduel). Aucune contradiction de fond ; Luna est la plus opérationnelle (méthode de profilage), gpt-oss reformule la direction de l'écart de façon un peu ambiguë sans changer les recommandations.

---

## Question 5 — Noyau GEMV templaté sur NV : non bit-exact entre NV=1 et NV=2

### Gemma 4 31B
Cause : non-associativité de l'addition flottante, (a+b)+c ≠ a+(b+c), déroulage différent modifiant l'ordre des FMA. Solution : ordre de réduction déterministe (réduction locale par thread puis globale via shared memory, pattern identique) ou accumulateurs en INT32/INT64 (fixed-point) si la plage dynamique le permet, reconversion en flottant à la fin.

### gpt-oss 120B
Même cause. Solution : forcer le mode de précision (`-ftz=true -prec-div=true -prec-sqrt=true`, désactiver fast-math), réductions dans un ordre lexicographique fixe, **accumulateurs en double précision (FP64) même pour des entrées BF16/FP16** (coût ≈+5–10%), ou compensation type Kahan si on reste en FP16/BF16.

### GPT-5.6 Luna
Même cause, formulée avec la distinction FMA vs multiply-then-add. Solution : réduction logique fixe indépendante de NV (chaque thread réduit une partition canonique, arbre de combinaison fixe, même type d'accumulateur sur tous les chemins) ; **accumulateur idéalement en FP32** (pas FP64) ; désactiver les transformations agressives (`--fmad=false` si besoin, éviter `--use_fast_math`), pas d'atomics flottants, mêmes options de compilation sur toutes les variantes. Ajoute une vérification au niveau assembleur (SASS) pour confirmer qu'une instanciation n'utilise pas FFMA quand l'autre utilise FMUL+FADD. Note que si l'on exige à la fois l'identité bit-à-bit entre deux ordres différents ET le débit maximal des deux, les contraintes sont contradictoires — il faut un mode "reproductible" légèrement plus lent.

### Synthèse et contradiction à noter
Accord sur la cause et sur la nécessité d'un ordre de réduction déterministe. **Désaccord chiffré mineur sur le niveau d'accumulateur nécessaire** : gpt-oss recommande une accumulation en FP64 (coût +5–10%) pour garantir l'indépendance vis-à-vis du nombre de threads, tandis que Luna considère qu'un accumulateur FP32 avec arbre de réduction fixe suffit normalement. Gemma reste plus vague (INT32/INT64 fixed-point comme option). À trancher expérimentalement selon la plage dynamique réelle des données du moteur.

---

## Question 6 — Format Marlin (nvfp4) : dépaquetage au prefill à ~45ms/requête sur 27–31B

### Gemma 4 31B
Recommande le GEMM direct sur la disposition Marlin : le format est conçu pour être dépaqueté à la volée dans les registres/shared memory pendant l'exécution du GEMM, ce qui élimine le goulot d'étranglement mémoire du dépaquetage explicite (écriture du tenseur dense puis relecture).

### gpt-oss 120B
Même recommandation. Ajoute une alternative : pré-dépaqueter une seule fois au chargement du modèle (cache persistant), ce qui reporte le coût de 45ms au démarrage uniquement ; et une approche hybride (dépaquetage rapide + GEMM dense pour le prefill court, kernel Marlin direct pour le decode où les poids sont déjà en cache).

### GPT-5.6 Luna
Confirme la préférence pour une GEMM directe sur layout packé plutôt qu'un dépaquetage global séparé (mêmes arguments : écritures mémoire multiples, perte du bénéfice mémoire NVFP4). **Ajoute une nuance absente des deux autres réponses et qui contredit l'idée que "Marlin direct" est simplement la meilleure solution** : pour NVFP4 W4A4 sur architecture Blackwell, Marlin serait surtout une couche de compatibilité, et des backends alternatifs (CUTLASS, FlashInfer, CuTe DSL) exploitant directement les Tensor Cores FP4 peuvent être plus rapides — cite des chiffres de benchmark externes pour un modèle 27B : ≈105,6 tok/s avec Marlin contre 113,5 tok/s avec CUTLASS et 125,9 tok/s avec CuTe DSL en W4A4. Recommandation pratique : GEMM directe NVFP4 via CUTLASS/CuTe DSL/FlashInfer pour prefill long ou batch important, Marlin éventuellement compétitif seulement pour de petites matrices en decode, dépaquetage global séparé à éviter dans tous les cas sauf forme de matrice atypique où le coût est amorti sur beaucoup de requêtes.

### Synthèse et contradiction à noter
Accord unanime sur un point : éviter le dépaquetage global séparé, privilégier une fusion dépaquetage+calcul. **Contradiction/nuance propre à Luna** : alors que Gemma et gpt-oss présentent "GEMM direct sur layout Marlin" comme la solution, Luna relativise Marlin lui-même en le qualifiant de couche de compatibilité potentiellement sous-optimale face à CUTLASS/CuTe DSL/FlashInfer sur Blackwell, avec des chiffres de débit à l'appui. Cette nuance n'a pas été vérifiée indépendamment (chiffres issus d'une source externe citée par Luna, non recoupés ici) et mérite un banc d'essai local avant décision.

---

## Notes générales
- Réponses obtenues en mode Raisonnement pour les trois modèles.
- Luna a mobilisé une recherche externe avec citations (GitHub, man7.org, nvidia.com, arXiv, unsloth.ai) ; ces sources n'ont pas été vérifiées indépendamment dans cette revue et servent d'indices à recouper, pas de preuves.
- Le point le plus significatif pour la suite du travail sur ce dépôt est le désaccord de la question 1 : avant d'interpréter une PPL de 344805 comme un simple effet de gabarit manquant, vérifier le chemin d'inférence text-only et l'alignement logits/labels, comme le suggère Luna.

## Suivi (ordre chef) — vérification directe du ticket cité par Luna (question 1)

* URL exacte obtenue de Luna sur nouvelle relance, PUIS lue directement (page GitHub,
  pas la description de Luna) : https://github.com/huggingface/transformers/issues/46531
  — **réelle, existe, contenu vérifié au mot près, pas inventée.**
* `[Gemma 4] Gemma4UnifiedForConditionalGeneration text-only inference produces degenerate
  output (token repetition collapse)`, ouverte sur `google/gemma-4-12B-it` (12B, pas 31B),
  transformers `5.10.0.dev0` (main, testé 2026-06-08), reproduit CPU/fp32/eager comme MPS —
  cross-entropy ≈17,9 sur "The capital of France is" en texte brut, contre ≈2-3 attendu ;
  onze composants du forward écartés un par un par les auteurs (RoPE par type, masque par
  type, mise à l'échelle de l'attention, embed_scale, softcap final, convention RMSNorm+1,
  layer_scalar, v_norm) — **aucun ne recrée le défaut seul.**
* **résolution documentée dans le fil, ce que le résumé initial de Luna ratait
  (« Closed, sans résolution technique documentée » — FAUX, il y a bien une résolution) :**
  un mainteneur (`zucchini-nlp`) applique le gabarit de conversation
  (`tok.apply_chat_template`) au lieu du texte brut → sortie cohérente immédiatement.
  L'auteur du ticket confirme, referme en acceptant que le texte brut sans gabarit est
  hors distribution pour un modèle `-it`, et annonce basculer son évaluation de PPL vers
  le modèle DE BASE (`google/gemma-4-12B`, non `-it`) — **exactement la même conclusion que
  notre verdict-etalon-hf-avec-bos.md, indépendamment, par un tiers, sur le même symptôme.**
* transformers installé dans notre venv : **5.17.0** (`python -c "import transformers;
  print(transformers.__version__)"`) — postérieur à la fermeture du ticket (5.10.0.dev0,
  06/2026) et à sa résolution ; comme la cause n'était pas un bogue de code mais un usage
  hors gabarit, la question « notre version est-elle touchée » ne se pose plus au sens où
  chef la posait — il n'y a rien à corriger dans transformers, le comportement est celui
  attendu d'un modèle -it sans gabarit, à toute version.
* onze composants écartés dans le ticket incluent RoPE par type et le softcap final — les
  MÊMES catégories que nos propres pistes réfutées à sec dans la priorité 1
  (`verdict-logits-std.md`, `verdict-decode-pur.md`) — convergence supplémentaire,
  vérification k_eq_v/partial_rotary jugée sans objet vu la résolution réelle du ticket
  (pas un bogue interne, donc rien à chercher fichier:ligne côté acvram sur ce point).
* pas la même taille de modèle (12B ici, 31B pour nous) : ticket connexe trouvé et lu,
  `google-deepmind/gemma#622` — « Gemma 4 token repetition collapse during long
  generation — affects both 31b Dense and 26b MoE », **ENCORE OUVERT**, mais décrit un
  phénomène DIFFÉRENT : répétition de jetons en GÉNÉRATION longue, surtout sous sortie
  structurée (JSON contraint), pas une PPL catastrophique en teacher-forcing sur texte brut
  — un accident de génération sous contrainte de grammaire, pas notre défaut de mesure PPL.
  Non pertinent pour notre priorité 1 (protocole différent), gardé ici pour mémoire du dépôt.
* **verdict du suivi : le ticket cité par Luna est réel, pertinent, et son contenu
  confirme — via un tiers, indépendamment — notre propre conclusion (gemma4-it hors gabarit
  de conversation = PPL/loss dégénérée, pas un bogue de calcul). La priorité 1 reste close.
  La seule correction à apporter à la synthèse précédente : le ticket A une résolution
  documentée, contrairement à ce que Luna avait résumé.**
