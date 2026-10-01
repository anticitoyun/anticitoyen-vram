# duck.ai 30/09 — LeapQuant/STEPQuant × états GDN, noyau decode fp32 (ordre chef, verdict poste5)

Sources primaires : arXiv 2609.38166 (LeapQuant) et 2609.38169 (STEPQuant), HTML complet lu
directement sur arxiv.org. 3 modèles duck.ai raisonnement (Gemma 4 31B, gpt-oss 120B, GPT-5.6
Luna), web search activé.

## Q1 — impact mesuré de la quantification de l'état récurrent (GDN/KDA), Mamba2 non couvert

**LeapQuant** (Qwen3.5-9B, Qwen3.5-35B-A3B, Kimi-Linear-48B-A3B-Instruct évalués en qualité ;
Qwen3.8-Flash et GLM-5.3-Flash en 8 couches, EFFICACITÉ SEULEMENT, pas de chiffre qualité
transposable au modèle complet). Ablation Qwen3.5-9B (AIME/LiveCodeBench) : FP32 87,9/64,1 —
INT8 par pas (naïf) 7,1/9,2 (effondré) — INT8 par fenêtre 82,4/60,6 — LeapQuant complet
(compensator tokens) 87,9/64,1 (= FP32). Même BF16 par pas perd 15,8 pt sur AIME. À 6 bits :
LeapQuant 72,4 % moy. contre 29,6-31,9 % pour les meilleures bases, FP32 75,5 %. À 4 bits :
LeapQuant 60,4 % contre MXFP4 9,7 % / TurboQuant 22,9 % ; sur Kimi-Linear à 4 bits, LeapQuant à
moins de 1,1 pt du FP32 sur chaque tâche.

**STEPQuant** (Qwen3.8-27B GDN, Kimi-Linear-48B-A3B-Instruct KDA). Ablation Qwen3.8-27B
(AIME/GPQA/LiveCodeBench, moyenne) : FP32 84,61 — INT4 uniforme 3,97 (effondré) — Q-Mamba 4 bits
7,64 — spatial seul 73,95 — temporel seul 12,87 — STEPQuant 4 bits 84,72 (≥ FP32). À 6 bits :
STEPQuant 84,51 contre FP32 84,61 (quasi identique). Protéger 1,39 % des heads en pivots FP16
suffit à relever fortement AIME à 6 bits — la moyenne d'erreur de quantification n'est pas un
proxy fiable de l'erreur fonctionnelle.

**Accumulation d'erreur** : liée à la durée de vie (lifetime/rétention du head), pas simplement
à la longueur brute du contexte. STEPQuant mesure une corrélation de Spearman ≈ 0,80 entre
demi-vie du head et erreur accumulée sous INT6 uniforme (equation Eₜ = AₜEₜ₋₁ + εₜ, rétention
proche de 1 + direction peu sollicitée par les clés futures = persistance longue). LeapQuant
réduit l'erreur d'état à 64K tokens d'environ 39× (INT8) / 100× (BF16) par quantification en
fenêtre de 16 tokens contre quantification par pas.

**Mamba2** : aucun des deux papiers ne donne d'ablation qualité comparable — ne pas extrapoler
leurs chiffres GDN/KDA à Mamba2 sans mesure séparée.

## Q2 — optimisations au bit du décodage GDN fp32 (FLA/vLLM/SGLang, b=1 à 12)

Documenté : noyau récurrent fusionné pour la mise à jour d'état GDN (1 launch contre ~20
opérations séparées en décomposition primitive — issue de conception GitHub, exemple Qwen3.5-9B
24 couches : 480 000 launches décomposé contre 24 000 fusionné, pour la récurrence seule) ;
état gardé en registres/shared memory pendant le pas (évite les allers-retours HBM) ; lecture/
écriture d'état fusionnées et contiguës au niveau du noyau. Chemins KDA de vLLM : convolution
causale + récurrence + RMSNorm gated fusionnées en une seule launch, état mis à jour en place.

**Non documenté comme généralité** : fusion des projections x→qkv dans le même noyau que la
récurrence (FLA garde les projections en amont, séparées du noyau `fused_recurrent_delta_rule`)
— ne pas supposer cette fusion par défaut. Nombre exact de blocs/launches pour b=1…12 : aucune
table publique fiable — dépend du backend (Triton/CUDA/FlashInfer), du sharding tensor-parallèle,
de dk/dv, de la présence de MTP/speculative decoding, de la capture CUDA Graph. Bitwise-identité
après fusion : NON garantie en général — ordre de réduction et placement des conversions peuvent
changer les derniers bits même en fp32 pur (cf. vLLM IsoExec, alignement bit-à-bit prefill/decode
sous contrat d'exécution explicite).

Sources citées et vérifiables : `flash-linear-attention/fla/layers/delta_net.py` (GitHub, chemin
`fused_recurrent`) ; issue de proposition GitHub « Operators for Linear Attention/Recurrent
State » (chiffres 20 vs 1 launch) ; issue vLLM « [Bug]: GDN MTP fused decode kernel » ; blog vLLM
IsoExec (`vllm-project.github.io/_posts/2026-07-27-k3.md`) ; issue SGLang « Triton GDN kernel
produces garbled text » (bug qualité SGLang vs vLLM correct sur Qwen3.5-9B — le choix de backend
est aussi une variable de correction, pas seulement de performance) ; notes de version SGLang
(noyaux FlashInfer GDN Qwen3.5 sur Blackwell, Piecewise CUDA Graph).

## Désaccord signalé

**gpt-oss écarté** (confabulation, même défaut que les lots précédents) : invente « GLM-4-X »
(le papier dit GLM-5.3-Flash), un gain « 1,47× » et un « 2,05-3,70× kernel-level speed-up » absents
du papier (le papier donne 1,18-1,57× selon modèle/longueur/batch, jamais ces valeurs), une issue
GitHub « fuse kv projections for DeltaNet » et une section README SGLang « Optimised
recurrent-state IO » non retrouvées. Gemma correcte mais moins précise (pas de chiffres
d'ablation). Luna la plus sourcée et la plus nuancée, recoupée avec la lecture directe du HTML
arXiv faite avant la session duck.ai — retenue comme version de référence ci-dessus.

## Pour poste5 (verdict revue/poste5-leap-verdict-30-09.md)

Gain LeapQuant à b ≤ 16 non chiffré dans le papier (seuls b=512 et « plus grand batch qui tient »
sont testés, 1,22-1,57× sur B200/RTX PRO 6000) ; le texte dit explicitement que le gain CROÎT avec
le batch. Extrapolation à b ≤ 16 : gain attendu sous 1,2×, potentiellement proche de 1,0-1,1× —
**seuil de réfutation de poste5 (≤ 1,2×) vraisemblablement atteint**, mais aucun chiffre mesuré
publié à ce batch précis, à vérifier par mesure directe si le levier est jugé prioritaire. Code
LeapQuant non trouvé publié (pas de dépôt identifié à ce jour).

**RESTE** : rien en cours après ce lot. Prochaine reprise : ordre de chef ou repos.
