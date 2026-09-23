# Pièce 111 — b12x (cache + MTP) et sglang-rtx5090-mtp, à sec

Ordre chef : pistes 1 et 2 de `recherche-rtx5090-github-gitlab-23-09.md`, la 3 (JustVugg/colibri) sans objet.

## Piste 1 — `local-inference-lab/b12x`, format de cache et fusion MTP contre nos pièces 104/105

**Format de cache — NE CONTREDIT NI NE CONFIRME notre K8V4, géométrie différente.** Le fichier
`b12x/attention/_shared/mla/kv_cache.py` (GitHub, HEAD) décrit des enregistrements pour de la **MLA compressée
façon DeepSeek/GLM** : latent 512 dims + RoPE découplé, trois formats selon le mode — 528 o FP8 (512 × E4M3 en
quatre groupes de 128 + 4 échelles fp32, sans RoPE), 304 o NVFP4 (layout empaqueté 288 o + échelle par jeton),
368 o `nvfp4_ds_mla` (256 o NoPE E2M1 en 32 groupes de 16 + 32 échelles E4M3 + 64 o RoPE E4M3). **Notre K8V4**
(`poste5-piece104-kv-k8v4-23-09.md` § 3, `kvcache.py:276,292,313,323,328,411-418,512-531`, `acvram_kernels.cu:
4175-4260`) vise le **GQA classique de Coder 30B-A3B** (HKV=4, D=128, pas de latent compressé) : K int8 par canal
(bloc 16 jetons), V int4 par groupe de 32 canaux, échelle **half (fp16)**. Les deux projets ne quantifient pas le
même objet — b12x compresse un latent MLA partagé entre têtes, nous quantifions des K/V par tête non compressés.
**Aucune confirmation ni contradiction directe possible** sur le format lui-même.

**Point méthodologique transférable, lui, comparable** : b12x utilise l'**E4M3 (FP4/FP8 flottant) comme codec de
groupe**, jamais l'INT4 uniforme que notre V4 emploie (`kv_canal.py` : « int8 » linéaire, pas flottant). Le choix
FP4 (E2M1) donne une plage dynamique log plutôt que linéaire — exactement le mécanisme que Luna nommait comme
absent chez nous dans la pièce 103 § 5 (« clipping implicite si un outlier élargit la plage du groupe » — un
codec flottant y résiste mieux qu'un uniforme). **Ne contredit pas notre scellé (§ 5 pièce 104) mais nomme une
alternative non testée** si le scellé PPL échoue : passer V4 d'INT4 uniforme à E2M1 (NVFP4) par groupe de 16 ou
32, comme b12x.

**Fusion MTP — porte sur une AUTRE couche que notre pièce 105, pas de recoupement direct.**
`b12x/sequence/mtp_feedback/reference.py` (`feedback()`) fusionne RMSNorm(jeton)·W_embed + RMSNorm(état
multi-flux)·W_hidden pour produire l'entrée BF16 `[T,S,H]` de la couche de décodage MTP elle-même — c'est la
**construction de l'entrée de la tête MTP** (architecture du modèle), un noyau de fusion norme+projection.
**Notre pièce 105** (`speculative.py:464-594`, `MTPProposer`) porte sur la **vérification et l'état du
proposeur** : k passes eager, **tampon d'état caché pour une seule séquence à la fois** (docstring), coût en
graphes CUDA pour q_len > 1, absence de preuve bout-en-bout sur hybride GDN. `b12x/sequence/mtp_feedback/
_preparation.py` (`compile_feedback`, `max_tokens`) dimensionne sa capacité par jetons génériques, pas par
séquence — **ne confirme ni ne contredit** la limite « une séquence à la fois » de notre proposeur, car
`mtp_feedback` est un noyau de calcul (utilisable en lot), pas le gestionnaire d'état de la spéculation ; les deux
projets n'opèrent pas au même niveau de la pile. **Rien à retirer chez nous sur ce point.**

## Piste 2 — `lucacadalora/sglang-rtx5090-mtp` : taux d'acceptation MTP publié

**Chiffre publié, réel** (README, section « What changed » point 1) : **longueur moyenne acceptée 2,57 jetons
sur le banc, 2,85 en conversations réelles**, EAGLE à 3 étapes, top-k 1, **4 jetons de brouillon**, sur
**Qwen3.8-27B NVFP4** avec la tête MTP greffée depuis le checkpoint bf16 (`scripts/graft_mtp.py`), servi par
SGLang v0.5.20 sur une RTX 5090. Corpus : leur propre harnais A/B (`bench/ab_bench.py`) et des conversations en
direct (« live chats ») — **pas un benchmark de code publié** (HumanEval/MBPP non cités), gates fonctionnels
(appels d'outils typés, réflexion, image, rappel long contexte, 20 problèmes d'arithmétique) mais pas de mesure
séparée sur un corpus de code. Lien : `github.com/lucacadalora/sglang-rtx5090-mtp` (commit de tête au 23/09,
README daté du même jour, résultats bruts sous `bench/results`).

Dérivé (non publié tel quel, calculé ici pour comparaison) : 2,57 accepté / 4 proposé ≈ **64 % d'acceptation
moyenne par jeton** sous cette géométrie précise — à ne pas confondre avec un accept-rate MTP1/2/3 par position
comme demandé en pièce 106 (ce chiffre est une longueur moyenne agrégée sur 4 étapes EAGLE, pas une décomposition
position par position).

Note utile en marge : le dépôt liste **b12x explicitement dans ses essais rejetés** (« Tested and rejected » :
« FlashInfer b12x FP4 GEMM for small batches (slower on this card) ») — les deux projets se connaissent et
b12x perd sur ce cas précis (petits lots).

## Verdict § pour chef

Aucune contradiction de notre travail dans les deux pistes ; un levier nommé (codec FP4 au lieu d'INT4 uniforme
pour V4, si le scellé PPL de la 104 échoue) et un chiffre MTP réel à verser en pièce 106 (2,57/2,85 jetons
acceptés, pas un accept-rate par position, pas sur corpus de code isolé).
