# État — pièce 199 (poste2, 25/09, ordre chef) : périmètre Marlin dense/MoE, chiffrage à sec, aucune mesure

instrument : lecture de code + revues existantes (grep/lecture, tokensave insuffisant sur ces symboles français)
commit : origin/main (be837ca1 + 196/198), branche poste2-p199, worktree sans carte
régime : à sec, 0 min de carte
scellé : aucun — état demandé par chef, pas un verdict de mesure
mesuré : rien de nouveau ; chiffres repris des pièces 59, 73, 116, 142, 156, 157, 62
verdict : chiffrage + recommandation (voir ci-dessous)
durée : recherche seule, ~25 min

## (1) Quels MoE sont exclus, et pour quelle raison mesurée

**L'exclusion n'est PAS une liste de modèles nommés** — elle est structurelle : tout modèle qui contient un
`MoEBlock` (détecté par `type(m).__name__.startswith("MoEBlock")`, `acvram/kernels/__init__.py:1176-1178`) est
exclu EN BLOC (denses ET experts) par `interdire_marlin(modele)` (`:1181`, docstring `:1144-1147`), tant que
`ACVRAM_PROJ_MARLIN_PORTEE=denses` (défaut, `:1058`).

**Raison mesurée = deux choses distinctes, pas une seule :**
* **Jamais mesuré, pas rejeté** (`acvram-memoire/revue/poste1-piece142-inventaire-denses-24-09.md:14-17`) :
  l'inventaire à sec d'poste1 (24/09) trouve **32 alias MoE** dont les linéaires HORS experts (attention, GDN, etc.,
  0,01-1,0 Go chacun) seraient AUSSI éligibles si la portée passait à `global` — nemotron_h 30B (7), qwen3_next
  35B-A3B (8), gemma4 26B-A4B (2), qwen3_moe 30B-A3B (6), deepseek_v2/GLM (4), autres. Conclusion de la pièce :
  « la décision ne peut pas être une variable globale en l'état […] il faut une garde modèle dense » — garde qui
  existe maintenant (`sous_moe`, `:1176-1178`), mais **aucun des 32 alias n'a jamais été banché** avec elle activée.
* **Un vrai bogue de justesse trouvé sur le chemin experts** (`acvram-memoire/revue/verdict-157-marlin-sous-normales-24-09.md:3-18`) :
  les échelles NVFP4 sous-normales en format S0E5M3 (plage Marlin, ≈2^14,8) sont mises À ZÉRO au lieu d'être
  représentées — sur Qwen3-Coder-30B, 137 656/1,81 milliard d'échelles touchées, concentrées dans les experts de la
  couche 0 (43-44 experts sur 128) ; erreur mesurée max|Δlogits| = 4,52 sur Qwen3-14B (dense, mais le mécanisme est
  générique). Correctif livré : facteur d'échelle **par ligne** pour les denses, poids encore inexact **exclu et
  nommé** ; « les piles MoE dont un poids écrase restent en disposition naturelle » (`:17-18`) — donc même sous
  `global`, un sous-ensemble d'experts resterait au naturel PAR PILE, avec la raison imprimée, jamais un refus muet.

**Conclusion (1)** : aucun MoE n'est exclu pour une raison de justesse démontrée sur ses linéaires denses — l'exclusion
actuelle est un principe de précaution (mesure manquante), pas un correctif à une régression trouvée sur ces 32 familles.
Le seul bogue réel touche le chemin EXPERTS (moe.py), déjà géré pile par pile via 157, indépendamment de la portée Marlin.

## (2) Ce qui sert aujourd'hui les experts, et leur part du pas de décodage

Chemin servi : `nvfp4_gemv_marlin` (GEMV/GEMM Marlin nvfp4, `acvram/kernels/marlin_port/`), imprimé par
`chemin_moe=…` sur la ligne de régime ; opt-in tensor-core `ACVRAM_MOE_TENSOR=1` (aligneur CUDA porté de vLLM,
`gemm_experts_tensor`) mesuré en b=12 seulement.

* **Qwen3-Coder-30B-A3B-nvfp4(-qkvo-i8c), b=1** (`acvram-memoire/revue/poste3-piece116-nsys-moe-24-09.md:16-19`,
  nsys par couche, 48 couches, 65 pas) : **GEMM experts = 36,0-36,2 % du pas** ; auxiliaires (routage+glue) = 6,4-6,5 %,
  sous le seuil de 10 % — piste de fusion morte ; **reste = 57,4-57,7 %**, dominé par les projections QKVO étroites
  int8 + attention/normes/rope (dense, hors MoE).
* **Qwen3-Coder-30B-A3B, b=12** (le seul B mesuré au-delà de 1) : deux instruments concordants mais pas identiques —
  `acvram-memoire/revue/poste5-ou-sont-les-8-pour-cent-23-09.md:11-20` (GEMV Marlin, avant 62) : GEMM experts
  3,921 ms sur un mur de 7,34 ms/pas (**53,4 %**), projections qkv/o (int8 étroit) 1,345 ms (18,3 %) ;
  `acvram-memoire/revue/poste5-p62-A4-decodage-tensor-23-09.md:7` (après le chemin tensor-core opt-in, non défaut) :
  experts_marlin 3,178 ms sur 6,189 ms de noyaux (51,3 %).
* **Qwen3.5-35B-A3B** : **aucune décomposition par famille de noyaux trouvée** dans les revues (recherche par nom
  d'alias, `chemin_moe`, `experts_marlin` — rien à b=1 ni b=8). Lacune à signaler : pas de profil nsys pour cette
  famille, contrairement à Coder-30B.

**Conclusion (2)** : à b=1, les experts sont MINORITAIRES (36 %) — le dense/attention domine (58 %). À b=8-12, les
deux se rapprochent (51-53 % experts). Le manque de mesure sur Qwen3.5-35B-A3B est réel ; extrapoler depuis Coder-30B
suppose une architecture MoE comparable (nombre d'experts actifs, ratio dense/MoE par couche), non vérifié.

## (3) Gain maximal si Marlin servait les projections denses non-expert de ces MoE

**Mécaniquement, aucun code neuf n'est requis** : dans `preparer_disposition_marlin`, la boucle des candidats
(`acvram/kernels/__init__.py:1187-1189`, `if id(m) in sous_moe: continue`) exclut déjà, TOUJOURS, les sous-modules
d'un `MoEBlock` — donc si `ACVRAM_PROJ_MARLIN_PORTEE=global`, les linéaires hors experts d'un MoE recevraient
naturellement la disposition Marlin, sans toucher les experts. Le SEUL obstacle au chiffrage (1) l'a déjà nommé :
la garde `denses` existe précisément pour EMPÊCHER cette bascule tant qu'aucun des 32 alias n'a été mesuré.

**Chiffrage** : la pièce 156 elle-même mesure le gain Marlin sur des linéaires denses ÉQUIVALENTS (mêmes formes,
modèles purement denses) : **+57 à +90 % de débit à b=8** (`CHANGELOG.md:85-86`, gemma4 31B et Qwen3.8-27B). Si les
projections denses d'un MoE (qkv/o/gate/up/down non-expert) suivent le même ordre de grandeur, et qu'elles pèsent
≈58 % du pas à b=1 sur Coder-30B (dont une fraction seulement est du GEMM — le reste attention/normes/rope, INTOUCHÉ
par Marlin) : le plafond théorique est nettement SOUS 58 % × 90 %. Un repère plus direct et déjà mesuré,
`acvram-ou-sont-les-8-pour-cent` (b=12) : nos projections qkv/o coûtent 1,345 ms/pas contre 0,771 ms chez vLLM pour
la même famille au Marlin 4 bits — un écart de 0,57 ms/pas (**7,8 % du mur de 7,34 ms**), mais cette comparaison
mélange format (int8 W8A16 chez nous contre 4 bits chez eux), donc ce n'est PAS un delta Marlin-vs-actuel propre.

**Obstacles connus, tous nommés dans les revues, aucun n'est un blocage technique** :
1. **Jamais mesuré** (142) — 0 des 32 familles banchées avec `portee=global`, donc aucun chiffre propre par famille.
2. **Justesse** (157) : le correctif par-ligne couvre déjà les denses génériquement, mais n'a été VÉRIFIÉ qu'sur
   des poids denses de modèles denses (Qwen3-14B, qwen32) — pas encore sur les linéaires denses D'UN MoE.
3. **Effet de bord potentiel non mesuré** : `role_marlin` (`:1132-1141`) ne connaît que MLP et GDN.out — un MoE avec
   un rôle différent (routeur dense, tête partagée) ne recevrait pas la disposition « pile » optimisée, seulement la
   disposition seule (moins de gain, non quantifié).

## Recommandation

Ne pas basculer `portee` à `global` sans mesure : le gain plafond crédible est de l'ordre de **2 à 5 % du débit
servi** (pas 57-90 %, cette fourchette est pour la PART dense seule, elle-même minoritaire à b=8-12 sur un MoE), et
deux des trois obstacles (142, 157-denses-de-MoE) sont des lacunes de mesure, pas de code. Proposition de pièce
suivante (bornée) : reprendre le plan déjà écrit par 142 (`:20-24`) — 2 représentants MoE (un `qwen3_moe`, un
`qwen3_next`), ABBA b=1/b=8 sous `ACVRAM_PROJ_MARLIN_PORTEE=global`, KL contre témoins, avant toute décision de
défaut. Qwen3.5-35B-A3B devrait être l'un des deux représentants, puisqu'aucun profil n'existe encore pour lui.
