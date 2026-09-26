# Pièce 106 — duck.ai sur les têtes MTP (pour la 105 d'poste1, à sec), 23/09

Question posée (identique aux trois modèles, mode raisonnement) : taux d'acceptation publiés des têtes MTP
(MTP1-3) de Qwen3.5/3.6 contre un brouillon n-gram sur du code ; coût de vérification à q_len 2-4, b=1 et b=8 ;
pièges connus (cache KV des têtes MTP, échantillonnage non glouton). Références vérifiables exigées, aucun
chiffre non sourcé.

Extension Chrome déconnectée après Luna, revenue avant gpt-oss/Gemma (2 retentatives) : 3/3 modèles rendus.

## GPT-5.6 Luna (raisonnement, 26s) — seule réponse obtenue

**Constat général de Luna avant tout chiffre** : « il existe beaucoup de chiffres de débit, mais très peu de
mesures publiées donnant séparément l'acceptation de MTP1, MTP2, MTP3 sur un corpus de code, et pratiquement
aucune table complète de coût de vérification selon q_len et le batch. »

### 1. Taux d'acceptation publiés

| Source | Modèle | Corpus | Chiffre publié |
|---|---|---|---|
| FastMTP, arXiv:2509.18362 (`arxiv.org/html/2509.18362v1`) | MTP générique, **pas Qwen** | 7 tâches dont raisonnement et génération de code, pas de ventilation par benchmark dans l'extrait | Vanilla MTP : ≈70 % à k=1, 10 % à k=2, ≈0 à k=3. FastMTP fine-tuné : ≈80 %/56 %/36 %. **Pas des valeurs Qwen3.5/3.6.** |
| Qwen3.6-27B-MTP-pi-tune, model card HF (`huggingface.co/phucngodev/Qwen3.6-27B-MTP`) | Qwen3.6-27B MTP | Workloads d'agent de code privés, non rendus publics | ≈78 % d'acceptation, profil local représentatif, **pas un benchmark reproductible publié** |
| Test Pi Coding Agent (`testquality.com/pi-coding-agent-qwen3-benchmark/`) | Qwen3.6-35B-A3B | Agent de codage local, LM Studio, Apple M5 Max | ≈82 % au premier tour puis ≈69 % quand le contexte s'allonge (réponse tronquée avant le détail exact) |
| `github.com/shreyansh26/qwen-spec-decode-benchmarking` | Qwen3.6-35B-A3B | Benchmark corrigé (une ancienne valeur « 100 % » **rétractée**, artefact de comptage) | ngram-cache : 5,2 % accept, 93,7 tok/s (−19,0 % vs référence) ; spec-draft-n8 : 29,5 %, 30,9 tok/s ; spec-draft-n1 : 69,7 %, 29,2 tok/s. **Concerne un brouillon externe et des variantes n-gram, pas les têtes MTP natives** ; le dépôt affirme que MTP natif gagne sur les fenêtres courtes mais ne publie pas le taux MTP1/2/3 correspondant dans ce tableau |
| `zolotukhin.ai/blog/2026-05-08-...` | Qwen3-A3B | Discussion argumentée pro-MTP | Cité par Luna comme contexte, pas de chiffre isolé retenu ici |

**Aucune mesure trouvée ne compare directement MTP1/2/3 de Qwen3.5/3.6 contre n-gram sur le MÊME corpus de code,
dans le MÊME banc.** Le plus proche (github shreyansh26) compare n-gram/spec-draft externes, pas les têtes MTP
natives.

### 2. Coût de vérification q_len 2-4, b=1/b=8

**Aucune référence vérifiable trouvée donnant simultanément** temps de vérification + mémoire supplémentaire +
q_len∈{2,3,4} + b∈{1,8} + Qwen3.5/3.6 + matériel/quantification/moteur précisés — Luna le dit explicitement plutôt
que d'inventer. Ce qui existe :
* llama.cpp (`forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298`) : tg t/s, TTFT, temps
  total pour Qwen3.6-27B à 5 tokens de draft — **pas le coût isolé de la vérification ni la mémoire par q_len**.
* Benchmark Qwen3.6-35B-A3B (`github.com/thc1006/qwen3.6-speculative-decoding-rtx3090` probable) : TTFT, E2E,
  débit, longueur acceptée — **pas de décomposition draft/verify ni de tableau b×q_len**.
* Documentation vLLM (`docs.vllm.ai/projects/ascend/.../speculative_decoding.html`) : pour K tokens spéculatifs,
  la vérification utilise une taille de capture proportionnelle à b(K+1) — ex. [5,10,15,20] pour K=4 et batchs
  1-4. **Documente la forme du calcul, pas des temps mesurés.**

### 3. Pièges connus, documentés

* **Cache KV des têtes MTP** (`docs.atlascybernetics.ai/deep-dives/mtp.html`) : le cache KV final doit suivre
  `KV_préfixe_accepté + KV_tokens_acceptés`, **jamais** `KV_préfixe_accepté + KV_tous_les_tokens_proposés`. Après
  le premier rejet, le token rejeté et tout ce qui en dépend doit être retiré ; l'état du sampler doit être
  restauré de la même façon.
* **Architectures hybrides (SSM/DeltaNet)** : plus délicat qu'un simple troncage de tableau KV — l'état récurrent
  peut avoir avancé sur une branche rejetée. Un compte-rendu de PR llama.cpp (cité par Luna, non repris ici faute
  d'URL directe dans l'extrait) documente qu'une restauration complète peut obliger à recalculer des tokens
  pourtant valides.
* **Échantillonnage non glouton** : le moteur doit (a) appliquer les MÊMES transformations de logits (température,
  top-p, top-k, pénalités, masque de grammaire) aux distributions draft ET target, (b) calculer la règle
  d'acceptation sur les probabilités effectivement utilisées, (c) rééchantillonner correctement le premier token
  rejeté, (d) ne pas consommer l'état RNG des tokens rejetés comme s'ils avaient été produits, (e) ne pas mettre à
  jour les compteurs de répétition/pénalités avec les branches supprimées. Luna note que le support greedy/nucleus
  dans un dépôt de référence ne prouve pas que chaque moteur MTP Qwen applique correctement cette correction dans
  toutes les combinaisons.

## gpt-oss 120B (recherche web activée sur demande, raisonnement 1s)

Reconnaît d'emblée l'absence d'accept-rate publié pour les têtes MTP Qwen3.5/3.6 sur du code. **Citations à
écarter, non vérifiables** : « Speculative Decoding with Cache-Efficient Branching » (Jiang et al., 2024),
`github.com/microsoft/SpecDec-v2`, « Speculative Decoding under Stochastic Sampling » (Rao et al., 2024),
« Speculative Beam Decoding » (Kim et al., 2024) — **aucun lien cliquable dans la page** (contrairement à Luna),
noms d'auteurs et de dépôts non retrouvés ; classique du paramétrique qui invente une référence plausible plutôt
que d'avouer l'absence de source (REGLES § 1). Le reste de la réponse (KV pollué par les branches rejetées, biais
de rejet sous échantillonnage stochastique, explosion mémoire ≈ b×q_len avec un beam) est correct en principe
mais à traiter comme un raisonnement générique, pas une source.

## Gemma 4 31B (raisonnement, 17s)

« Je n'ai trouvé aucune documentation technique, papier de recherche ou dépôt de code officiel » pour un chiffre
Qwen MTP sur du code — le seul modèle des trois à le dire d'entrée sans détour. Cite HumanEval/MBPP comme
corpus usuels pour ce type de mesure (sans chiffre MTP dessus). Mécanisme de vérification : un seul passage
forward du modèle cible sur q_len positions, coût dominé par la projection de logits — pas de KV cache
supplémentaire pour les têtes elles-mêmes (elles lisent les activations de la dernière couche du modèle cible).
Sur l'échantillonnage non glouton : cite **Leviathan et al. 2023** (« Fast Inference from Transformers via
Speculative Decoding », ICML 2023) — **référence réelle et vérifiable**, règle d'acceptation par rejection
sampling P_target(x) ≥ P_draft(x) sinon accepté avec probabilité P_target(x)/P_draft(x) ; conclut que la
température fait chuter le taux d'acceptation effectif, rendant les têtes MTP moins efficientes en mode non
glouton.

## Bilan pour la 105 d'poste1

**Aucune mesure MTP1/2/3-vs-n-gram-sur-code n'existe, sourcée, pour Qwen3.5/3.6, dans une même séance de banc**
— convergence des 3 modèles, Gemma et gpt-oss l'admettent explicitement, Luna le dit en préambule avant de
lister les meilleures approximations disponibles (aucune n'est la mesure demandée). Toute comparaison chez nous
serait une première, pas une confirmation de chiffre publié. Seule référence solide et directement utilisable :
Leviathan et al. 2023 pour la mécanique d'acceptation sous échantillonnage non glouton (Gemma) — le reste
(tableau Luna § 1, mise en garde KV de `docs.atlascybernetics.ai`) est indicatif, pas une mesure Qwen. Écarter
intégralement les 4 citations de gpt-oss (non vérifiées, probablement fabriquées).
