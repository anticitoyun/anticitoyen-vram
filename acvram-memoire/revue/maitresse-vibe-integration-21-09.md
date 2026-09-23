# Maîtresse — ce que les notes Vibe (plan_acvram_vibe.md, RECHERCHE.md) apportent au projet (21/09, 13 h 23)

Lu : `~/Bureau/Vibe/plan_acvram_vibe.md` (358 l.) et `RECHERCHE.md` (1 114 l.), 15 références arXiv, toutes vérifiées existantes (titres relus sur export.arxiv.org). Tri : ce qui s'applique à des poids existants servis au bit, mesurable sur notre carte, contre ce qui suppose un réentraînement, du multi-GPU ou ce que nous avons déjà.

## Retenu (entre dans la file)

| pièce | source | où | gain prédit / réfutation | qui, quand |
|---|---|---|---|---|
| **Q1 — Four Over Six** : échelle de bloc adaptative (pour chaque bloc de 16, deux candidats d'échelle — max → 6 ou max → 4 — le moins d'erreur gagne), PTQ pur, aucun coût mémoire, compatible AWQ | 2512.02010 | `quant/nvfp4.py` (spécification), noyaux inchangés (le format E2M1+E4M3 ne change pas), `quant/convert.py` | 31B dense texte : KL max 2,12 → ≤ 1,2 nat (réfuté si > 1,5) ; PPL relative +2 SE ≤ 1 % ; test : référence PyTorch == noyau au bit, reconversion du 31B (prise Manon ≈ 15 min) puis scellé qualité E | Océane, après la cellule b=12 |
| **Q2 — ScaleSweep** : initialisation des échelles de bloc par balayage (MSE par bloc) plutôt que max/6 | 2606.07618 | `quant/calibrate.py`, même chemin que Q1 | additif à Q1 : −15 à −20 % de KL médiane ; réfuté si < 5 % | Océane, après Q1 |
| **Q3 — ARCQuant** : canaux résiduels augmentés (+5-10 % de mémoire, noyau à adapter) | 2601.07475 | `quant/nvfp4.py` + `kernels/` | seulement si Q1+Q2 laissent le 31B en « défaut » ; coût noyau réel | chantier, non ordonné |
| **M1 — modèle analytique du pic mémoire** (poids + KV + graphes + activations) confronté aux 9 ctx_tenu mesurés aujourd'hui (verdicts chauffe) | 2503.08311 (méthode), nos verdicts | `memory/tiering.py` (plan), note à sec | si l'écart prédit/mesuré ≤ 5 % sur 9/9 : la dichotomie devient formule + passe de confirmation (chargement −2-4 min) ; sinon on garde la dichotomie et on nomme le terme manquant | Laurine, à sec, après le protocole A/B b=12 |
| **E1 — banc énergie NVML commun aux quatre moteurs** (J/jeton, W, à horloge égale, ≥ 20 s) | 2509.08867, 2504.17674 | `outils/gpu/mesure/banc-4moteurs.py` (+ `CUDA_VISIBLE_DEVICES=0`) | = pièce G de la file, inchangée | Laure (instrument), Manon (mesure), après TRT-LLM |
| **C9 — déchargement sélectif des neurones/experts actifs** (PowerInfer) | 2504.14893 | `memory/repin.py`, cache d'experts (chantier C9-M3 en branche) | seulement pour les modèles qui ne logent pas (119B) ; question utilisateur ouverte | après réponse utilisateur |

## Écarté, et pourquoi (une ligne chacun)

* MoEpic (2509.08342) : découpage d'experts CPU/RAM pour modèles qui ne tiennent pas — nos modèles servis logent en VRAM ; sa « glue » n'est pas la nôtre (H2 fermé : la frontière de pas, pas le MoE).
* Aurora (2410.17043) : ordonnancement communication/calcul multi-GPU — une carte.
* Expert pruning / dynamic routing : changent la sortie → interdits (une optimisation qui change la sortie est un bogue).
* GLA (2505.21487), MTLA (2505.13544), TransMLA (2502.07864), MHA2MLA : architectures d'attention à entraîner ou à convertir avec perte — nous servons des poids existants au bit ; MLA GLM reste notre `attn_mla_causal`.
* MemOS : système de mémoire pour agents, hors sujet ; « hiérarchie mémoire » chez nous = tiering.py déjà en place.
* DVFS : c'est le régime éco 2700 (J −9 %/−27 % mesurés) ; continuous batching : c'est `_admit/_build_batch` depuis le début ; décodage spéculatif : existe (n-grammes, MTP), à mesurer en J, pas à « ajouter ».
* Roadmap 12 semaines et cibles chiffrées du plan Vibe (« KL < 1 nat », « 2× », « −30 % ») : non mesurées chez nous — remplacées par les seuils ci-dessus, réfutables sur la carte.

## Ordre

1. Océane : Q1 après la cellule b=12 (une fonction : `quant/nvfp4.py` échelle adaptative + option de conversion `--echelle=4sur6`, test au bit référence/noyau, reconversion 31B par Manon, scellé E rejoué).
2. Laurine : M1 à sec après le protocole A/B (note `laurine-pic-memoire-21-09.md`, table alias → prédit → mesuré → écart).
3. Laure : E1 = pièce G, inchangée ; PLAN_ACVRAM.md du dossier Vibe mis à jour par la Maîtresse avec ce tri.
