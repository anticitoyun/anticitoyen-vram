# Journal des changements

* **24/09/2026 — pièce 156** : les linéaires NVFP4 des modèles DENSES sont servis par défaut en disposition Marlin unique
  (GEMV v2, TPB par forme) : +57 à +90 % de débit à b = 8, b = 1 inchangé (0,979 à 0,996), sortie qualifiée (KL sous 2 ×
  témoin, PPL identique), TTFT +2 à +4 ms (B/A 1,001 à 1,012 sur gemma4 31B et Qwen3.8-27B, invites de 512, 2 048 et
  4 096 jetons : revue/poste6-piece147-verdict-24-09.md, dépaquetage CUDA de la pièce 147) ; MoE inchangés ; repli
  `ACVRAM_PROJ_MARLIN=0`. Cinq variables pilotent la disposition (`ACVRAM_PROJ_MARLIN` défaut 1, `ACVRAM_PROJ_MARLIN_PORTEE`
  défaut `denses` — un modèle à `MoEBlock` garde son chemin naturel, ses 32 alias touchés par les linéaires hors experts
  n'étant pas mesurés : revue/poste1-piece142-inventaire-denses-24-09.md —, `ACVRAM_GEMV_MARLIN_V2` défaut 1,
  `ACVRAM_GEMV_MARLIN_TPB` et `ACVRAM_GEMV_MARLIN_S` défaut 0 = règle automatique) ; replis nommés au chargement si la
  capacité KV ou la mémoire manquent (`ACVRAM_PROJ_MARLIN_CAPACITE`, refus explicite, jamais un exil silencieux).

* **24/09/2026 — pièce 157** : les échelles de bloc NVFP4 sous-normales en S0E5M3 (le format d'échelle du port Marlin, plage
  ≈ 2^14,8 contre 2^17,8 pour l'E4M3 d'origine) faisaient perdre des blocs de 16 poids entiers, mis À ZÉRO plutôt que
  représentés, dans la disposition Marlin — touché au défaut servi (MoE : max|Δ logits| = 4,52 sur Qwen3-14B, trouvé par
  la suite complète de la bascule 156). Corrigé : facteur d'échelle PAR LIGNE pour les poids denses (au lieu d'un facteur
  par tenseur/pile), poids encore inexact exclu de la disposition et rendu au chemin naturel (raison nommée) ; les piles
  MoE dont un poids écrase restent en disposition naturelle. Preuve : Qwen3-14B au bit contre le naturel après correctif
  (avant : 4,52 ; après : 0,0) ; coût ≤ 2 % (b = 1 1,006, b = 8 1,017, Qwen3-Coder-30B).
  Détail : revue/verdict-157-marlin-sous-normales-24-09.md.

* **24/09/2026 — pièce 146** : trois défauts corrigés dans le chemin par défaut du cache KV. (1) Une séquence tronquée
  par épuisement du KV n'était JAMAIS livrée à la requête HTTP (chemin pipeline et spéculatif) — la sortie finie était
  jetée sans clore la requête ; corrigé, les séquences épuisées sont livrées dans le pas même où `step` les détecte.
  (2) Le budget KV tombait à 6 % de la VRAM dès qu'une seule séquence y tenait (marge de 7 % posée en dur, avant la 146
  au chargement) ; corrigé par un départage min(KV, demande) après exil. (3) La tête liée (`lm_head`) laissée en bf16
  puis convertie en fp32 à la volée (5,25 Gio) provoquait un OOM à la chauffe sans `empty_cache()` préalable ; corrigé.
  Preuve bout en bout (uvicorn réel, graphes + pipeline) : rouge sur le commit d'avant, vert après, pour chacun.
  Détail : revue/verdict-146-kv-defaut-24-09.md.

* **24/09/2026 — pièce 156 (fusions GDN)** : six fusions du décodage des couches à récurrence linéaire (GDN), toutes
  numériquement identiques (au bit ou à l'ulp) au chemin qu'elles remplacent, mesurées sur Qwen3.8-27B, b = 8, disposition
  Marlin qualifiée, **toutes par défaut** (poste5, 255042e8) : **F2** conv de décodage fusionnée, **F4** état GDN mis à
  jour en place, **F5** résidu différé des couches GDN (ensemble : −7,8 % de temps de pas,
  revue/poste5-piece156c-verdict-24-09.md, au bit) ; **F6** RMSNorm en registres (+8,1 % à b = 8 seule avec F1/F3, au bit
  — l'ordre de sommation d'un fil n'est pas observable en sortie bf16 sur ≤ 8 carrés) ; **F1** portes dans le noyau fla et
  **F3** norme gated Triton (± 1 ulp bf16, KL/PPL tenues contre le témoin, initialement laissées en opt-in le temps de la
  revue — désormais par défaut). Détail : revue/poste5-piece156c-verdict-24-09.md, revue/poste5-piece156d-verdict-24-09.md.

* **24/09/2026 — pièce 147** : le dépaquetage Marlin → bf16 au préfill (disposition unique) coûtait +26 à +34 ms par
  requête (v1, un fil par tuile, copies `.contiguous()` des vues q/k/v) ; réécrit (v2, un fil par colonne, lignes
  entières en deux uint4, vues à pas libre sans copie) : **+2 à +4 ms** à toute longueur d'invite (512 à 4 096 jetons),
  au bit contre les chemins Triton et torch de référence. `ACVRAM_DEPAQUETAGE=auto` choisit CUDA si l'extension l'a,
  sinon Triton. Détail : revue/poste6-piece147-verdict-24-09.md.

* **24/09/2026 — pièce 161** : le `.so` compilé du port Marlin était PARTAGÉ entre worktrees sous un nom de cache fixe —
  deux arbres aux sources différentes alternant sur la même carte se recompilaient l'un l'autre (25 s de nvcc à chaque
  changement d'arbre, y compris hors verrou carte.sh) sans jamais désigner l'autre arbre comme cause. Corrigé : cache
  keyé par empreinte sha256 des sources (comme `kernels/__init__.py`, pièce antérieure sur l'extension principale),
  sources copiées dans le cache, le moteur en service ne relance jamais ninja (charge le `.so` de son empreinte ou
  replie au naturel, raison imprimée). Détail : revue/poste6-piece161-verdict-24-09.md.
