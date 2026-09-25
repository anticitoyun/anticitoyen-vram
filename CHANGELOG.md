# Journal des changements

* **25/09/2026 — pièce 187** : le GEMV int8 traite ses activations par tranches de **6** au lieu de 16
  (`ACVRAM_INT8_TRANCHE` pour N ≤ 16, `ACVRAM_INT8_TRANCHE_PREFILL` au-delà ; 16 = témoin d'avant). Sortie **identique au
  bit** : test sur les poids réels du mixte (N 2-80, bf16 et fp32) et bras cassant. À NV ≤ 6, le noyau tient en 128
  registres, soit 2 blocs par SM au lieu d'un, ce qui l'emporte sur les relectures des poids. GEMV −22 à −44 % de 16 à 78
  jetons ; servi sur Qwen3.8-27B mixte-i8c : **+6,65 %** au banc chat b=8, **+10,93 %** à b=16, J/jeton −3 à −5 % ;
  Qwen3.8-27B-nvfp4 (pas d'int8 servi) inchangé. Détail : revue/poste5-piece187-verdict-25-09.md.

* **25/09/2026 — pièce 172** : au préfill de plusieurs séquences, la boucle par séquence des couches à récurrence
  linéaire (Gated DeltaNet de Qwen3.5/3.8) déquantifie chaque poids NVFP4 UNE fois au lieu d'une fois par séquence
  (`ACVRAM_DEPAQ_PARTAGE`, défaut 1 ; 0 = témoin). Sortie **identique au bit** : tests, bras cassants, logits de
  Qwen3.8 et Qwen3.5-35B-A3B. Forward de préfill −12,5 % / −11,3 % sur Qwen3.8 et −3,8 % / −3,6 % sur Qwen3.5-35B ; TTFT
  servi sous 8 requêtes −9,2 % / −8,1 % et −3,1 % / −3,4 % (8 × 78 jetons / longueurs mêlées). La cause de l'écart de
  sortie de `GDN_PREFILL_LOT=1` est trouvée (cuBLAS bf16 réduit en bf16 selon M, pièce 169). Détail :
  revue/poste5-piece172-verdict-25-09.md.

* **24/09/2026 — pièce 166** : l'opt-in `ACVRAM_PREFILL=marlin` (pièce 147 L2, GEMM Marlin W4A16 au préfill de la disposition
  unique, sans dépaquetage) est RETIRÉ, verdict FAUX : TTFT servi b=1 +4 / +27 / +35 % à 512 / 2 048 / 4 096 jetons, J/préfill
  +5 / +28 / +36 %, KL 2,3-3 × les témoins (PPL par fenêtre tenue) — Marlin perd à grand M contre dépaquetage + cuBLAS ;
  `ACVRAM_PREFILL` revient à `bf16 | w4a16 | w8a8 | w4a4`, `ACVRAM_PREFILL_MARLIN_MAX_M` disparaît. Détail :
  revue/poste6-piece147L2-verdict-24-09.md ; mécanisme : acvram-memoire/MECANISMES.md.
* **25/09/2026 — pièce 165** : `ACVRAM_GDN_PREFILL_LOT=1` est l'option TTFT des modèles Gated DeltaNet (Qwen3.5/3.8) :
  au préfill de plusieurs séquences, les projections GDN du lot passent en un appel au lieu d'un par séquence. TTFT servi
  sous 8 requêtes simultanées (Qwen3.8-27B-nvfp4) : **−18,6 %** (8 × 78 jetons, 431,5 → 351,1 ms) et **−15,7 %**
  (longueurs mêlées, 466,6 → 393,6 ms). Elle reste en OPT-IN, car elle change la sortie au-delà du critère scellé :
  accord d'argmax 98,38 % contre 99,19 % pour les séquences servies seules (Qwen3.8, 8 × 78), KL_max jusqu'à 2,2 × le
  témoin (0,319 contre 0,146 ; 1,240 contre 0,568 sur Qwen3.5-35B-A3B). La PPL par séquence reste au niveau des témoins.
  Détail : revue/poste5-piece165-verdict-24-09.md.

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

* **24/09/2026 — pièce 162** : bilan chiffré matin/soir (`c10cfee5` contre `HEAD` `5375945b`), ABAB × 5, -lgc 2700,
  Qwen3.8-27B-nvfp4 et gemma-4-31B-it-nvfp4-vision, b = 1 et b = 8, plus TTFT à une invite de 2 048 jetons. Débit à
  b = 8 : Qwen3.8 **+66,66 %** (274,4 → 457,3 t/s), gemma **+56,95 %** (252,4 → 396,8 t/s, dans la bande prédite
  57-60 %). Énergie à b = 8 (J/jeton net, BAISSE = gain) : Qwen3.8 1,164 → 0,691 J/jeton (**+40,65 %** d'économie),
  gemma 1,265 → 0,803 J/jeton (**+36,53 %**). TTFT inchangé aux deux modèles (± 1 %, Marlin/GDN sont des leviers de
  décodage, pas de prefill). Isolation des deux leviers (Qwen3.8, HEAD seul, b = 8, `ACVRAM_GDN_ETAT_EN_PLACE`,
  `ACVRAM_GDN_CONV_FUSEE`, `ACVRAM_GDN_RES_DIFFERE`, `ACVRAM_GDN_PORTES_NOYAU` (F1), `ACVRAM_GDN_NORME_FUSEE` (F3),
  `ACVRAM_NORME_REGISTRES` (F6) tous à 0) : GDN seul **+8,76 %** de débit (bande prédite 3-15 % tenue), Marlin seul
  (déduit) **+53,5 %** ; composition vérifiée 1,535 (marlin) × 1,0876 (gdn) = 1,670 contre 1,667 mesuré directement
  (écart 0,2 %). Détail : scratchpad/poste2-piece162-bilan-24-09/verdict-final.md.
