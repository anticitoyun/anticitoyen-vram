# Verdict — 157 : échelles sous-normales dans la conversion Marlin NVFP4 — 24/09 (poste1)

* **défaut** : `traiter_echelles_nvfp4` (marlin_port/__init__.py:151, transcrit de vLLM) met À ZÉRO toute échelle de bloc
  < 2 après ×facteur×2⁷, avec un facteur calé sur le MAX du poids (`facteur_nvfp4`) ou de la PILE d'experts entière
  (:185). S0E5M3 n'a qu'environ 2^14,8 de plage, contre 2^17,8 pour e4m3 : un poids qui mêle 448 et des sous-normales perd des
  blocs de 16 poids entiers. Touché au DÉFAUT SERVI : le Marlin des experts MoE (ACVRAM_GEMV_LAYOUT=marlin, moe.py:506).
  Trouvé par la suite complète de la bascule 156 : Qwen3-14B, préfill Marlin contre naturel, max|Δ logits| = 4,52.
* **comptage** (safetensors, processeur) :
  * experts : Qwen3-Coder-30B 137 656 / 1,81 × 10⁹ échelles, concentrées dans la couche 0 (gate 0,571 %, up 0,558 %,
    43-44 experts sur 128), puis les couches 1, 2 et 4 (≈ 0,002 %) ; GLM-4.7-Flash 0 (une couche sur 6).
    Un facteur par expert ne sauve RIEN (les experts fautifs couvrent eux-mêmes la plage).
  * denses, facteur par poids → par ligne : Qwen3-14B 6 poids / 2 978 145 → 2 / 613 ; qwen32 9 / 8 420 → 7 / 588 ; Qwen3.8
    3 / 452 → 0 ; gemma31 0 ; 24B 0.
* **correctif** (branche poste1-157, sans la bascule) : dense, facteur PAR LIGNE (e4m3 × 2^k ≤ 448 exact, repris dans le g PAR
  COLONNE déjà servi) seulement si le scalaire écrase, et les autres poids gardent au bit la préparation d'avant ; poids
  encore inexact exclu de la disposition (`inexacts`, naturel, nommé sur la ligne de régime ; le Marlin paresseux le respecte
  aussi). MoE : pile refusée si écrasement, avec la raison nommée par le mécanisme existant (« up_proj : 67 477 échelles
  sous-normales non représentables en Marlin (S0E5M3) »), et la pile naturelle gardée.
* **preuve de justesse** :
  * au niveau des POIDS : dépaquetage au bit de `nvfp4_dequant` après correctif, sur des sous-normales FORCÉES (tests
    test_marlin_echelles_157, 4 verts ; cassants : facteur par ligne retiré → 3 rouges, exclusion retirée → 1 rouge) et sur
    les poids RÉELS (Qwen3-14B couches 1, 2 down et 2 gate faux avant ; pile de la couche 0 de Coder-30B refusée et nommée) ;
  * au niveau du MODÈLE : **Qwen3-14B, préfill de 64 jetons, Marlin contre naturel : AU BIT** (max 0,0 ; avant : 4,52).
  * La KL de Coder-30B contre le naturel (avant 0,1655, après 0,1659) était un **critère mal posé** : elle mesure le chemin MoE
    par défaut (mma-a4, marlin_tensor, 0,166 dans les deux cas), pas la conversion des échelles. Ce n'est pas un résultat.
* **coût** (ABBA Coder-30B, certifie-b12 CERT_PUR, CERT_CTX 8 192, -lgc 2700, prise poste1-p157c 16:46 → 17:10,
  A = aaf2add8 avant, B = 157, médianes de 5 lots) : **b=1 3,422 → 3,442 ms (B/A 1,006)** ; **b=8 5,500 → 5,595 ms (1,017)** ;
  seuils scellés ≤ 1,03 et ≤ 1,04 : TENU. Le prix des 4 piles naturelles sur 48 est ≤ 2 %.
* **conduite** : trois relances pour fautes d'instrument (montage de test ×1e-6 → échelle 0 et non sous-normale ; bras A de la KL
  mal posé ; certifie-b12 sans ACVRAM_ARBRE et fenêtre trop longue), chacune consignée en addendum au scellé avant relance.
