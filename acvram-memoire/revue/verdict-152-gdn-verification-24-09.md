# Verdict — 152 (1) : projections GDN à la vérification spéculative des hybrides — 24/09 (poste1)

* **instrument** : `scratchpad/poste1-p152-24-09/banc-152.py` (un lot = un processus ; Qwen3.8-27B-nvfp4, b = 1, défaut
  servi : graphes, ngram k = 4 ; invite de code, recopie de `acvram/engine/lot_etats.py`, 655 jetons ; 384 jetons
  gloutons, ignore_eos) ; `prise-152.sh` (copie figée, ABBA A B B A A B B A A B, -lgc 2700)
* **commits** : f63e3fb4 (GEMV à NV lignes, prise arrêtée) ; 99be0f97 (voie groupée par poids, prise complète)
* **régime** : RTX 5090, -lgc 2700 ; cpu-safe 100 → 100 ; compute-apps début = fin (4627, la 3080 Ti) ;
  `graphes=on(hybrides≤4)`, NOMINAL, 0/64 exilée ; prise poste1-p152 12:16:44 → 12:24:51
* **scellé** : `scratchpad/poste1-p152-24-09/scelle.md` + addendum 152 bis (H0/H1, écrit avant)
* **mesuré** :
  * GEMV NVFP4 à NV lignes contre NV = 1 (f63e3fb4) : **PAS au bit** (qkv 10240×5120, gate 6144×5120, out 5120×6144,
    dès NV = 2 ; seul 48×5120 au bit). Ma prédiction à sec (au bit) était fausse. Prise arrêtée avant l'ABBA.
  * ABBA, voie groupée par poids (mêmes appels M = 1) : **sha des jetons identique sur les 10 lots** (1a3996e2cd75476e),
    jetons/pas identiques (4,5176, 85 pas pour 384 jetons). t/s de décodage, médianes de 5 lots : **A 124,28**
    (124,01-124,77), **B 123,65** (123,53-123,70), **B/A = 0,995**.
  * Test d'équivalence au bit : ROUGE par `RuntimeError` du montage-jouet (non diagnostiqué, sans objet après retrait).
* **verdict** : **H0 TENUE, H1 réfutée**. Les relectures des projections GDN du déroulé jeton par jeton étaient DÉJÀ
  servies par le L2 (≈ 65 Mo par couche entre deux jetons, sous 96 Mo). Regrouper par poids ne gagne rien (−0,5 %,
  constant, frais de concaténation). **Code retiré**, retour à f1dcd98b pour gdn.py, couches.py et cli.py. Le verdict 151 (1) est corrigé en conséquence :
  ses « +9,4 Go » étaient des octets algorithmiques, pas DRAM. Même raisonnement pour la 152 (2), eager b > 8 : boucle
  des séquences dans la couche, même ensemble de travail de 65 Mo, donc probablement L2 aussi. Je la propose close sans
  mesure, ou à trancher par un ncu des octets DRAM plutôt que par un ABBA.
* **fait nouveau** : sur une invite de code, ngram k = 4 atteint 4,52 jetons par pas sur Qwen3.8 (graphes hybrides ≤ 4).
