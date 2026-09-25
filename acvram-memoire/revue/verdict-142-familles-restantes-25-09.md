# Verdict — 142 suite : les 6 familles denses restantes, Marlin au DÉFAUT contre le naturel — 24-25/09 (poste1)

* **instrument** : `scratchpad/poste1-p142-24-09/prise-defaut.sh` (copie figée ; B = défaut servi, rien de posé ; A =
  ACVRAM_PROJ_MARLIN=0 ; T2 = naturel + GEMV_MAX=0) ; kl-chemins (5 invites neuves × 8 pas ; invite brute si le tokenizer n'a
  pas de gabarit de chat, nemo12) ; eval PPL (fenêtres de la 102) ; certifie-b12 CERT_PUR, CERT_CTX 8 192, CIBLE 15 s,
  ABBA A B B A A B B A A B, -lgc 2700
* **commit** : 3a111c98 (scellé et 17 prises), 48d43527 (KL nemo12 rejouée) ; code : main + bascule 156 + 157
* **régime** : cpu-safe 100 → 100 ; compute-apps début = fin (llama-server sur la 3080 Ti) ; A `marlin(off:ACVRAM_PROJ_MARLIN=0)`,
  B `marlin(doubles=0,seuls=N,…)` ; **65 lots et journaux B contrôlés, 0 repli (alerte d'poste6 sur le cache du .so), 0 lot rejoué
  pour repli** ; 7 journaux d'eval sans ligne de régime (l'eval ne l'imprime pas : bras PPL B prouvé par la KL du même bras et
  le même environnement)
* **scellé** : `scelle-familles-restantes.md` + addendum (avant la relance de nemo12 kl)
* **mesuré** (médianes de 5 lots par bras) :

| famille (représentant) | KL B / T2 (argmax) | PPL A = B | b=1 B/A (J) | b=8 B/A (débit, J) |
|---|---|---|---|---|
| granite30 (granite-4.1-30b) | 0,0032 / 0,0050 (40/40) | 6,6864 | 0,982 (−2,1 %) | 0,652 (+53,3 %, −34,9 %) |
| muse30 (muse-glimmer-30b) | 0,0102 / 0,0076 (40/40) | 6,1014 | 0,999 (−3,5 %) | 0,666 (+50,2 %, −34,4 %) |
| 14B (phi-4) | 0,0003 / 0,0006 (40/40) | 5,1679 | 0,986 (−5,7 %) | 0,574 (+74,2 %, −42,6 %) |
| Nemo 12B (Mistral-Nemo-12B-Heretic) | 0,0069 / 0,0222 (39/40 ; T2 37/40) | 4,0539 | 1,001 (−6,6 %) | 0,683 (+46,5 %, −39,9 %) |
| petit dense (deepseek-coder-6.7b) | 0,0013 / 0,0019 (40/40) | 15,6731 | 1,012 (−6,2 %) | 0,772 (+29,5 %, −29,2 %) |
| petit hybride (Qwen3.5-4B-heretic) | 0,0008 / 0,0014 (40/40) | 8,3938 | 1,001 (−7,7 %) | 0,804 (+24,4 %, −33,3 %) |

* **verdict** : **TENU sur les 6 familles**. Aucune régression à b=1 (max 1,012, seuil 1,03) ; b=8 dans les fourchettes
  prédites (0,574 à 0,804) ; KL ≤ 2 × T2 partout (muse30 seule avec B > T2, à 0,67 × le seuil) ; PPL au 10⁻⁴. 157 : nemo12
  `inexacts=1` sur la ligne de régime, comme compté à sec ; les 5 autres 0. Avec les 4 familles déjà mesurées (Qwen3.8,
  gemma31, qwen32, 24B), **10 familles denses qualifiées** pour le défaut Marlin.
* **conduite** : 2 prises tombées en abandon du verrou (rc=3, 1 800 s, sans mesure) et rejouées ; nemo12 kl sans mesure la 1re
  fois (instrument sans repli pour un tokenizer sans gabarit), rejouée après addendum. Modèles heretic : débit, J et KL chiffrés
  seulement, aucune génération lue ni versée.
* **dumps hors git** : kl-{A,T2,B}.pt par famille dans `scratchpad/poste1-p142-24-09/<famille>/`, non suivis.

sha256 (16) des dumps :
* granite30 kl-A.pt 876170a8b5d5dc08
* granite30 kl-T2.pt 9ea4be5175210eb3
* granite30 kl-B.pt 3e1995ba9e0ce0fc
* muse30 kl-A.pt 1d8032b12eecb0be
* muse30 kl-T2.pt ee84b2652c89c74e
* muse30 kl-B.pt b3e68d983e27c955
* phi4 kl-A.pt c14f8712c32ecdd6
* phi4 kl-T2.pt 82f3f8331f1c7144
* phi4 kl-B.pt 7931e1320d9934d4
* nemo12 kl-A.pt bbbfd3f948b9450e
* nemo12 kl-T2.pt a9e10b7f230737ce
* nemo12 kl-B.pt 6bbd9d2aeed91969
* dscoder7 kl-A.pt cb39f9047ef0e3ef
* dscoder7 kl-T2.pt 53bb2f3a5d8aed58
* dscoder7 kl-B.pt 3f0aa6c03b82cf87
* qwen35h4 kl-A.pt d08878454397401a
* qwen35h4 kl-T2.pt bc667b23dd9d9d36
* qwen35h4 kl-B.pt d9011d74838b6eb4
