# 245 — cœur GDN du préfill en longueurs variables (fla cu_seqlens), au-dessus de la 243 : AU BIT, mais SANS GAIN — FAUX

* instrument : `scratchpad/poste1-p245-26-09/` — `profil-245.py` (copie de `profil.py` 204 b, compteur COEUR_LOT_APPELS),
  `prise-p8.sh` (A B B A, un processus par bras) ; `prise-banc.sh` (banc chat 102 8 × 256, serveur neuf, A B B A A B, preuve
  GDN_COEUR_LOT=1 dans regime_ligne) ; tests `tests/test_gdn_coeur_lot_245.py`
* commit : poste1-245 8f1f92c3f (base origin/fusion-070 fcd73f875 = 0.7.0, 243 à 16 par défaut ; code 67564012)
* régime : Qwen3.8-27B-unsloth-mixte-i8c, carte 0, -lgc 2700 (banc : horloge moyenne 2 561-2 604, `bridages: puissance` dans les
  6 passes, les deux bras), ACVRAM_ECO=off, cpu-safe 100
* scellé : `scratchpad/poste1-p245-26-09/scelle.md` (avant le code) + addendum 2 (8f1f92c3f, avant les prises 5-6)
* mesuré : tests 35 passed / 2 skipped (au bit fp32 et bf16, cassants rouges) ; p8/p1 2 paires ; banc 3 passes par bras
* verdict : **FAUX** — Δp8 moyen −0,65 ms > −3 ms (seuil FAUX) ; banc +0,01 % (z 0,02) ; p1 inchangé
* durée : prévu 8 + 12 min ; tenu 214 s (p8) + 346 s (banc) (`carte.sh` journal `tenue=`)

## Chiffres
| | A (défaut) | B (COEUR_LOT=1) | Δ |
|---|---|---|---|
| p8 médiane (paire 1 / paire 2) | 427,1 / 426,8 ms | 427,5 / 425,1 ms | +0,4 / −1,7 ms (moy. −0,65) |
| p1 médiane | 188,9 / 189,0 ms | 188,9 / 188,9 ms | 0 |
| COEUR_LOT_APPELS | 0 | 336 (= 48 couches GDN × 7 préfills multi-séquences) | preuve de prise |
| banc chat b=8 (t/s, 3 passes) | 462,9 ± 2,4 | 463,0 ± 2,4 | +0,01 % |
| J/jeton brut | 0,854 moy. | 0,853 moy. | nul |

## Lecture
* La 0.7.0 a divisé p8 par deux (886 → 427 ms, 243 comprise) : le plafond de la 245 (fla 19,5 ms GPU, 2 688 lancements sur
  855 ms de noyaux, 204 b) était mesuré sur l'ANCIENNE base. Le prédit (−10 à −25 ms) supposait que le temps fla par séquence
  était sur le chemin critique ; il ne l'est pas, ou plus : grouper 8 appels en un ne retire rien au mur. Cause non mesurée
  (lancements recouverts par le GPU, ou travail fla borné par les données et non par le lancement) — une trace nsys de p8
  sur 0.7.0 la trancherait, je ne la propose pas : le levier est au plus −1,7 ms sur 427 (0,4 %).
* Issue qui me gênait (B plus lent) : non arrivée.

## Suite proposée (décision chef)
* Ne pas fusionner la 245 (code opt-in sans gain = surface de test sans bénéfice) ; branche poste1-245 gardée comme trace.
* Si un poste p8 sur 0.7.0 est rouvert : repartir d'une décomposition par famille sur la NOUVELLE base (profil 204 b rejoué), pas
  des plafonds du 25/09.
