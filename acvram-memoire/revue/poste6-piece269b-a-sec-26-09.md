# 269 b — porte de la fenêtre d'admission ouverte à UNE requête si une autre est déjà entrée (opt-in, à sec, scellé AVANT mesure — poste6, 26/09)

instrument (prévu) : `scratchpad/poste6-p269b-26-09/prise-269b.sh` (instruments 262, A B B A B A, 7 tours à 12 + 5 solo par bras), serveur neuf par bras
commit : poste6-269b (code + test + scellé) ; mesure APRÈS la remise de 15 h (régime mesuré, ordre de chef), sur main fusionnée avant la prise
régime : celui de la 262/269 (Coder-30B nvfp4, -lgc 2700, ECO=off, spéculation coupée, max-batch 16), fenêtre 5 ms des deux côtés
scellé : ci-dessous, avant toute prise
mesuré : rien encore
verdict : à venir
durée : prévu 6 bras × 45 s ≈ 5 min + chargements → 1 prise ≤ 15 min

## 1. Le code (opt-in `ACVRAM_ADMISSION_GUET=1`, défaut 0 : rien ne change)
* `acvram/server/app.py` : `EngineService.en_entree` = requêtes HTTP entrées dans un gestionnaire de génération (chat, messages, completions)
  et pas encore soumises (`with service.entree():` du début du gestionnaire jusqu'au `submit` inclus ; libéré aussi sur exception avant submit).
* `_attendre_les_arrivees` : à 1 requête en file, la porte s'ouvre si `en_entree > 0` (quelqu'un est derrière) ; dans la boucle, `en_entree > 0`
  compte comme une file qui grossit (le calme de FENETRE ne commence qu'une fois tout le monde soumis) ; plafond 4 × FENETRE inchangé.
* Une requête seule : `en_entree` = 0 au réveil du fil (elle-même est déjà soumise) → porte fermée, 0 ms (179 b). Test à sec :
  `tests/test_admission_guet_269b.py` (solo 0 ms avec guet ; 1 + 1 entrée → attente jusqu'au plafond ; défaut coupé ; rafale à 2 comme avant ;
  compteur libéré sur exception ; variable déclarée). Régime : `ADMISSION_GUET` dans regime.VARIABLES (lu_a app._GUET_ADMISSION), cli, défauts servis.

## 2. Ce que la 269 a mesuré et que la 269 b vise
* Après la 268, A5 : 8/14 tours à un pas ; tout tour à deux pas restant = [1, 11] (réveil du fil avec 1 en file, la 2e à ≤ 5 ms derrière dans le
  gabarit/tokeniseur hors boucle). Coût d'un tel tour : mur +20-35 ms, et 11 requêtes sur 12 prennent le TTFT du 2e pas (263-282 contre 242-262).
* Avec le guet, au réveil à 1 en file, `en_entree` ≥ 1 (les 11 autres sont entrées, en cours de gabarit) → fenêtre ouverte → un pas de 12.

## 3. Scellé (chef, avant mesure) — A = guet 0, B = guet 1, fenêtre 5 ms, 14 tours à 12 et 10 solo par bras
* P1 : B ≥ 12/14 tours à un pas (A : 8/14 dans la 269). FAUX si B ≤ 10/14.
* P2 : p50 par requête B ≤ 250 ms (A : 262,5) ; p95 et max par tour B ≤ ceux de A (lecture pire cas de la 269).
* P3 : solo B = A ± 1 ms sur la médiane de 10 tours, ET fenêtre 0,00 ms dans les 10 tours solo de B (trace) — c'est la trace qui juge le
  mécanisme, la médiane peut bouger de ± 2 ms d'une instance de serveur à l'autre (269 : 37-41 ms sur A).
* Issue défavorable nommée : le guet ouvre la fenêtre mais la 2e requête met > 4 × 5 = 20 ms à être soumise (gabarit long, images) → le
  fil attend 20 ms pour rien puis fait un pas de 1 quand même : mur +20 ms sur ces tours, pire qu'avant. Visible à la trace (fenêtre ≈ 20 ms, pas
  de 1). Si ≥ 2 tours/14 le montrent, B est FAUX même avec P1 tenu.
* Issue qui me gênerait : P1 tenu (≥ 12/14) mais P2 faux (p50 > 250) — alors le pas de 12 lui-même (225-245 ms) est le plancher et le guet ne
  peut pas faire mieux : gain réel mais scellé mal placé, à dire tel quel, sans redescendre le seuil.
* Décision annoncée : P1-P3 tenus → proposer le défaut 1 à chef (test cassant « défaut 1 », CHANGELOG) ; sinon opt-in documenté ou retrait.
