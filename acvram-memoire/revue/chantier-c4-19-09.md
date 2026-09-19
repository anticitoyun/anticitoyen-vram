# Chantier C4 — godets sur `b` : prérequis `_bind_hybrid`, preuve à sec, `ACVRAM_GODETS_B` (19/09, branche `poste1-c4-godets-b`)

Source : `poste7-poursuite-chantiers-19-09` § 2 ligne C4 ; `MECANISMES.md` « Prérequis bloquant du chantier godets sur `b` ». Tout à sec, CPU, aucune carte.

## Objectif
Arrondir la dimension `b` de la clé de graphe CUDA à un godet sans qu'un créneau de rembourrage corrompe l'état récurrent d'une séquence vivante (hybrides GDN/Mamba2/KDA), et le prouver par un test qui casse sur l'ancienne liaison.

## Prédiction scellée et seuil (copiés de la commande, AVANT toute mesure)
« ms/pas b=4 ≤ 0,95 × le pas actuel (godet exact) ; PPL décodage hybride = prefill ± 0,002 (Qwen3-Next ou Nemotron, chemin GDN) ».
**Prémisse à corriger avant la fenêtre carte** : « le pas actuel » n'est PAS le godet exact — le godet sur `b` est en place depuis le 11/09 (70c10a3 `bucket_batch`, 69fc60d test, 1213554 `range(godet)`), et à b=4 le godet EST exact dans les deux régimes (4 = 2²) : ce rapport ne peut rendre que 1,00 ± bruit, il ne mesure rien. Le bras qui donne un sens au 0,95 est un lot qui se vide (12 → 1 : douze clés exactes contre cinq godets sous `MAX_GRAPHS=16`, graphs.py:123), godets (défaut) contre lot exact (`ACVRAM_GODETS_B=0`, témoin ajouté ce soir). Proposition à valider par poste7 : temps total du drain 12 → 1 sous godets ≤ 0,95 × sans godets ; ms/pas à b=4 exact = 1,00 ± 0,02 (contrôle de l'instrument) ; PPL décodage hybride inchangée à ± 0,002 dans les deux régimes.

## Ce qui existait déjà (fichier:ligne, HEAD 47812d9)
* `preparer` : `b = bucket_batch(b_reel)` (graphs.py:356 avant, :382 après), `godet_hybride` (:364, plafond `ACVRAM_HYBRID_SLOTS`, poste3 13/09 bead x0s), `key = (b, ql, nblk, lb)` (:408).
* `_fill` rembourre x à zéro, slot −1, seq_len 0, table bloc 0 (graphs.py:642-671).
* `_bind_hybrid` lie `range(godet)` avec une sentinelle DISTINCTE par créneau vide (−1, −2, … ; graphs.py:568-607) — le prérequis de MECANISMES est fait depuis 1213554 ; MECANISMES.md le disait encore « à faire », corrigé ce soir.
* Tests : `test_godet_hybride.py` (simulacre `+1`, 12 tests, 13/09) ; `test_graphs.py::test_graph_bucket_padding_equivalence` (dense, carte).

## Ce qui manquait, trouvé par une sonde (pas par lecture)
`static_bind` (model.py:2348) traitait la sentinelle comme un sid ordinaire : à l'éviction du créneau de rembourrage, son état (avancé par les rejeux) était **exporté sous `store[-1]`** — une clé absente de tout lot — puis **rechargé** au cycle suivant : le créneau fantôme n'était neutre qu'au premier cycle. Sonde `scratchpad/sonde-sentinelle-c4-19-09.py` (à rejouer sur 47812d9 pour revoir la faute) sur le simulacre : cycle 1 |S| = 0, magasin `[-1, 10, 20, 30, 40]`, cycle 2 |S| = 1. Sur un GDN à entrée nulle l'état reste à zéro (projections et conv sans biais) — invisible en valeur ; sur Mamba2 (mamba2.py:68 `conv_b`, :160 `dt_bias`) il bouge. Corrigé : `_sid_fantome` (model.py:2243), sentinelle jamais exportée, jamais dans le magasin, zéro à chaque liaison (model.py:2360-2385).

## Preuve à sec faite ce soir — `tests/test_godets_b.py` (9 tests) + `test_godet_hybride.py` (12) : 21 passed en 0,19 s
Rejouer : `CUDA_VISIBLE_DEVICES= ACVRAM_TESTS_PENDANT_MESURE=1 PYTHONPATH=<arbre> <venv>/bin/python -m pytest tests/test_godets_b.py tests/test_godet_hybride.py tests/test_regime_noyaux.py tests/test_cadrage_perplexite.py -q -p no:cacheprovider`
Montage : `GatedDeltaNet` RÉEL (projections, conv à état, `decode_static` gdn.py:221-225 qui mute en place), `DecoderLayerGDN.static_bind` RÉEL, `GraphRunner._bind_hybrid` RÉEL, `decode_fixed` → `_la_decode` (model.py:2398-2420, un `decode_static` par créneau, comme sur carte sans fla). `transformers` absent du venv : règle récurrente transcrite dans le test (`_regle_recurrente_locale`), la référence réelle sert si elle est installée — les invariants portent sur liaison et export, pas sur la règle.
* `test_pas_a_godet_superieur_au_lot_reel` : lot 3, godet 4 → créneau 3 lié à −1, |S| = |conv| = 0 ; sorties et états (conv, S) des 3 vivantes **égaux au bit** au même pas à lot exact ; magasin = {10, 20, 30}.
* `test_scenario_du_10_09_…` : [10,20,30,40] → [10,20,30] → [10,20,30,40], trois pas, tout égal au bit au lot exact ; aucune clé négative.
* **Témoin cassant 1** `test_temoin_ancienne_liaison_sur_sids_corrompt_40` : `enumerate(sids)` (graphs.py avant 1213554, recopié) → pas 0-1 égaux, pas 2 ≠, S(40) ≠, 10 intact — l'ancienne liaison fait échouer le test précédent.
* `test_sentinelle_neutre_a_chaque_cycle_et_absente_du_magasin[gdn|etat-mobile]` : cinq lots alternés, créneau 3 à zéro aux deux cycles, magasin {10,20,30,40,50}, 40 exporté intact sous sa clé.
* **Témoin cassant 2** `test_temoin_ancien_static_bind_exportait_la_sentinelle` : `static_bind` d'avant ce soir (copié) → `-1 in store` sur les deux montages, |S| > 0 au second cycle sur l'état mobile.
* Régime : `ACVRAM_GODETS_B` (regime.py:148, `lu_a` graphs `_GODETS_B`, témoin `0` ; cli.py:138). **Défaut 1 = comportement actuel inchangé**, prouvé : `godet_lot(b) == bucket_batch(b)`, `godet_hybride(b,12) == min(bucket_batch(b),12)` ; `0` : lot exact, refus au-delà des créneaux inchangé ; `regime.masquer(["GODETS_B"])` bascule le module importé ; `preparer` passe par `godet_lot`/`godet_hybride` (contrôlé sur la source).
* En passant : `test_cadrage_perplexite::test_la_liste_des_variables_lues_ne_derive_pas` était rouge sur main (5 variables d'autres chantiers absentes de la garde cli.py : DOUBLE_DISPOSITION_DIAG, DUMP_MOE, GEMV_LAYOUT, PREFILL_A8, PREFILL_A8_FMT) — ajoutées. 58 passed, 18 skipped (carte) sur les 9 fichiers touchant graphes/régime.

## État
* FAIT : sentinelle neutre à chaque cycle et hors magasin (`static_bind`) ; `ACVRAM_GODETS_B` 1|0 déclarée, testée ; preuve à sec avec deux témoins cassants ; MECANISMES.md mis à jour (ligne 52 et § prérequis).
* RESTE : la mesure carte (ci-dessous) ; `ppl-decode-kv-17-09.py` est mono-séquence — à b=1 le godet vaut le lot, le rembourrage n'est PAS exercé : il faut une variante à 3 séquences (godet 4, une notée + deux de remplissage) avant de conclure sur la PPL hybride sous rembourrage.
* NON VÉRIFIÉ : tout chiffre de temps ou de PPL ; le chemin fla (`decode_static_batch`, `tranches` contiguës) sous rembourrage — le test à sec passe par `decode_static` créneau par créneau (fla absent du venv) ; Mamba2/KDA réels (montage « état mobile » seulement).

## Première fenêtre carte (≤ 30 min, verrou, `regime_ligne()` en tête, `ACVRAM_HYBRID_SLOTS=16`)
1. Qwen3.8-27B-nvfp4 (chemin GDN, fla) : capture godets {1, 2, 4, 8, 16} avec `ACVRAM_TRACE_STEPS=1` — cinq lignes « capture clé », zéro repli eager ; puis lots réels 3, 5, 9 : clé (4|8|16), aucune recapture.
2. `scratchpad/ppl-decode-kv-17-09.py` (Qwen3.8-27B, prefixe 8192, notes 512) sous défaut puis `ACVRAM_GODETS_B=0` : deux PPL identiques à ± 0,002 (b=1 : même clé (1,…), le contrôle de l'instrument) ; puis la variante à 3 séquences (godet 4) : PPL de la séquence notée = même valeur à ± 0,002 — c'est LA mesure du chantier ; si elle diffère, refaire sous `ACVRAM_GDN=torch` pour séparer fla du rembourrage.
3. Drain 12 → 1 (`mesure-debit-concurrent-13-09.py`, 12 invites, longueurs échelonnées) godets vs `ACVRAM_GODETS_B=0` : captures, replis eager, temps total ; seuil proposé ci-dessus, à faire valider par poste7 avant de lancer. Nemotron-3.5-Lightning-30B-A3B-nvfp4 en second si le temps reste (Mamba2 : l'état du fantôme bouge, c'est lui que la correction de ce soir protège).
