# Pièce 90 — `graphes=on` ment quand des clés sont refusées (Laure, 23/09)

instrument : `tests/test_regime_graphes_vivants.py::test_graphes_on_nomme_les_cles_refusees_a_cote`
régime : à sec (faux GraphRunner), pas de carte — la seule mesure de carte a été le
`uptime`/`nproc` avant de jouer sous `ACVRAM_TESTS_PENDANT_MESURE=1` (voir § carte ci-dessous)
scellé : le test doit casser sur le code d'avant pour la bonne raison (refus_n forcé à 0)
mesuré : cassé une fois (refus_n=0 codé en dur) → `graphes=on(refus=3:...)` absent, test rouge ;
restauré → vert. 17 tests ciblés graphes/compteurs/régime verts, 38 tests /metrics verts.
verdict : défaut confirmé et corrigé
durée : aucune prise de carte

## Cause (reste (c) de la pièce 88, Océane)

`Engine.regime()["graphes"]` lit `self.graphs.enabled` — vrai tant qu'AU MOINS UNE clé de
graphe est vivante. Depuis le tri des échecs de capture (pièce 88), une clé PEUT être
refusée (OOM, opération non capturable) sans que `enabled` bascule : les autres clés restent
capturées. `regime_ligne()` continuait donc d'afficher `graphes=on` nu, alors qu'un bras entier
sert en eager sans que la ligne ni `/metrics` ne le disent — seul `repli_eager` (compteur brut,
sans détail par clé) bougeait.

## Correctif

- `acvram/engine/runner.py` : deux clés ajoutées à `regime()`, lues sur
  `GraphRunner._echecs()["refusees"]` (dict clé → raison, déjà écrit par la pièce 88) :
  - `graphes_refus_n` = nombre de clés refusées ;
  - `graphes_refus_principale` = raison la plus fréquente parmi les refusées
    (`_raison_principale_refus`, `Counter.most_common`).
- `regime_ligne()` : quand `graphes` est vrai ET `graphes_refus_n > 0` :
  `graphes=on(refus=N:raison principale)` au lieu de `graphes=on` nu. Sans refus, la ligne
  est inchangée (contrôle négatif : `test_graphes_on_sans_refus_reste_nu`).
- `acvram/server/app.py::metrics()` : `graphes`, `graphes_refus_n`, `graphes_refus_principale`
  exposés en champs structurés (même source `engine.regime()` que `regime_ligne`, pas une
  seconde lecture qui pourrait diverger).

## Carte

Le verrou était tenu par Manon (p89-b12-abba, mesure ABBA) au moment de jouer les tests ; ils
sont purement CPU (faux GraphRunner, aucun modèle chargé sur carte). `uptime` : load1 2,18 sur
8 cœurs (< nproc/2), donc `ACVRAM_TESTS_PENDANT_MESURE=1` utilisé en connaissance de cause
(REGLES § 2) plutôt que d'attendre une fenêtre dont la durée n'était pas connue d'avance.
