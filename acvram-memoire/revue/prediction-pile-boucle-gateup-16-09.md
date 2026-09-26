# Prédiction scellée — pile vs boucle par expert, GLM alpha-commun 21:56

poste1, 16/09/2026, avant mesure (poste7, poste7-glm-gateup-16-09.md §2.1).

Converti `GLM-4.7-Flash-srcbf16-nvfp4` (21:56:36, alpha-commun experts).
Deux premières couches (0 dense, 1 MoE — `first_k_dense_replace=1`),
16 positions (mêmes jetons que l'équivalence GLM du 15/09), pile
(nominal) contre boucle par expert (repli forcé : `_stack_state="non"`
sur chaque `MoEBlock`, même geste que le monkeypatch de
`test_pile_egale_boucle_a_un_ulp`). Critère `poste7-glm-equivalence §2`
(`acvram.quant.equivalence.verdict_position`, multiplicateur 5 ulp).

## Prédiction

**Conforme** (pile == boucle au critère) : la cause du 1,031 est la
métrique de calibration (alpha commun proche de l'identité mais pas
identique, PPL n'y est pas corrélée), pas un bogue d'application — la
pile fait ce que la conversion a écrit.

**Seuil de réfutation** : une seule position hors critère → bogue
d'application de l'échelle (conversion `forced_scale` ≠ table `[E,K]`
chargée, ou mauvais chemin prefill/décodage) → correctif + re-PPL scellé
≤ 1,010 (réfuté si > 1,015).

## MESURÉ : RÉFUTÉ

8 positions sur 16 en échec (8-15), coupure nette (pas progressive) —
détail : revue/poste1-equivalence-pile-gateup-16-09.md.
