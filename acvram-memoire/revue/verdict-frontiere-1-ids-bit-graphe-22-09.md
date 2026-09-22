# Frontière (1) : ids/logprobs au bit b=1 et b=12 sous ACVRAM_SAMPLER_GRAPHE=1 — TENU — 22/09 (poste2)

* instrument : `outils/carte.sh pytest tests/test_pipeline_decodage.py -q` (worktree poste2-w-21-09, main à jour)
* commit : b0c25910 (avant le correctif d'instrument ci-dessous)
* régime : `ACVRAM_SAMPLER_GRAPHE=1`, `sampler=graphe` confirmé sur la ligne de régime, `pipeline=1`, graphes on(hybrides≤12), kv=int8
* scellé (poste1, `poste1-levier-1-conception-21-09` § 4.1) : `test_pipeline_par_defaut_ids_au_bit_b1_et_b12[1]` et `[12]`, ids ET logprobs au bit (`torch.equal`) — un octet change = réfuté, arrêt
* mesuré : `test_pipeline_par_defaut_ids_au_bit_b1_et_b12[1]` PASSED, `[12]` PASSED — **2/2 TENU**. Un 3e test du même fichier (`test_pipeline_bit_identique_a_egalite_pres`, hors scellé, pas b=1/b=12) a échoué : `TypeError` sur son espion `sample_espion(logits, seqs)`, incompatible avec le kwarg `depuis_graphe` que `pipeline.py:81` passe désormais à `_sample_only` — même classe de défaut d'instrument que `mesure-c.py` ce soir (monkeypatch non préparé pour un nouveau kwarg d'API), pas une divergence du sampler
* verdict : **TENU** — ids et logprobs identiques au bit sous le nouveau chemin `sampler=graphe`, b=1 et b=12. Correctif d'instrument appliqué et poussé (`sample_espion(logits, seqs, **kw)`, transmet `**kw` à `orig_sample`) pour que le 3e test redevienne significatif au prochain rejeu — non rejoué ici (hors scellé de cette pièce, pas nécessaire pour statuer sur (1)).
* durée : 84,8 s (1 échec + 2 succès dans la même prise)

## Suite
Frontière (2) : `outils/gpu/mesure/frontiere-pas.py` A/B/B/A sous `ACVRAM_SAMPLER_GRAPHE`, 300 pas, seuil ≥ 30 µs sur `echantillon + trou_gpu`.
