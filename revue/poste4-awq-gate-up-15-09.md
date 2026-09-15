# AWQ par expert : gate et up à échelles distinctes dans la pile — 15/09 (poste4)

Ordre de chef (chantier 1) ; décision poste7 (a) `poste7-glm-gateup-16-09` ; témoin plancher
d'poste1 conforme (2845b31) → **prêt à fusionner** (16/09 06:45). Contexte : `revue/awq-pile-15-09.md`, `revue/poste7-glm-awq-pile-15-09.md`.

## Ce que le patch change
- Chargeur (`_try_build_stacks`) : plus de refus « gate ≠ up ». Tables égales → une seule table
  partagée (`awq["up_proj"] is awq["gate_proj"]`, `up_distinct=False`, chemin inchangé) ;
  distinctes → `up_distinct=True`. Garde d'unité conservée (table = 1 → None ; TEMOIN=1/2).
- `moe_route_pack(…, awq, awq2=None)` : `awq2` = table d'up → seconde ligne `xs2 = x / s_up[e]`
  écrite dans le même lancement (10e sortie ; sans `awq2`, `xs2` **est** `xs`, même tenseur).
- Décodage MMA : `xq2 = quant_act(xs2)` seulement si distinct, up lit `xq2` ; témoin torch
  identique ; noyau fusionné (b12x) exclu quand distinct (un seul `xq`).
- Décodage GEMV : `x_u = x[tok] / s_up[e]` ; `nvfp4_gemv_grouped_gateup` (un seul x) exclu quand
  distinct → deux `_grouped` (gate sur `x_g`, up sur `x_u`).
- Prefill groupé : `xs_u` distinct, seconde quantification (mma) ou seconde GEMM (direct /
  `_grouped_mm`).
- Coût quand distinct : +1 ligne [G, Hpad] bf16 écrite par route_pack, +1 `quant_act`
  (~2 µs/couche estimés, à mesurer) ; quand égal : 0 (mêmes objets qu'avant).

## Tests (tests/test_moe_awq_pile.py, carte requise — pas encore exécutés, carte tenue)
- `test_gate_up_differents_acceptes` : distinct → `up_distinct`, tables différentes ; égal → alias.
- `test_pile_egale_boucle_a_un_ulp[{mma,gemv,prefill_mma,prefill_direct}-{gate=up,gate!=up}]` :
  8 cas, mêmes seuils (GEMV ≤ 1 ulp ; W4A4 ≤ 1,25 × sans-échelle ; témoin sans table > 2×).
- `test_route_pack_awq_egal_torch` : `xs2` bit à bit = `x / s_up[e]` ; sans `awq2`, même `data_ptr`.
- À sec : `nvcc -c` du .cu passe (sm_120f) ; py_compile ; CPU 10 passed / 16 skipped.

## Tests sur carte (16/09 06:27, poste4 5dcffdf)
`tests/test_moe_awq_pile.py` + route_pack + graphe + fused : **25 passed** (équivalence ×8 dont
gate≠up sur les 4 chemins, `xs2` bit à bit, 8/9 lancements, table d'unité).

## Contrôle du RÉFUTÉ d'poste1 (dd51a3c) : c'est W4A4, pas l'échelle
Son script (`equiv-pile-boucle-glm.py`, converti alpha-commun, 16 préfixes, critère 2 ulp) relancé
tel quel, même carte, même créneau :

| réglage | positions ok | delta | lecture |
|---|---|---|---|
| défaut (MIN_T=9, poste1) | 0-7 ok, 8-15 échec | 0,9-1,8 | coupure = `model.py:1340` `t >= MIN_T` |
| `ACVRAM_MOE_DECODE_MMA=0` | **16/16** | 0,04-0,13 | GEMV W4A16 partout = boucle |
| `ACVRAM_MOE_DECODE_MMA_MIN_T=1` | 1/16 (un ex-aequo) | 0,7-2,3 | MMA W4A4 partout |

La coupure à 9 est la garde de lot : préfixe ≥ 9 → `_forward_grouped_mma`, activations
quantifiées E2M1 bloc 16 (chemin officiel v0.6.1, PPL 0,9995). L'échelle AWQ est appliquée des
deux côtés (sinon MMA=0 n'aurait pas rendu 16/16). Le critère « 2 ulp contre la boucle W4A16 » ne
peut pas être tenu par le chemin W4A4 par construction — c'est la table des trois issues de poste7
(`poste7-glm-w4a4-16-09`) qui tranche : bogue (non), W4A4 coûte (à mesurer en PPL), métrique.

## Pas GLM b=12 (16/09, carte exclusive, `ACVRAM_TYPE=mesure`, fenêtres 25 s ABAB)
Scellé poste7 (a) : B/A ≤ 1,03× (réfuté > 1,05×), coût attendu +0,1-0,3 ms/pas.
A = `GLM-4.7-Flash-srcbf16-nvfp4-sansawq` (aucune échelle, tables absentes) ;
B = `-avant-noawq-experts` (1614/3008 paires gate≠up, tables gate, up et down appliquées) ;
B∅ = B avec tables ignorées (`ACVRAM_MOE_AWQ_TEMOIN=3`, sorties fausses, coût du chemin seul).
Régime NOMINAL sur les six bras (graphes=on, 0 exilé, chemin_moe=mma).

| bras | pas (ms) | t/s | J/jeton | W |
|---|---|---|---|---|
| A1 / A2 | 55,51 / 55,61 | 197,2 / 196,8 | 1,548 / 1,572 | 305 / 309 |
| B1 / B2 | 58,61 / 58,71 | 186,8 / 186,4 | 1,601 / 1,619 | 299 / 302 |
| B∅3 / B∅4 | 57,61 / 57,62 | 190,0 / 190,0 | 1,563 / 1,571 | 297 / 299 |

- **Tel que scellé (B/A) : 1,056× → RÉFUTÉ** (> 1,05), Δ = +3,10 ms/pas.
- Décomposition par le témoin B∅ : **+2,06 ms tient au converti** (B∅ − A, tables ignorées,
  même code) ; **+1,04 ms tient aux échelles** (B − B∅, soit 22 µs/couche : deuxième
  `nvfp4_quant_act`, deuxième ligne de `moe_route_pack`, division dans `moe_act`) — 1,018× de B∅,
  sous les 1,03× si le sans-AWQ de référence est le même converti.
- Le scellé comparait deux convertis, pas deux codes : la part converti (+2,06 ms) n'est pas
  expliquée ici (même nombre de couches, d'experts et de régime ; format des autres tenseurs à
  vérifier par poste2). La part échelles dépasse aussi l'attendu +0,1-0,3 ms (×3-10).
- Journaux : `scratchpad/glm-gateup-16-09.log`, `scratchpad/glm-gateup-temoin3-16-09.log` ; JSON
  dans le répertoire de la campagne.

## Reste
Part converti (+2,06 ms) à expliquer avant de retrancher le scellé ; part échelles (+1,04 ms) à
profiler par lancement (ncu, b=12) ; correctif d'équivalence pile/boucle avec table réelle
(discriminateur d'poste1) ; montée 0.6.6 (MIN_T=5 déjà câblé bdb9f14, narrow OFF, re-tampon OFF
selon `revue/poste7-narrow-verdict-16-09.md`).
