# AWQ par expert : gate et up à échelles distinctes dans la pile — 15/09 (poste4)

Ordre de chef (chantier 1) ; **patch préparé sur `poste4`, non fusionné** : attend la
note de décision de poste7. Contexte : `revue/awq-pile-15-09.md`, `revue/poste7-glm-awq-pile-15-09.md`.

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

## Reste avant fusion
Note de poste7 ; tests carte ; coût mesuré distinct vs égal (b=12) ; PPL GLM avec échelles
distinctes (poste2).
