# GDN par flash-linear-attention — implémentation à sec (poste4, 17/09), en attente de la fenêtre carte de poste3

Commande : poste7-priorite-apres-campagne-17-09 § 2 (Qwen3.8-27B b=12 : 97 j/s
pour 621 chez vLLM ; la récurrence tournait en torch, séquence par séquence).

## Fait

1. `flash-linear-attention 0.5.2` (+ fla-core, einops) installé dans `.venv`
   par `uv pip install` (le venv n'a pas de module pip). Imports contrôlés :
   `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule`,
   `fla.ops.kda.{chunk_kda, fused_recurrent_kda}`,
   `fla.ops.simple_gla.{chunk_simple_gla, fused_recurrent_simple_gla}`.
   **JIT sm_120 non contrôlé à sec** (pas de carte) : c'est le premier
   point de la fenêtre de poste3 (faux si l'import ou le JIT échoue).
2. `engine/gdn.py` :
   - `ACVRAM_GDN = fla (défaut) | torch` (« 1 » vaut fla, « 0 » reste le
     refus des hybrides dans quant/gguf.py) ; `_fla()`, `gdn_regime()`
     (voie EFFECTIVE : fla | torch | torch(fla absent) | torch(sans carte)).
   - prefill : `chunk_gated_delta_rule` ; décodage : `fused_recurrent_gated_
     delta_rule` — entrées fp32, disposition d'état [B, H, K, V] identique à
     la référence transformers (qui l'a portée de fla), `TRITON_F32_DEFAULT=
     ieee` posé par fla (exact, pas TF32).
   - décodage du LOT : `forward_batch(h, etats)` (eager, b séquences, un
     lancement) et `decode_static_batch(h, statics)` (graphes) ; les
     créneaux fixes sont des VUES dans un tampon groupé de 16 (`new_static`,
     `_LOT`) : les créneaux 0..b-1 sont des tranches contiguës, aucun
     rassemblement ni redistribution d'état, `S.copy_(S_new)` en un
     lancement. Projections, convolution causale et portes sur le lot
     (GEMM à M = b).
   - **Piège trouvé** : depuis que fla est installé, transformers enveloppe
     `torch_chunk_gated_delta_rule` (`integrations/hub_kernels.py:885`) et le
     renvoie vers fla, silencieusement — la voie « torch » aurait été fla.
     `_refs()` prend `__wrapped__` (la fonction nue).
3. `engine/model.py` : `DecoderLayerGDN.forward` batche le décodage eager
   pour tout `linear_attn` qui expose `forward_batch` (avant : MLA seule) ;
   `_la_decode` (graphes, b > 1) appelle `decode_static_batch` pour GDN.
4. `regime.py` : `Variable("GDN", "fla", ("acvram.engine.gdn", "_GDN_VOIE"),
   "torch")` ; `regime_ligne()` écrit TOUJOURS `ACVRAM_GDN=<voie effective>`
   (défaut compris) ; `runner.regime()["gdn"]` et sa ligne aussi.
5. `tests/conftest.py` : sous l'interpréteur, l'autoréglage de fla est réduit
   à sa première configuration (le banc échoue sans pilote : « 0 active
   drivers »). Le noyau récurrent tourne sous l'interpréteur ; le noyau par
   blocs y bute (`chunk_delta_h.py:165`, `i_t.to` sur un int) → jugé sur
   carte seulement.

## Juge (même commit, règle 9) : `tests/test_gdn_fla.py`

- décodage fla après un prefill torch (40 → 64 jetons) contre tout-torch :
  écart relatif 1,9 × 10⁻⁷ (seuil 2⁻⁷) ; état final idem ;
- **bras cassant de poste7** : l'état passé à fla avec l'axe [K, V] inversé
  rend rouge (d_k = d_v : la transposition est silencieuse par la forme, le
  test la voit par la suite de 24 jetons) ;
- lot b = 3 en un lancement (`forward_batch` et `decode_static_batch`)
  contre b appels torch : sorties et S à 2⁻⁷, état de convolution à 10⁻⁵
  (GEMM M = b contre M = 1, pas le même bit) ; créneaux 0..b-1 contigus ;
- 17 créneaux → deux lots, vues sur le tampon ;
- la ligne de régime d'un MOTEUR CHARGÉ à sec porte `ACVRAM_GDN=fla`
  (fixture `converted`, `engine.regime_ligne()` / `regime()["gdn"]`), puis
  `torch` sous `ACVRAM_GDN=torch` — le contrôle demandé par chef ;
- sur carte (skip ici) : prefill fla contre torch sur 64 jetons, et
  continuité prefill fla → décodage fla contre tout-torch.
Suite : 786 passed.

## KDA et Mamba2 (second commit) : décodage du lot en un lancement

`engine/lot_etats.py` factorise les créneaux-vues (LOT = 16) ; `kda.py` et
`mamba2.py` gagnent `forward_batch` / `decode_static_batch` /
`peut_batcher_decode` sur `fused_recurrent_kda` (état transposé [V, K] →
[K, V], comme le prefill) et `fused_recurrent_simple_gla` (q = C, k = B,
v = xs·dt, g = A·dt, échelle 1), mêmes interrupteurs que leur prefill
(`ACVRAM_KDA_CHUNK` / `ACVRAM_MAMBA_CHUNK` = 0 : tout torch). À b = 1 sous
graphes, KDA garde son noyau CUDA fusionné (`kda_decode`) ; le lot fla ne
prend que b > 1 (`model.py:_la_decode`). Juge : `tests/test_hybrides_fla_lot.py`
— lot b = 3 contre b appels torch (sorties et états ≤ 2⁻⁷), bras cassant :
l'état KDA passé sans transposition rend rouge. Suite 789 passed.

## Non fait (à suivre)

- fla en bf16 (vLLM) au lieu de fp32 : à mesurer après le scellé.

## Scellés de poste7, rappelés (fenêtre poste3 ≤ 40 min, Qwen3.8-27B calibA)

b=12 97 → ≥ 400 j/s (faux si < 250 : profiler) ; prefill 2 048 hybrides
≥ 5× la voie torch ; PPL ± 0,002 ; b=1 : t/s ≥ 85 ET J/jeton ≤ 0,75 × calibA
(63,8 t/s, dénominateur corrigé par poste7) — si W plafonne des deux côtés,
J/jeton seul (faux si > 0,9×). Bras torch : `ACVRAM_GDN=torch`, même
instrument, même JSON.
