# Verdict — kv_budget=16384/8 (sage-reprise-ordre-18-09 §2, à sec)

instrument : lecture de code, aucun GPU touché
commit : e812c5a (branche oceane, main fusionné)
régime : à sec
scellé : § 2 — ce que désigne « /8 », fichier:ligne, budget réellement planifié
mesuré : `/8` = `plan.kv_planned_seqs`, posé `tiering.py:465`
  (`plan.kv_planned_seqs = opts.max_concurrent_seqs`) depuis
  `PlannerOptions.max_concurrent_seqs: int = 8` (`tiering.py:160`, défaut de
  classe). Rapporté au log par `runner.py:552`
  (`kv_seqs = getattr(self.loaded.plan, "kv_planned_seqs", 0)`).
  Le profil Coder cité (`verdict-profil-coder-2-17-09.md`) vient de
  `scratchpad/profil-pas-coder-17-09.py:73` : `load_model(MODEL, dtype=...,
  max_model_len=...)` **sans** passer par `auto_plan(..., PlannerOptions(
  max_concurrent_seqs=N))` — replanification interne au défaut 8 — puis
  `Engine(loaded, None, max_batch_size=B, ...)` ligne 75 avec `B=12`.
verdict : (b) — le champ n'est pas mal nommé, il rapporte fidèlement la
  valeur avec laquelle le budget VRAM a été dimensionné (8, jamais relevé
  à 12). C'est le script de profilage qui rate le garde-fou de REGLES §6
  (`load_model` sans `Plan` explicite = même faille que `loader.py:1459`,
  pas re-corrigée ici). Pas de troncature ce run : 16384/8 = 2048 jetons
  planifiés par séquence, contre ~330 réels servis par les 12 ; un lot
  plus long tronquerait en silence sous ce même script.
suite : passer `scratchpad/profil-pas-coder-17-09.py:73` par
  `auto_plan(spec, rig, PlannerOptions(max_model_len=..., max_concurrent_seqs=B))`
  + `load_model(plan=...)`, comme le remède déjà écrit dans REGLES §6.
