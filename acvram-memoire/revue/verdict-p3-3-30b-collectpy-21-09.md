# P3 (3) Qwen3-VL-30B après correctif collect.py — RÉFUTÉ, mais net progrès — 21-22/09 (poste2)

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-3.sh` (worktree poste2-w-21-09), référence `ref-30b-bf16.pt` réutilisée
* commit : bc2958b8 (moteur) ; alias reconverti avec correctif `collect.py` 5ec5ba95 (`experts_sans_stats=2487/18432`, `verdict-reconversion-30b-vl-collectpy-21-09`)
* régime : `ctx_tenu=non-chauffe deepstack=3 eco=2700(2692) graphes=on(hybrides≤4) kv=int8 masque_images=causal mrope=[24,20,20](interleaved) vision=bf16(eager,transformers=5.17.0)`
* scellé : « non établi pire que 3 % à 2 SE » — tenu ssi `haut_2se_pct` ≤ 3 %
* mesuré (essai 2, valide) : `greedy8_identiques=1/3`, `premier_jeton_identique=3/3`, `ppl_desc_geo_pct=-15,58 %`, **`haut_2se_pct=12,610 %`**
* verdict : **RÉFUTÉ** (12,61 % > seuil 3 %), mais **net progrès** — 12,61 % contre 26,1 % de l'alias cassé (0 % d'experts calibrés) et 21,8 % de l'alias pré-cassure du 20/09 (`verdict-p3-3-30b-665eeacc-21-09`) : la calibration à 87 % de couverture (2487/18432 sans stats) réduit l'écart de plus de moitié vs le pire cas, mais reste au-dessus du seuil. **Défaut d'instrument confirmé et corrigé** : essai 1 (même jour) avait PLANTÉ (`TypeError` sur `mesure-c.py:32`, `force()` incompatible avec le kwarg `depuis_graphe` ajouté par `pipeline.py:81,117`, levier sampler `b3c15a01`) et le script avait silencieusement rejugé un ancien `c-30b.json` périmé (26,1 %, invalidé) — corrigé par `force(logits, seqs, **_kw)` (mesure-c.py) et suppression du json avant chaque prise dans `chaine-p3-3.sh` (un plantage ne laisse plus de fichier à réutiliser).
* durée : essai 1 (invalide) 19 s ; essai 2 (valide) 17 s ; deux corrections poussées avec ce verdict

## Suite
Qualité encore hors seuil : le résidu de 2487 experts sans stats (routage top-8/128 sur un corpus de calibration limité) reste la cause probable, pièce 25 (politique de repli) transmise à poste1. Pointeur transmis à la chef. J'enchaîne sur le trou de poste3 (TRT-LLM) puis la frontière.
