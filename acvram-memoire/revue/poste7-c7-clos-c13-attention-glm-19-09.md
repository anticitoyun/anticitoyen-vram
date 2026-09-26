# poste7 — C7 (MLA en FP8) n'est pas le levier du prefill GLM ; l'arithmétique désigne le cœur d'attention en fp32 sans TF32 → C13 (19/09, 20 h 05)

Source : `chantier-c7-19-09` (poste1, `poste1-c7-mla-fp8` 89a3710, `_scaled_mm` opt-in, 31 tests) ; `chantier-c5-19-09` (821a18a) ; `poste1-11` 0ccd17c (P2 au défaut fait, classés inchangés au bit) ; `mla.py` (forward : einsum fp32, TF32 non activé).

## 1. Deux prémisses de poste7 retirées

* « MLA en bf16 » (`poste7-pistes-evolutions` piste 5, `poste7-c9` § 1) : les projections MLA des convertis GLM sont **int8 g128 (calibA) ou i8c**, pas bf16 ; REGLES § 9 interdit le NVFP4, pas l'int8. Une FP8 par jeton (erreur 0,037, ×6 l'int8) remplacerait un int8 par un format moins précis : le ratio ≤ 1,005 est à risque par construction. **C7 fermé comme levier** ; le chemin `_scaled_mm` reste opt-in, témoin.
* « Le MLA domine le prefill GLM » : faux sur les projections (3 TFLOP/pas), vrai sur **le cœur d'attention** : `mla.py` calcule les scores et la sortie en einsum **fp32 sans TF32** — 8,6 TFLOP/pas sur les cœurs CUDA fp32 (~105 TFLOPS) = **≥ 82 ms des 372 ms du pas** (2 047 / 5 502 j/s). Le tenseur core n'y touche pas.
* Trouvaille d'poste1 à garder : la porte A8 ne couvrait pas `kv_a` sur les convertis calibA (pile `q_a + kv_a` empilée) — toute porte MLA mesurée avant est partielle sur ce tenseur.

## 2. C13 — cœur d'attention GLM au prefill

Deux marches, la première ne coûte qu'une ligne :

| marche | quoi | prédiction | scellé (avant) |
|---|---|---|---|
| C13-a | einsum du cœur en **TF32** (portée limitée à `mla.py`, `torch.backends.cuda.matmul.allow_tf32` scoped ou `torch.matmul` avec `set_float32_matmul_precision('high')` autour des deux produits) — entrées à 10 bits de mantisse (plus que bf16), accumulation fp32 | cœur 82 → 25-35 ms ; pas 372 → 315-325 ms → **6 300-6 500 j/s** ; PPL identique | PPL GLM tranche 1 = défaut **± 0,001** ; prefill **≥ 6 200** tenu / < faux |
| C13-b | cœur en bf16 sur un noyau d'attention (SDPA à dims qk 192 / v 128, ou attention paginée Triton étendue au prefill MLA) | cœur → 8-12 ms ; reste du pas à chiffrer par nsys | après `verdict-budget-prefill-glm` : scellé dérivé du budget, pas avant |

Le reste du pas (372 − 82 − 15 de projections ≈ 275 ms : experts, glue, tête, échantillonnage) n'est pas connu : **nsys GLM prefill** (10 min, poste2) avant C13-b. Ma prédiction : experts 60-90 ms, glue 30-50 ms, « autres » > 100 ms — et c'est ce « autres » qui décidera si GLM a un poste caché (comme la tête W8A8 de P2).

## Ordre

* **poste1** — C13-a maintenant (une portée TF32 dans `mla.py`, test à sec : sortie du cœur TF32 vs fp32 sur tenseurs réels, écart ≤ 2⁻¹⁰ relatif ; `regime_ligne()` porte `mla_tf32=1`, défaut 0 jusqu'au verdict) ; C7 : branche gardée, plus de fenêtre carte ; C13-b après le nsys GLM.
* **poste2** — après nsys Coder : nsys GLM prefill 2 047 (10 min, mêmes colonnes, `verdict-budget-prefill-glm-19-09.md`) ; puis C13-a (15 min : PPL GLM tranche 1 ± 0,001, prefill ABAB ≥ 6 200), `verdict-c13a-19-09.md`.
* **chef** — `ETAT` : C7 fermé comme levier (int8 déjà là), C13 ouvert ; `REPRISE` § 10 : ligne GLM corrigée (« MLA int8 g128/i8c, cœur d'attention fp32 sans TF32 = 8,6 TFLOP/pas ») ; INDEX.
