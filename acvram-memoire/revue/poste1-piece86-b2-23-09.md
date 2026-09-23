# Pièce 86 — b=2 sous b=1 en service (poste1, 23/09)

Constat (pièce 85, trois bras) : b=2 99,8-247,5 t/s sur lots de 1 024 jetons, contre ~560 sur les passes courtes du
même palier et 311 à b=1.

## Lecture à sec, avant la prise (relevés /metrics du bras D, `scratchpad/poste1-p85-23-09/metrics-D.jsonl`)

Au palier 2, ~53 pas/s (≈ 19 ms/pas contre ~3,2 en graphe) ; sur 9 s (horodatages 54766-54775) `graphes_replays`
figé à 568 pendant que `steps` passe de 665 à 885, `repli_eager` = 0, `gain_moyen` 3,3-3,5 : des pas spéculatifs
hors graphe, ~40 ms chacun, sans compteur.
Chaîne : `runner.py` `_speculative_decode` — en dense, chaque séquence garde SA proposition n-gramme (0 à k jetons) →
`_build_spec_batch` pose des `query_lens` différents → `graphs.py:573-574` rend False sans `_eager` (« longueurs
mixtes ») → `self.model(batch, logits_positions=…)` en eager. La garde (`speculative.py:85-92`) compte les jetons
émis par pas, pas le temps : 3,5 jetons/pas la garde active alors que le pas coûte ~12 × un pas de graphe.
À b=1, une seule longueur par lot : clé (1, ql, nblk) capturable, d'où b=1 sain.

## H86 et ce qui la réfuterait (écrit avant la prise)

H86 : à b=2, la vérification spéculative à longueurs mêlées tombe en eager muet et coûte plus qu'elle ne rapporte.
Deux bras, serveur neuf chacun, paliers 1 puis 2 (banc de la 85, fenêtre 10 s, 1 024 jetons) :
S0 défaut (lot_max 2) ; S1 `ACVRAM_SPECULATION_LOT_MAX=1` (pas de spéculation à b=2).
Prédit : S0 b=2 ≤ 300 t/s, Δreplays/Δsteps ≤ 0,7 au palier 2 ; S1 b=2 ≥ 500 t/s, Δreplays/Δsteps ≥ 0,95 ; b=1
identique entre bras à ± 5 %.
**H86 réfutée si** : (a) S1 b=2 < 400 t/s (la lenteur n'est pas la spéculation) ; ou (b) S0 Δreplays/Δsteps ≥ 0,9
(les pas spéculatifs passent en graphe) ; ou (c) b=1 diffère de > 5 % entre bras (autre chose a bougé).
Issues nommées : H86' le proposeur coûte en Python (le chronomètre hôte le dirait, pas les replays) ; H86'' capture
d'une clé par (ql, nblk) en cours de lot (graphes_captures monterait au palier 2 — lu, pas prédit).
