# GLM b=1 A/B, 6f v2 (garde intra-fenêtre) — 21/09

* instrument : `scratchpad/glm-b1-ab-21-09/chaine-6f-v2.sh` (nouveau)
* commit : A = worktree figé 59533b29, B = worktree figé d12b2a08 (arbres chauds)
* régime : B porte `kv=latent-bf16 pipeline=1 eco=2700` ; garde intra-fenêtre identique au protocole Coder (3 échantillons `ps`, rejet si un échantillon > 50 % ou watts moyen > 395)
* scellé : médiane(B) ≥ médiane(A)×1,02
* mesuré : A1 163,8 (etr_max 29,6) · A2 163,9 (etr_max 29,5) · A3 161,3 (**etr_max 558,1 — REJETÉE**) · B1 165,2 (etr_max 29,5) · B2 165,5 (etr_max 29,5) · B3 164,8 (**etr_max 348,7 — REJETÉE**). Watts : A 230,5-232,7, B 233,2-234,6 — tous sous 395.
* verdict : **RÉFUTÉ** — A retenues = [163,8 ; 163,9], médiane 163,85 ; B retenues = [165,2 ; 165,5], médiane 165,35 ; ratio = **1,0091** (< 1,02). Même conclusion que le calcul automatique (qui incluait à tort A3/B3, ratio 1,0085) : le sampler vectorisé gagne ~0,9-1 % sur GLM b=1, pas les 2 % du seuil. A3/B3 rejetées ensemble en fin de fenêtre (15:50-15:52) — charge étrangère montée en même temps sur les deux bras, cohérent avec une reprise d'activité d'un pair (Laure sort peut-être de son trou pip à ce moment).
* durée : 408 s (15:45:31–15:52:19), 6/6 fenêtres mesurées, 4/6 retenues

nvidia-smi propre avant/après.
