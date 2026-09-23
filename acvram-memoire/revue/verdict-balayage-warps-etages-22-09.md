# Balayage warps×étages étroites int8, sous graphe (banc-etroites-occupation) — RÉFUTÉ, au bit — 22/09 (Manon)

* instrument : `outils/carte.sh .venv/bin/python outils/gpu/mesure/banc-etroites-occupation.py --rep 200`, mesure chronométrée sous graphe CUDA (défaut du script, `--ptxas` non posé)
* commit : main à jour (04683f8b)
* scellé (Océane) : (8,3) ou (4,2) gagnant ; o 11,8 → ≤10,6 µs (≥10 %), qkv ≥5 % ; réfuté si <5 % sur les deux ; égalité au bit exigée
* mesuré : **au bit sur les 12 combinaisons** (`au_bit: True` partout, aucun écart). qkv : 9,15-9,44 µs sur les 6 formes, quasi plat (défaut 4w3s déjà optimal, 9,15 µs). o : défaut 4w3s 9,38 µs, meilleur 4w2s 9,28 µs (**gain 1,07 %**), 2w et 8w tous PIRES (11,2-11,5 µs).
* verdict : **RÉFUTÉ** (sortie de l'instrument lui-même) — aucune forme ne gagne ≥5 % ni sur qkv (gain ~0 %) ni sur o (gain 1,07 %, loin des 10 % prédits). L'occupation warps/étages n'est pas le levier sur les étroites int8 — cohérent avec `verdict-banc-etroites-splitk-22-09-v2.md` (même conclusion, autre paramètre) : l'écart TRT-LLM sur ce poste reste à expliquer ailleurs (réduction finale ou format, note du script elle-même).
* durée : ~1 min de carte

## Suite
Deux leviers occupation/split-K fermés aujourd'hui sur les étroites int8 (warps×étages, split-K). Reste : réduction finale ou changement de format, hors de cette pièce.
