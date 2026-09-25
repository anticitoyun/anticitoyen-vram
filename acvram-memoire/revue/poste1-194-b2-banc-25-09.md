# 194 (b2) banc — verdict (poste1, 25/09) : α/β sur un second flux, −12,15 µs/couche au bit = 0,583 ms/pas (seuil 0,30) → code

* instrument : `scratchpad/poste1-p194-25-09/banc-ab-flux.py` (16 couches segments int8 16384 × 5120 + α/β bf16 96 × 5120 fp32
  + lecteur, 4 copies en rotation, graphe, médiane de 40 rejeux, deux séries par bras) ; `prise.sh` → `banc.jsonl`, `prise.txt`
* commit : ffece2b3 (poste1-194), ATTENDU vérifié par `prise.sh`
* régime : carte 0 seule, -lgc 2700, b = 8, noyaux servis (`gemm_etroit` compact, `gemv_bf16_etroit` fp32), cpu-safe 100
  début/fin, compute-apps début = fin (4242 seul)
* scellé : `scratchpad/poste1-p194-25-09/scelle.md` (ffece2b3, avant la mesure)
* mesuré : S 74,50 / 74,40 ; P 62,35 / 62,35 ; segments seuls 59,69 ; α/β seul 8,86 µs ; au bit P = S
* verdict : TENU — gain 12,15 µs/couche (prédit 9-14), 0,583 ms/pas ≥ 0,30, 120 × le témoin (0,1) ; au bit TENU
* durée : prévu ≤ 3 min ; tenu 11:44:37-11:44:40, après 892 s d'attente du verrou

Alarme scellée ATTEINTE : α/β seul 8,86 µs < 10 → le banc ne reproduit pas les 14,3 µs servis (trace 185 c) ; le gain servi peut
être plus grand que 0,58. Le gain (12,15) dépasse α/β seul (8,86) : en série, α/β paie aussi sa montée et l'attente de la fin
des segments ; en parallèle ses 3 programmes passent sous les 1 024 des segments (P − segments seuls = 2,66 µs, les deux copies
du lecteur). Suite (feu de chef) : code dans `gdn.py:_projections`, témoin `ACVRAM_GDN_AB_FLUX=0`, test au bit qui casse sans
jointure, capture godets {1, 2, 8, 16}, ABBA servi b=8 et b=1, mixte et Qwen3.8.
