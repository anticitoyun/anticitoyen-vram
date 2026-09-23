# banc-etroites-noyaux, (c) fp8 rejoué (correctif stride) — RÉFUTÉ sur (b) et (c) — 22/09 (Manon)

* instrument : `outils/carte.sh .venv/bin/python outils/gpu/mesure/banc-etroites-noyaux.py --rep 200`, sha `1fe613a6` (correctif B colonne-majeure)
* scellé (Océane) : (c) FP8 `_scaled_mm` 1,3-1,8 To/s ; réfuté si aucun noyau >1,0 To/s
* mesuré : (c) ne crashe plus, mesure aboutie — **0,605 To/s (qkv), 0,304 To/s (o)**, très en dessous de la prédiction (1,3-1,8), et **pas au bit** (écart 0,039/0,041, disqualifiant même sans le débit). (b) inchangé vs l'essai précédent : 0,935/0,549 To/s, pas au bit (écart 0,013/0,017).
* verdict : **RÉFUTÉ sur (b) ET (c)** — le défaut (a, triton int8 groupe) reste le seul noyau au bit et le plus rapide des trois (1,185/0,823 To/s). Aucun noyau alternatif ne dépasse 1,0 To/s. Le noyau n'est pas le levier sur ces formes.
* durée : <1 min de carte

## Suite
Pièce 43 fermée : (a) reste le défaut, aucune alternative testée (b, c) ne bat ni le débit ni l'exactitude au bit.
