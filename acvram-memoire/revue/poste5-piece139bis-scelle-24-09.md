# Pièce 139 bis — scellé (poste5, 24/09, écrit AVANT la prise) : PROJ_MARLIN sur l'alias mixte, et part des int8 à b=8

Ordre chef. Le bras acvram de la 139 tournait au régime PAR DÉFAUT (sans `ACVRAM_PROJ_MARLIN`).

## 1. À sec : part du pas prise par les couches fp8 → int8 à b=8

Cellules mesurées au même banc chat (256 jetons, -lgc 2700) : pas = b / débit.

| alias acvram | b=8 t/s | ms/pas | source |
|---|---|---|---|
| `Qwen3.8-27B-nvfp4` (tout nvfp4, nos poids) | 299,1 | 26,7 | 102, `verdict-b8.md` |
| `…-unsloth-mixte-i8c` (10,62 G paramètres en int8) | 245,6 | 32,6 | 139 (c) |
| NInfer (W4A4/W8A8, même point de contrôle) | 463,3 | 17,3 | 139 (c) |

* Passer les 10,62 G paramètres de nvfp4 (5,97 Go) en int8 (10,63 Go) coûte **+5,8 ms/pas** pour +4,66 Go lus, soit
  0,80 To/s au marginal, cohérent avec les étroites int8 de la p57 (0,83-1,11 To/s). Part estimée des GEMV int8 dans
  le pas : 10,63 Go à 0,79-1,11 To/s = **9,6-13,5 ms sur 32,6, soit 29-41 %**.
* Écart à NInfer : 15,3 ms/pas. **La 102 montrait déjà 9,6 ms d'écart SANS aucun int8** (nvfp4 pur : 26,7 contre 17,1).
  Les couches int8 expliquent donc les **~5,8 ms ajoutés** (≈ 38 % de l’écart). Les ~60 % restants existaient avant :
  GEMV W4A16 contre GEMM W4A4 sur les MLP, attention/GDN, hôte.
* Réserve : trois cellules, deux séances (102 le 24/09 vers 03 h, 139 le 24/09 à 10 h), aliases aux poids différents ;
  estimation, pas une trace. Une trace nsys b=8 la confirmerait (hors de cet ordre).

## 2. Prise : A = mixte défaut, B = mixte PROJ_MARLIN

B : `ACVRAM_PROJ_MARLIN=1 ACVRAM_PROJ_MARLIN_DOUBLES= ACVRAM_GEMV_MARLIN_V2=1 ACVRAM_GEMV_MARLIN_TPB=1
ACVRAM_GEMV_MARLIN_S=0`, la configuration de la 142 (disposition unique, v2, préfill exact). Même alias, même banc
(`banc-chat-openai.py`, 256 jetons), A B B A A B B A A B par b (5 passes par bras, ≥ 5 lots chacune), serveur neuf,
-lgc 2700, `--max-model-len 4096`. Preuve de prise : ligne de régime de B avec `+marlin(…)`, sinon la passe est nulle.

**Prédiction.** Marlin ne touche que les poids nvfp4 de N ≥ 2 048 : ici les MLP des couches 0-55 (8,42 Go), pas les
int8. Sur `Qwen3.8-27B-nvfp4` pur, b=8 gagnait +57 %, avec tout le pas nvfp4 concerné (≈ 14,4 Go). Ici 8,42/14,4 = 58 %
de cette part. Gain prédit ≈ 0,58 × 9,7 ms = 5,7 ms/pas :
* **b=8 : B = +12 à +30 %** (272-319 t/s) ; FAUX si < +8 % ou > +40 %.
* **b=1 : B entre −3 et +5 %** (GEMV M=1, v2 à parité) ; FAUX si hors de −8 / +10 %.
* J/jeton suit le débit au premier ordre (même puissance plafonnée) : b=8 −10 à −23 %.

**Issues nommées.**
* La disposition Marlin réduit la capacité KV (142 : 8 × 2 560 refusé). Ici l'alias remplit déjà 26,8 Gio à b=8/4096.
  Si B est refusé au chargement, c'est un refus NOMMÉ : je le publie tel quel, puis je rejoue b=8 à `--max-model-len
  2048` pour les DEUX bras (décidé ici, avant).
* Ce qui me gênerait : gain b=8 < +8 %. La part nvfp4 ne serait alors pas le goulot sur ce pas mixte, et le reste de
  l'écart à NInfer viendrait des int8 et du hors-GEMV, pas des MLP.
