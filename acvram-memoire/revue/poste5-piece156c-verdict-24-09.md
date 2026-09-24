# Verdict — pièce 156 (c) : fusions AU BIT F4, F5, F2 du décodage GDN (poste5, 24/09) — TENU

* **instrument** : `scratchpad/poste5-p156c-24-09/prise.sh` — (tests) suite GDN + 3 bras cassants dans une copie ;
  (abba) `frontiere-pas.py` b=8, 300 pas, en processus, ordre A B B A, puis `jetons.py` (64 jetons gloutons × 8, deux
  compositions, un processus par bras) comparé au jeton
* **commit** : 43d03399 (tests), 1591330f (ABBA ; seul le script de prise diffère) — branche poste5-156c
* **régime** : `Qwen3.8-27B-nvfp4`, config Marlin qualifiée (`PROJ_MARLIN=1 PROJ_MARLIN_DOUBLES= GEMV_MARLIN_V2=1
  GEMV_MARLIN_TPB=1 GEMV_MARLIN_S=0`), b=8, ctx 2048, invite 256, -lgc 2700, cpu-safe 100, régime NOMINAL ; bras A les
  trois drapeaux à 0, bras B à 1 (lus sur la ligne de régime de B : `ACVRAM_GDN_ETAT_EN_PLACE=1`, `…_CONV_FUSEE=1`,
  `…_RES_DIFFERE=1`). Carte 0 seule à nous (llama-server sur la 3080 Ti).
* **scellé** : `revue/poste5-piece156c-scelle-24-09.md` (écrit avant le code)
* **durée** : tests 14:58:52-15:04:27 ; ABBA 15:38:30-15:39:54 (une première ABBA à 15:22 s'est arrêtée après A1 sur une
  faute de mon script, un `grep` sous `pipefail` ; A1 y valait 15 607,7 µs, le même chiffre)

## Au bit

* 36 tests verts, dont `test_gdn_fusions_au_bit.py` : F4 (5 pas, sorties et états `torch.equal`, chemin en place
  vérifié pris), F5 (pile de 3 couches GDN bf16, x et norme finale `torch.equal`), F2 seule et F2 + F4, en fp32 ET en bf16,
  têtes q/k ≠ têtes v.
* Bras cassants, tous ROUGES : F4 `ht` omis (état jamais écrit) ; F5 résidu de la couche précédente perdu ; F2 `tl.exp`
  (approchée) au lieu de `libdevice.exp`. Le dernier montre que le test voit une faute à l'ulp.
* **Rejeu sur le modèle servi : 8/8 et 8/8 séquences identiques, 1 024 jetons**, même invite ×8 et 8 invites distinctes.

## Vitesse (pas GPU médian, µs, b=8)

| A1 | B1 | B2 | A2 | Δ B − A | prédit | seuil (70 % de la borne basse) |
|---|---|---|---|---|---|---|
| 15 608,7 | 14 395,4 | 14 395,6 | 15 606,5 | **−1 212 µs (−7,77 %)** | −950 à −1 200 | −665 |

Trou GPU entre pas inchangé (23,9 → 22,9 µs). A1 ≈ A2 et B1 ≈ B2 à 2 µs près : pas de dérive dans la prise.

## Verdict

* **TENU** : au bit (tests et rejeu), gain −1,21 ms/pas, soit **+8,4 % de débit de décodage à b=8** sur le défaut futur.
  C'est la borne haute de la prédiction, dépassée de 12 µs.
* La prise unique ne sépare pas F4, F5 et F2 (ordre de chef : une ABBA pour les trois). Leur part respective reste
  celle du dossier (0,40 / 0,09 / 0,55-0,65), pas une mesure.
* Hors champ : b=1 (`decode_static`, créneau unique, non touché par F4 et F2) ; alias mixte (même code GDN, non mesuré) ;
  le préfill (chemins inchangés).
* **Proposition au chef** : les trois drapeaux à 1 par défaut, après la suite CI complète et la capture des godets ;
  au bit, aucune KL n'est requise. F1, F3 et F6 (± ulp) attendent son mot.
