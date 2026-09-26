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

## Addendum 24/09 16 h — bascule au défaut (feu de chef), conditions et prédiction écrites AVANT la prise

Les trois drapeaux passent à 1 par défaut (0 = témoin). Conditions de chef avant le push :
1. **ABBA b=1** Qwen3.8, config Marlin, `frontiere-pas.py` B=1, 300 pas, A (trois drapeaux à 0) B (défaut). FAUX si
   B/A > 1,01. Prédit : **B/A = 0,993-1,000**. À b=1, le créneau unique passe par `decode_static` → `forward` : F4 et F2
   n'y sont pas prises (elles vivent dans `decode_static_batch`) ; seule F5 joue (96 additions de moins, ~1 µs chacune,
   sur un pas de ~13 ms).
2. **Capture des godets** 1 à 8 (`capture-godets.py`, Qwen3.8, défaut + config Marlin) : capture ok, `graphes=on`,
   `repli_eager=0` pour chacun ; ligne de régime relevée. Prédit : 8/8.
3. Suite complète sous mon verrou APRÈS la fusion de la bascule Marlin d'poste1 depuis main (attente de son push).

### Résultats de la bascule (prise b2c507bc, 15:42:00-15:43:22)

1. **ABBA b=1** (pas GPU médian, µs) : A1 13 049,9 · B1 13 043,7 · B2 13 043,7 · A2 13 047,8 → **B/A = 0,9996** (−5 µs),
   dans la prédiction (0,993-1,000), loin du seuil FAUX (1,01). **TENU.** Le trou GPU entre pas vaut 23,4 µs sous B contre
   15,4 sous A. Il reste compris dans le pas, qui baisse quand même : je le note sans l'expliquer.
2. **Godets 1 à 8** au défaut (config Marlin) : **8/8 ok**, lot = godet, 0 repli eager, `graphes=on` ; ms/pas 13,05 · 12,96
   · 13,68 · 13,64 · 13,85 · 14,07 · 14,22 · 14,39. Ligne de régime : `ACVRAM_PROJ_MARLIN=1 … ACVRAM_GDN=fla extension=oui
   torch=2.14.0+cu130 triton=3.8.0 fla=0.5.2 … mla_glue=2 glue=compact(8) prefill_glue=compact` (les drapeaux GDN n'y
   figurent pas : ils sont à leur défaut). **TENU.**
3. Suite complète : en attente de la bascule Marlin d'poste1 dans main.

Limite nommée : alias mixte (`Qwen3.8-27B-unsloth-mixte-i8c`) non mesuré, même code GDN.
3. **Suite complète** (bascule Marlin d'poste1 SUSPENDUE, pièce 157 : ordre de chef, fusion de origin/main tel quel,
   bb5fed65) sous mon verrou, 15:45:37-16:02:36 : HEAD e8b69ac5 **7 échecs, 2 689 verts** ; base bb5fed65 **7 échecs,
   2 679 verts**. **Aucun échec propre à la branche** (différence vide dans les deux sens). Les 7 communs, préexistants
   sur main : `test_cadrage_perplexite::test_la_liste_des_variables_lues_ne_derive_pas`,
   `test_pipeline_decodage::test_pipeline_par_defaut_ids_au_bit_b1_et_b12[12]`,
   `test_prefill_bf16_egale_tout_torch::test_ppl_prefill_bf16_egale_tout_torch`, `test_regressions_0_4_4x` ×2 (préfixe
   publié), `test_scission_moe_au_bit::test_generation_dense_au_bit`,
   `test_troncature_kv_bout_en_bout_146::test_requete_close_quand_le_kv_s_epuise_en_decodage`. **TENU.**
