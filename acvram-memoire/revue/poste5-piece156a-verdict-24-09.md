# Verdict — pièce 156 (a) : une seule marge KV (poste5, 24/09) — TENU

* **instrument** : `scratchpad/poste5-p156-24-09/prise.sh` (tests processeur, 3 bras cassants dans une copie, `chauffe.py`
  de la 146 : défaut servi, 8 × 2 560 jetons, 64 générés chacune, avant puis après, un processus par bras)
* **commit** : 2d647d7f (après), ee779a9e (avant, worktree détaché) ; `carte.sh` 14:48:46-14:58:52
* **régime** : défaut servi, graphes, pipeline, eco 2700 ; carte 0 seule à nous (un llama-server sur la 3080 Ti, bus 02)
* **scellé** : `revue/poste5-piece156a-scelle-24-09.md` + addendum (écrit avant la chauffe)

## Changement

`_marge_carte(capacite, reserve)` = max(1,5 Gio, 5 % × capacité) + réserve de préfill (`loader.py`), lue par
`_borner_kv_par_la_vram` ET `_reajuster_plan` (qui prenait max(2 Gio, 7 %) en dur depuis le 08/09). La borne rend ses
bornes brutes et `_borner_kv_avec_exil` exile le déficit brut d'un coup (addendum : sans cela, le montage du 70B
refusait après 4 tours de 325 Mio).

## Mesuré

| modèle | capacité avant | après | prédit | pic alloué avant → après | nvidia-smi après | exil | finies |
|---|---|---|---|---|---|---|---|
| gemma31 (`gemma-4-31B-it-nvfp4-vision`) | 13 408 | **14 624 (+1 216, +9,1 %)** | 14 300-14 800 | 26,96 → 27,55 Gio | 28 809 / 32 607 Mio | 0/60 | 8/8, 0 tronquée |
| qwen32 (`DeepSeek-R1-Distill-Qwen-32B-…-nvfp4`) | 20 480 | 20 480 | inchangé | 25,17 → 25,17 | 23 591 | 0/64 | 8/8 |
| Qwen3.8 (`Qwen3.8-27B-nvfp4`) | 20 480 | 20 480 | inchangé | 20,56 → 20,56 | 22 075 | 0/64 | 8/8 |

Régime NOMINAL partout, graphes capturés (2 captures, 128 rejeux pour gemma31), aucun OOM.
Tests : 25 verts. Cassants : (i) 7 % en dur → ROUGE `test_l_exil_suit_la_marge_unique` ; (ii) borne en dur → ROUGE
`test_la_borne_du_kv_suit_la_marge_unique` ; (iii) manque sur le budget ramené à 0 → ROUGE les 2 tests du 70B.

## Verdict

* **Prédiction TENUE** : gemma31 +1 216 jetons (prédit +≈ 1 240), dans la fourchette. Pourquoi : c'était la marge de 7 %
  de `_reajuster_plan` (≈ 2,07 Gio sur ≈ 29,6 Gio nets) qui bornait ; à 1,5 Gio, la cession du KV (146, `kv_min`)
  rend 0,57 Gio × 495 360 o/jeton. Qwen3.8 et qwen32 sont déjà à la demande : rien ne bouge, au jeton près.
* **Marge restante sur gemma31** : 3,8 Gio libres à nvidia-smi après la chauffe à pleine demande (pic alloué 27,55 Gio).
* **Risque nommé, non rejoué** : Llama-3.3-70B-nvfp4 (cause historique des 7 %). Le montage processeur du 70B est vert
  sous la marge unique grâce à l'exil du déficit brut. Sur la carte, il n'a pas été chargé.
* Le bead `anticitoyen-vram-ba9` est clos par ce verdict.
