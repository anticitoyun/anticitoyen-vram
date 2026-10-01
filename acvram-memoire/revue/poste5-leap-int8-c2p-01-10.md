# INT8 C2′ (01/10 07:49) — la BORNE A ÉTÉ COMPLÉTÉE APRÈS L'ÉCHEC de 07:06 : 0 hors-borne sur les têtes sans bascule ; banc GO de justesse (29,1 ≤ 30), prédiction de temps RÉFUTÉE (poste5)
* instrument : `tests/test_gdn_etat_int8.py` (C2′ : têtes sans bascule à la borne stricte, têtes avec bascule à part) + `outils/gpu/mesure/banc-gdn-recurrence.py --bras int8` ; prise `scratchpad/poste5-int8-01-10/prise.sh 7df61d4f7`, sortie `prise2.txt`, `banc-int8.json`
* commit : 7df61d4f7 (HEAD asserté par la prise et par le banc)
* régime : carte 0 (RTX 5090), `carte.sh` mesure, cpu-safe 85, horloge non relevée (diagnostic de noyau) ; avant/après : sur la 5090 (bus 00000000:01:00.0) aucun autre PID ; llama-server permanent 4436 et leann-core 386005 (serveur d'embeddings bge-m3, 2,7 Gio) sur la 3080 Ti (bus 00000000:02:00.0, relevé `--query-compute-apps=pid,gpu_bus_id`) : **aucune contamination du banc** (rectification chef, vérifiée ; le premier relevé ne disait pas le bus)
* arbre importé : poste5-leap, prouvé deux fois — `prise.sh` lance depuis la racine du worktree (cd dirname/../..) sans ACVRAM_ARBRE_LIBRE, et la garde d'import (acvram/__init__.py:45-80) REFUSE un acvram d'un autre arbre que celui du cwd ; `gdn_etat_int8` (importé par les tests et le banc) n'existe pas dans l'arbre principal
* scellé : C2′ (revue/poste5-leap-int8-prise-01-10.md, accepté par chef) ; banc : revue/poste5-leap-int8-scelle-01-10.md § 3
* mesuré : 10/10 tests ; µs/couche ci-dessous (médiane = pas sans gel, moyenne = gel amorti, les 48 états gelant ensemble 1 rejeu sur 16)
* verdict : C2′ **tenu**, avec la borne complétée APRÈS l'échec ; **GO banc** (moyenne b=12 29,14 ≤ 30) ; prédiction de temps FAUSSE ; b=1 plus lent, comme prévu
* durée : prévu ≤ 10 min / tenu 24 s (journal `tenue=24s`, 07:49:00 → 07:49:24)
## Justesse (C2′)
| test | têtes×pas | avec bascule de clé (jugées à part) | hors borne stricte sur têtes AVEC bascule (expliquées, PAS « tenues ») | hors borne sur têtes SANS bascule |
|---|---|---|---|---|
| variante processeur | 320 | 0 | 0 | 0 |
| carte, F1 off | 23 040 | 198 (0,86 %) | 2 | **0** |
| carte, F1 on | 23 040 | 198 (0,86 %) | 1 087 | **0** |
Les sorties hors borne stricte des têtes avec bascule tiennent toutes la borne étendue (Δk mesuré) : elles sont expliquées,
pas déclarées tenues. Sur les 22 842 têtes×pas sans bascule, il n'y a aucun hors-borne : rien ne signe un bogue du noyau.
Sont verts aussi : la roue libre (256 pas contre la récurrence exacte, TOL_INT8, sans dérive), la couche entière et
l'export AU BIT.
## Banc (µs/couche, Qwen3.8 HV 48 ; fla σ ≤ 0,06)
| b | fla | int8 médiane | int8 moyenne | gain moyen ×48 | prédit (médiane / moyenne) |
|---|---|---|---|---|---|
| 1 | 4,37 | 4,76 | 6,22 | **−0,09 ms/pas** | 5-9, plus lent : tenu |
| 8 | 32,52 | 15,76 | 20,16 | +0,59 ms/pas | 8-18 / — |
| 12 | 49,25 | 23,29 | 29,14 | **+0,97 ms/pas** | 9-15 / 12-24 : **FAUX** |
| 16 | 65,85 | 30,95 | 38,35 | +1,32 ms/pas | — |
* GO banc à 0,86 µs du seuil. Le gain banc à b=12 (0,97 ms/pas) est sous M2 (2,0-2,4) : le noyau de pas lit ~0,6 To/s,
  il n'est pas borné par les octets mais par le calcul et la latence (4 programmes par tête qui refont chacun les
  normes et les produits du tampon). Un gel, quand toutes les têtes gèlent ensemble, coûte ~94 µs/couche.
* b=1 : −0,09 ms/pas (gel amorti), dans la prédiction.
## Suite (L1 du scellé, poste2) et leviers nommés
Garde qualité court + long à b=12 puis ABBA servi, par poste2 (l'auteure ne couronne pas). Leviers du noyau, à sceller
avant : tampon et normes calculés une fois par tête, et non par bloc de colonnes ; gel plus court (~94 µs/couche, R × 5 produits
matrice-vecteur à 1 programme par SM) ; au service les fenêtres ne s'alignent pas : même coût amorti, sans pic. Aucun
gain servi revendiqué ici.
