# Pièce 82 — gate·up fusionnés (w13) dans le chemin tensor, en disposition unique, jugés sur le SERVI — 23/09 (poste1)

## Ce qui change par rapport à la 71 bis

La 71 bis (poste5) a mesuré w13 au banc (−4,0 µs/couche), mais n'a pas pu le mettre en service : la pile w13
**ajoutée** à gate et up demandait +8,9 Go et ne tenait pas en mémoire. Ici, **w13 remplace gate et up**
(disposition unique, comme la pile naturelle rendue du 18/09) :

* **w13** = concaténation, ligne de tuiles par ligne de tuiles, des piles Marlin de gate et d'up
  (`[E, K/16, 2N]` ‖ `[E, K/16, 2N]` → `[E, K/16, 4N]` int32 ; échelles `[E, K/16, N]` ‖ → `[E, K/16, 2N]`). La
  disposition Marlin est locale par tuile de 64 colonnes, donc la concaténation donne un w13 valide ; la 75 l'a
  vérifié dans l'autre sens (A3, découpe d'un w13 vLLM, écart 8,3·10⁻³ en sortie bf16). Les piles gate et up sont
  ensuite rendues.
* **décodage, godets ≥ `MOE_TENSOR_MIN_T` (chemin tensor)** : une GEMM Marlin w13 (N = 1 536) **avec l'échelle
  globale de gate**, puis `moe_act(gs_gate = 1, gs_up = g_up/g_gate)`. La moitié gate est au bit de la GEMM
  séparée ; la moitié up subit un arrondi bf16 de plus, d'où « au 2⁻⁷ » et pas « au bit ». Je ne reprends pas
  l'échelle 1 de la 71 bis : l'échelle Marlin traitée vaut g·2¹¹⁹/facteur, et une sortie brute sans elle tomberait
  vers 10⁻³², au bord des sous-normaux bf16.
* **décodage, godets < `MIN_T`, dont b = 1 (GEMV Marlin gate·up)** : le même noyau `nvfp4_gemv_marlin_kernel`
  reçoit une **largeur de ligne stockée** `ldn` (= N partout sauf ici, où elle vaut 2N). Il lit gate et up dans w13
  avec leurs **propres** échelles globales, donc sortie **au bit** du chemin séparé. Aux godets 1 à 4 rien ne bouge.
* **préfill (GEMM groupée Marlin)** : la même GEMM w13 + `moe_act(gs, e_sorted)` que le décodage tensor.
* **refus nommés** sous w13 : gate/up à entrées distinctes (tables AWQ séparées), MMA C17 (qui lit gate/up
  séparément) ; ligne de régime `experts_layout=marlin-w13`.
* **opt-in** `ACVRAM_MOE_W13=1` : la sortie change au 2⁻⁷ (une optimisation qui change la sortie ne va pas par
  défaut). Passer au défaut serait une décision de l'utilisateur, portée par chef, sur les chiffres ci-dessous.

## Prédiction et seuils, écrits AVANT toute prise

| grandeur | prédit | réfuté / refusé si |
|---|---|---|
| Marlin MoE **servi**, b = 12 (trace nsys sous `serve`, train de décodage de la 79) | **2,877 → 2,65-2,70 ms/pas** (59,9 → **55-56 µs/couche**), 144 → 96 lancements | > 2,78 ms/pas (gain < 0,1 : la 71 bis ne se transporte pas en service) |
| pas servi b = 12 (même trace) | **6,13 → 5,90-5,95 ms/pas** (−0,18 à −0,23) | gain < 0,10 |
| sortie tensor w13 contre gate/up séparés (même entrée, même routage) | ≤ 2⁻⁷·max par ligne ; part hors 2⁻⁷ = 0 | une ligne hors 2⁻⁷ |
| GEMV w13 (ldn = 2N) contre GEMV séparé | **au bit** | un seul bit différent |
| chemin par défaut (`ACVRAM_MOE_W13` absent) | **au bit** d'avant la pièce (le noyau reçoit ldn = N) | un bit différent |
| KL Coder 5 invites contre bf16 (`kl-gabarit`), alias `qkvo-i8c` | kl_max ≤ **0,74** (référence nvfp4 0,735) et ≤ +0,05 sur la référence du même alias | 1 invite au-dessus du seuil 1,0, ou +0,05 |
| mémoire | pas plus que la disposition gate/up (même nombre d'octets, pic de chargement ≤ +1 couche) | OOM au chargement avec llama-server présent |

* **Ce qui me gênerait** : un gain servi < 0,1 ms/pas. La 79 aurait alors donné une fausse piste : « notre Marlin
  servi à 59,9 » ne se traduirait pas en gain de fusion, et l'écart restant à vLLM serait ailleurs que dans le
  nombre de lancements.
* **Seuil `MOE_TENSOR_MIN_T = 8` relu sur le servi** : cellules servies b = 2, 4, 8, 12 avec MIN_T = 2 (tensor
  partout) contre MIN_T = 99 (GEMV partout), sans w13, prédiction écrite dans la section dédiée avant sa prise.
