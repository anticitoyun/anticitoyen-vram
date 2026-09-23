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

## Verdict — 23/09 10 h 3x (poste1)

* **instrument** : code `bb6ec0c3` (noyau `nvfp4_gemv_marlin_kernel` à largeur stockée `ldn`, `nvfp4_gemv_marlin_w13`, w13 au chargement, GEMM w13 au préfill et au décodage tensor) ; `tests/test_moe_w13.py` ; empreinte GEMV contre `origin/main` ; KL par `kl-b.py` de la p81 (inchangé) ; serve sous nsys, lecture sur le train de 509 pas de la 79 (`scratchpad/poste1-p82-23-09/`)
* **commit** : bb6ec0c3 ; alias `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`
* **régime** : -lgc 2700 (2 667-2 669 MHz côté client) ; compute-apps début = fin = llama-server 4627 ; `experts_layout=marlin-w13` lu sur la ligne de régime du bras w13
* **scellé** : table « Prédiction et seuils » ci-dessus (commit 6ae… avant tout code)
* **mesuré** :

| critère | prédit / seuil | mesuré | issue |
|---|---|---|---|
| GEMV w13 contre GEMV séparé | au bit | **au bit** (b = 1, 3, 5, 12 ; bf16 et fp32 ; SiLU et GELU) | tenu |
| chemin par défaut | au bit d'avant | **35/35 sorties GEMV identiques à `origin/main`** ; suites GEMV et tensor vertes (48 passés) | tenu |
| tensor w13 contre séparé | 0 ligne hors 2⁻⁷ | **0** (b = 8, 12, 16), témoin négatif vu | tenu |
| Marlin MoE servi, b = 12 | 2,65-2,70 ms/pas ; > 2,78 réfute | **2,823 → 2,622** (58,8 → **54,6 µs/couche**, 144 → 96 lancements) | tenu, mieux que prévu |
| pas servi (train de 509) | 5,90-5,95 ; gain < 0,10 réfute | **6,084 → 5,907 (−0,177 ms/pas, −2,9 %)** ; glue MoE +0,025 (moe_act avec échelles) | tenu |
| t/s servis sous nsys, fenêtre 5 s | — | 1 838,9 → 1 857,2 (+1,0 %), fenêtre courte, un seul lot : la cellule HTTP de poste2 tranchera | indicatif |
| **KL b = 12**, kl_max par invite | ≤ 0,74 et ≤ +0,05 sur le témoin | témoin W13=0 (même commit, **= p81 au chiffre près**) 0,020 · 0,073 · 0,411 · 0,036 · 0,096 ; **w13 0,035 · 0,067 · 0,934 · 0,073 · 0,123** | **violé** (invite 2 : 0,934 > 0,74) |
| **KL b = 1** | idem | témoin 0,007 · 0,119 · 0,519 · 0,068 · 0,233 ; **w13 0,008 · 0,096 · 0,488 · 0,166 · 0,089** | **violé** (invite 3 : +0,098) |

* **verdict : vitesse TENUE, qualité REFUSÉE au critère écrit.** w13 ne passe **pas** au défaut. Il reste en opt-in
  `ACVRAM_MOE_W13=1` : le code est au bit sur tout le chemin par défaut, et le GEMV w13 est au bit du chemin séparé.
* **durée** : prévue ≤ 10 min de carte ; tenue poste1-p82-tests=tenue=41s poste1-p82-tests-r=tenue=30s poste1-p82-empreinte=tenue=27s poste1-p82-kl=tenue=33s poste1-p82-kl0=tenue=34s poste1-p82-serve=tenue=134s  (tests, empreinte, KL, témoin KL, serve).

### Où est l'écart de KL, et ce qu'il dit

* Le déplacement est au **pas 0**, c'est-à-dire aux logits de fin de **préfill** (invite 2 : 0,41 → 0,93 ; tous les
  autres pas de cette invite sont inchangés à 10⁻³ près). À b = 1, le décodage passe par le GEMV w13, **au bit** :
  les changements à b = 1 viennent donc du seul préfill w13. Trois invites y gagnent, une y perd.
* C'est la signature d'une perturbation d'un ulp sur up (un arrondi bf16 de plus) à des positions presque à
  égalité (argmax 7/8 déjà dans le témoin, sur les mêmes invites). Ce n'est pas un défaut de calcul : le tensor
  tient 0 ligne hors 2⁻⁷ et le témoin négatif est vu. Mais mon critère « +0,05 » était écrit pour un chemin
  exact, et il est violé.
* **Faute de protocole** : j'ai d'abord comparé à la p81 (autre commit) sans témoin au même commit ; le témoin
  W13=0, joué ensuite, reproduit la p81 exactement. La conclusion ne change pas, mais le témoin aurait dû être dans
  la même prise.

### Deux sorties, à trancher par l'utilisateur (chef)

1. **Mode « ± 1 ulp » (REGLES § 1, 22/09)** : w13 opt-in jugé par la PPL avec erreur-type et une KL contre le chemin
   exact, cellules étiquetées « ± 1 ulp », jamais agrégées. Gain servi −2,9 % du pas, Marlin sous vLLM.
2. **Forme exacte, pièce proposée** : une échelle globale **par demi-colonne** dans l'épilogue du port Marlin
   (`kernels/marlin_port`, g_gate pour les colonnes < N, g_up au-delà). Up redevient `bf16(acc · g_up)` comme dans
   la GEMM séparée. Prédiction : KL = témoin à 10⁻³ près. Au bit contre les GEMM séparées si l'ordre de réduction
   du Marlin ne dépend pas de N ; sinon ±1 ulp, à mesurer. Même gain de vitesse attendu. Coût : une modification
   du noyau vendu (vLLM, Apache), ≈ 2 h.
