# Verdict — pièce 209 (b) : échelle globale par (expert, colonne) dans le Marlin MoE porté et le GEMV Marlin CUDA — au bit là où c'est au bit par construction, 2⁻⁷ contre la référence sur les piles réelles du Coder ; TENUE (poste6, 25/09)

* **instrument** : `tests/test_marlin_moe_par_colonne_209b.py` (carte) + suites voisines (test_moe_w13, test_moe_tensor_decodage,
  test_gemv_marlin, test_marlin_echelles_157, test_marlin_pile_par_ligne_209, test_moe_tensor_glue_fusee, test_moe_tensor_defaut,
  test_marlin_prefill_p1) ; journaux `tests-b3.txt` (95 verts, 4 skips opt-in, 3 rouges = faute d'aide de test corrigée
  ensuite), `tests-b5.txt` (29 verts après correction) ; prises poste6-p209-b-tests… (verrou, 184 + 12 + 79 + 57 + 53 s).
* **commit** : (b) = ce commit sur poste6-209 (après a256676f8 = (a)).
* **régime** : carte 0, extension acvram recompilée sous verrou (182 s) ; port Marlin recompilé À SEC (27 s, empreinte
  4f93a775b340) après deux fautes de signature (`gs_par_colonne` absent du template ; contrôle placé avant la définition).
* **scellé** : `scelle-b.md` (avant la 1re prise) ; corrigé en cours : le w13 TENSOR par colonne n'est pas au bit du w13 scalaire
  par construction (up prend son g dans l'épilogue au lieu de la correction gu/gg dans moe_act : un arrondi bf16 de moins —
  la 82 ter dit déjà « au 2⁻⁷, pas au bit ») → jugé à 2⁻⁷ contre le chemin séparé, comme test_moe_w13.
* **mesuré** : (1) pile sans écrasement forcée par colonne = scalaire **AU BIT** : tensor sans w13 (godets 8 et 16 : les deux
  branches d'épilogue), GEMV w13 + down (vue d'up de stride 2N), GEMV gate‖up : 3/3 ; w13 tensor par colonne contre séparé :
  0 ligne hors 2⁻⁷ ; (2) pile réelle couche 0 (43 experts + 2) : dépaquetage Triton [E, N] **au bit** de la référence ; GEMV b=1 et
  tensor godet 8 contre la référence fp32 des poids dépaquetés : **0 ligne hors 2⁻⁷** ; (3) pile sans écrasement : g [E] (avant).
* **verdict** : (b) TENUE — le noyau MoE porté et le GEMV Marlin servent une pile à facteur par ligne ; les piles d'avant sont
  au bit d'avant. Suite : (c) scellé KL + ABBA Coder b=1/b=8.
* **durée** : 2 h (dont 6 prises courtes ; deux recompilations).

## Ce qui change
* `moe/marlin_moe_wna16/kernel.h` (macro : `int gs_par_colonne`), `ops.cu` (drapeau = `global_scale` à 2 dimensions, stride(0)
  = ldn exigé, passage à `marlin_mm`), `marlin_template.h` (:82/:286 signatures ; :555 g scalaire = 1 par colonne ; `write` :
  g[expert·ldn + col], geste du dense 101 :1657-1670, dans les deux branches d'épilogue) ;
* `acvram_kernels.cu` : `nvfp4_gemv_marlin_kernel` (`gs_ld0/gs_ld1` : index e·ld + colonne, 0 = par expert), `mb_gs_ld`,
  hôtes `nvfp4_gemv_marlin`, `_gateup`, `_w13` (vue d'up admise) ;
* `moe.py:_construire_marlin` : refus seulement si `echelles_ecrasees(bs, par_ligne=True)` ; w13 par colonne : g13 [E, 2N] et vue
  d'up, `moe_act` sans rapport ; `marlin_port.gemv_marlin_torch` [E, N].
* Non encore branché sur une variable de témoin : c'est l'instrument de (c) (A = préparation d'avant, B = par ligne).
