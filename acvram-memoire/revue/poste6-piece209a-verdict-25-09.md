# Verdict — pièce 209 (a) : préparation Marlin des piles d'experts à facteur PAR LIGNE — les 8 piles réelles du Coder-30B sont dépaquetées AU BIT de la référence (0 valeur fausse sur 70,8 M), le témoin sans facteur en a 1 044 552 ; les piles sans écrasement gardent la préparation d'avant au bit (poste6, 25/09, à sec)

* **instrument** : `tests/test_marlin_pile_par_ligne_209.py` (à sec, repack et dépaquetage torch) : pile synthétique sans
  écrasement (w, s, g, dépaquetage identiques à l'ancienne préparation), pile à sous-normales forcées (montage de la 157 :
  lignes × 1e-5) exacte + témoin `par_ligne=False` faux, et les **8 piles réelles** du Coder (experts touchés + 2 témoins,
  safetensors) au bit de `dequantize_nvfp4` ; contrôle chiffré hors test (ci-dessous). Suites voisines : test_marlin_echelles_157,
  test_depaqueter_* : 14 verts, 23 skips (carte) à sec.
* **commit** : (a) = ce commit sur poste6-209 (origin/main b28b45e94).
* **régime** : à sec ; alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c ; une prise nulle (nom de fichier de test faux, 6 min de file).
* **scellé** : la 208 (prédiction : 0 bloc écrasé par ligne → exact).
* **mesuré** : couche 0 gate : 43 experts touchés, 69 059 blocs écrasés par la pile ; **par ligne 0 valeur fausse / 70 778 880**,
  témoin facteur commun 1 044 552 fausses (valeurs des blocs écrasés ≤ 7,8 × 10⁻⁶ pour max |w| 0,275) ; couche 1 up : 1 expert,
  203 blocs, **0 / 4 718 592**, témoin 2 761 fausses ; les 6 autres piles : au bit (tests). g par (expert, colonne) : [E, N].
* **verdict** : (a) TENUE, au bit ; on continue vers (b).
* **durée** : 1 h à sec (dont 12 min de file de carte pour des tests processeur, à cause de la garde pytest).

## Ce qui change (`acvram/kernels/marlin_port/__init__.py`)
* `preparer_pile(..., par_ligne=True)` : si `echelles_ecrasees(bs)` (facteur commun écraserait), facteur `f[e, n]` =
  `_facteur_depuis_max(bs.amax(-1))` (puissance de 2), `bs' = bs·f` (e4m3 exact), assertion « plus rien d'écrasé », puis la
  préparation habituelle sur bs' et `g[e, n] = g[e] / f[e, n]` (×2¹¹⁹/facteur) → g_marlin **[E, N]** ; fl((s·f)·(g/f)) = fl(s·g)
  au bit (puissances de deux). Sinon : code inchangé, g [E] — les 44 autres piles sont au bit d'avant (test).
* `depaqueter_marlin` : torch et Triton acceptent g [E, N] (`PAR_COLONNE` : `g[e·N + n]`) ; CUDA refuse (nommé).
* Rien n'est branché dans le moteur : `moe.py:488` refuse toujours ces piles ; c'est l'étape (b) (échelle globale par colonne
  dans l'épilogue du Marlin MoE, `marlin_template.h:551/1803`), puis (c).
