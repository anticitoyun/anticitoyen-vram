# Pourquoi Devstral s'écarte de 10 × son témoin sous les morceaux — carte : seules `k_proj` et `v_proj` dépendent du découpage (15 % et 30 % des éléments), par la GEMM cuBLAS bf16 à précision réduite ; `allow_bf16_reduced_precision_reduction=False` la retire entièrement

instrument : `scratchpad/poste6-s1-dense/prise-h4.py` par `carte-h4.sh` (UNE prise `carte.sh` de type mesure à chaque fois), joué à sec sur processeur avant chaque prise ; diagnostic sur la couche 0, aucune génération, aucun texte
commit : poste6-gemma-anneau d88fe8026 (1re prise) et 2983a9af2 (2e ; même moteur, main e9fdc0629 fusionné avant), arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, 15 Mio occupés avant et après, carte libre depuis 18:15, aucun autre poste actif ; Devstral-24B `srcawq-nvfp4` chargé par `load_model` (chemin servi) ; entrée : plongements de 7 865 jetons pseudo-aléatoires (graine 0) après la norme d'entrée de la couche 0 ; torch 2.14.0+cu130
scellé : `poste6-s1-dense-bissection-scelle-02-10.md` (micro-prise, troisième bras, seconde micro-prise — trois sections écrites et poussées avant chaque mesure)
mesuré : 2 prises, 22:18:45-22:19:59 (74 s) et 22:22:05-22:22:43 (38 s)
verdict : **la cause est nommée** — `acvram/kernels/__init__.py:932` (et `:925`) : le `F.linear` du chemin NVFP4 naturel, pris par les projections de N = 1 024 (`k_proj`, `v_proj`, exclues de Marlin par `_PROJ_MARLIN_MIN_N` = 2 048, `:1270`). cuBLAS y change de réduction selon M quand N ≤ 1 024. **Mon hypothèse H4 telle qu'écrite (q_proj et gate_proj ≥ 10 %) est FAUSSE** ; l'attention de la carte ne dépend pas du découpage ; le réglage de poste4 retire toute la dépendance.
durée : 112 s de carte en deux prises (prévu ≈ 60 s pour une)

## Mesuré (M = 7 865 d'un coup contre 4 096 + 3 769 ; « témoin » = le même appel deux fois)

| | prédit | mesuré | |
|---|---|---|---|
| témoins (tous) | 0 | 0 élément différent partout | tenu |
| (i) `q_proj` (N = 4 096), `gate_proj`, `up_proj` (N = 32 768) | 20-50 % | **0 élément différent** | FAUX |
| (i) `k_proj` (N = 1 024) | 20-50 % | **15,12 %** (1 217 375 sur 8 053 760), 91,9 % à 1 ulp, toutes les lignes | sous la fourchette |
| (i) `v_proj` (N = 1 024) | 20-50 % | **29,63 %**, dont 78,7 % à 1 ulp — un écart sur cinq dépasse 1 ulp | tenu |
| contrôle `F.linear` nu, N = 4 096 | même ordre que (i) | 0 | FAUX |
| (ii) `attention()` un appel contre deux morceaux, mêmes q, k, v | < 1 % | **0 élément différent** (et `bas_droite` None contre True : 0) | tenu |
| (iv) `F.linear` nu selon N, réglage par défaut | N ≤ 1 024 : ≥ 10 % ; N ≥ 2 048 : 0 | N = 512 : 12,78 % (seulement les 3 769 lignes du 2e morceau) ; **N = 1 024 : 31,30 %** ; N = 2 048 et 4 096 : 0 | tenu |
| (iv), (iii bis) réglage `False` : produit nu à tout N, `k_proj`, `v_proj` | < 1 % | **0 élément différent** | tenu |
| sortie `False` contre `True`, même M = 7 865 | 15-60 % changent | **0 élément change** | FAUX |
| temps d'un `F.linear` nu à M = 7 865, `False` / `True` | 1 à 2 × | × 0,94 à 1,00 | tenu — mais voir limite |
| (iii) 1re prise : réglage sur `q_proj`, `gate_proj`, nu N = 4 096 | < 1 % | 0 avec comme sans : ce bras ne disait RIEN (je n'y avais pas mis k/v) | non concluant, ma faute |

## Lecture

* **La dépendance à M est dans cuBLAS, mais seulement pour les produits étroits** : le contrôle nu la reproduit à N = 512 et
  1 024, pas à 2 048 ni 4 096. Sur Devstral cela ne touche que K et V — donc ce que l'attention de TOUTES les lignes lit, à
  chacune des 40 couches. La reprise, elle, relit des K/V calculés au M d'origine : d'où un témoin dix fois plus petit.
* **Le réglage `False` ne change pas la sortie du seul tenant** (M = 7 865 : 0 élément) : c'est le calcul des MORCEAUX
  (M = 4 096 ou 3 769) qui passait par une réduction à précision réduite, et qui rejoint alors celui du seul tenant.
  Conséquence : il changera la sortie de toute invite dont la longueur tombe dans une forme « réduite » — je ne sais pas
  lesquelles, seules 3 769, 4 096 et 7 865 ont été vues.
* Tout ce que cette prise a regardé est alors invariant sur carte : q, k, v, attention, gate, up. Reste non regardé :
  `o_proj` et `down_proj` (int8 sur cette couche, produit entier `torch._int_mm`), normes, RoPE, écriture du cache.
* **Limites, dites** : (1) une couche, une séquence, un découpage ; (2) le temps n'a été pris qu'à M = 7 865, forme où le
  réglage ne change rien — le coût sur les formes qu'il change (M = 4 096) n'est PAS mesuré ; (3) cette prise dit quelle
  opération dépend du découpage, pas combien des 0,0428 elle explique.

## Reste (décision : chef)

Prédiction à sceller avant toute suite : avec `allow_bf16_reduced_precision_reduction = False`, la chaîne dense de S1 bis
(B / A1) tombe de 0,0428 à ≤ 2 × le témoin (0,008), peut-être à 0. Il faut pour la jouer un réglage de régime qui pose ce
drapeau au chargement (aucun code écrit), puis : la chaîne dense (≈ 3 min), le débit à M ≤ 4 096 avec et sans, et une garde
de qualité — c'est un réglage qui change des sorties servies, pas un correctif à poser par défaut sans elles.
