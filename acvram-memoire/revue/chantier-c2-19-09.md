# Chantier C2 — prefill : déquant par couche depuis la disposition Marlin vers un tampon bf16 réutilisé + `torch._grouped_mm` (19/09, Océane, À SEC)

Source : `sage-poursuite-chantiers-19-09` § 2 (C2). Branche `oceane-c2-prefill-bf16`. Aucune carte ce soir : ni mesure, ni compilation.

## Objectif
Un seul chemin de poids résident (la pile Marlin, `experts_layout=marlin`) ; au préfill, chaque projection (gate, up, down) est dépaquetée en bf16 dans UN tampon réutilisé puis multipliée par `torch._grouped_mm` (cuBLAS), sans seconde disposition résidente et sans quantifier l'activation. Sortie = celle du témoin `grouped_mm` (mêmes poids bf16 au bit), vitesse à mesurer.

## Prédiction scellée et seuil (copiés de la commande, AVANT toute mesure)
> 21 000-24 000 j/s au prefill Coder 2 047 jetons (Marlin à 47 % du plancher bf16 209,5 TFLOPS → cuBLAS ~70 %) ; seuil unique : ≥ 19 500 j/s tenu, < 19 500 faux ; PPL = défaut à ± 0,0005 (mêmes valeurs déquantifiées, ordre fp différent).

Correction de Sage (relayée 19/09 soir, recopiée telle quelle) : C2 = déquant bf16 + GEMM groupée cuBLAS → **PPL attendue 1,0144 ± 0,001** (témoin grouped_mm bf16 mesuré ce soir par Manon : 1,0144 ; le défaut Marlin est à 1,0155, sa GEMM coûte 0,001), et non « défaut ± 0,0005 ». Le seuil de vitesse ≥ 19 500 j/s reste.

Issues nommées, y compris celle qui gêne l'auteur (REGLES § 3) :
* tenu : ≥ 19 500 j/s ET PPL 1,0144 ± 0,001 → C2 candidat au défaut ;
* faux (< 19 500) — c'est l'issue que les chiffres du dépôt annoncent : le témoin grouped_mm + `nvfp4_dequant` rendait 8 633 j/s (model.py:1996) avec `nvfp4_dequant_kernel` à 50,8 ms par prefill (verdict-profil-coder-2-17-09, 144 appels) et `_grouped_mm` déroulé en 128 `aten::mm` + 128 splitKreduce sur sm_120 (verdict-profil-coder-pas-17-09, GEMM de 128 lignes à ~26 % du pic, model.py:1106-1108). Arithmétique : 48 couches × 3 × 128 × 2048 × 768 × 2 o = 58,0 Go de bf16 écrits par le dépaquetage puis relus par les GEMM ; 12,4 TFLOP ≥ 59 ms au plancher ; ≥ 19 500 j/s = ≤ 105 ms au total, soit GEMM ≥ 90 % du plancher ET dépaquetage à la bande passante crête (≈ 40 ms). Avec la déquant mesurée (50,8 ms) et cuBLAS à 70 % (85 ms) : 136 ms → ~15 000 j/s, sous le défaut Marlin (15 987). C2 ne peut tenir que si `_grouped_mm` fait bien mieux que le 26 % déjà mesuré ;
* PPL hors 1,0144 ± 0,001 : les poids ne sont pas ceux de grouped_mm — impossible si le juge au bit ci-dessous est vert sur carte ; chercher alors du côté de l'activation (rembourrage, AWQ) ;
* valeur d'alarme : > 24 000 j/s = on ne mesure probablement pas C2 (cache de préfixe, autre chemin) → `chemins` du bloc et `_chemin('c2')` assertés avant de publier.

## Ce qui existe déjà (fichier:ligne)
* Disposition unique : `_construire_marlin` model.py:1009-1061 (repack par `preparer_pile`), `_liberer_pile_naturelle` :981-1007 (pile NVFP4 rendue, gabarits vides) ; témoin `grouped_mm` :1448-1457 sur `_pile_bf16` :1081-1100 (`nvfp4_dequant` CUDA, torch à sec).
* Disposition Marlin : `gptq_marlin_repack.cu:117-209` (tuiles 16 k × 64 n, mot t·4 + w, pack_idx {0,2,4,6,1,3,5,7}), lue par `nvfp4_gemv_marlin_kernel` (acvram_kernels.cu:1898-1965, `mb_tuile`) ; échelles `permuter_echelles` marlin_port/__init__.py:118, `traiter_echelles_nvfp4` :139 (S0E5M3, annule s·facteur·2⁷ < 2), `traiter_echelle_globale` :154 (g·2¹¹⁹/facteur).

## Fait ce soir (à sec)
1. `marlin_port.repack_torch` (:248) — jumeau torch du repack CUDA ; `preparer_pile(…, repack=)` (:159) l'accepte → piles Marlin sans carte.
2. `marlin_port.depaqueter_marlin(w, s, g, K, N, out=, noyau=)` (:264) — [N, K] ou pile [E, N, K] bf16 ; torch (référence) et Triton `_depaqueter_kernel` (:432, un programme par tuile, arrondi bf16 RNE explicite `_bf16_rne` :422 parce que l'interpréteur Triton tronque). Échelle = fl(s·g) au bit : (s·facteur) × (g·2¹¹⁹/facteur) × 2⁻¹¹⁹, puissances de deux. Une seule différence possible avec la pile naturelle : les échelles annulées par le repack (0 ici comme dans le noyau).
3. Chemin `ACVRAM_PREFILL_GROUPED=c2` : model.py:1386-1412 (avant `marlin`, sinon `unique` le prend), tampon `_tampon_c2` :1504-1512 — UN par appareil pour toutes les couches et les trois projections, E × max(N·K) bf16 = **384 Mio sur Coder** (128 × 768 × 2048 × 2 o ; les « 1,8 Gio » de la note = les trois tampons gardés ensemble, 1,125 Gio, inutile) ; `c2` sans pile Marlin retombe sur `grouped_mm` nommé (:1448) ; validation à l'import :2109 ; disposition unique exigée (GEMV_LAYOUT=marlin) :2214 ; `_construire_marlin` :1029 ; `regime.VARIABLES` PREFILL_GROUPED (regime.py:89).
4. tests/test_depaqueter_marlin.py (9 tests à sec, 1 carte) : égalité AU BIT (vue int16, -0.0 compris) repack_torch → dépaquetage = `dequantize_nvfp4` sur K=2048/N=768, K=768/N=2048 et 128/64, bras cassant : un quartet retourné change UN poids à la place (n, k) prédite ; les échelles annulées sont la seule différence (bloc ciblé) ; contrôle INDÉPENDANT : l'émulation torch de la lecture de `nvfp4_gemv_marlin_kernel` (validé sur carte par test_gemv_marlin) sur repack_torch rend x·Wᵀ, quartets échangés → faux ; noyau Triton (interprété) = référence au bit, octet d'échelle corrompu → faux ; bloc MoE jouet CPU (`_bloc_moe_jouet`) : chemin `c2` asserté (`attendre_chemin`), sortie ÉGALE AU BIT au témoin `grouped_mm`, tampon = même tenseur entre deux appels et entre gate/up/down, échelles décalées → faux, `c2` sans pile → `grouped_mm` ; import : c2 admis, refusé avec GEMV_LAYOUT=naturel.

Rejouer : `CUDA_VISIBLE_DEVICES= ACVRAM_TESTS_PENDANT_MESURE=1 timeout 600 python -m pytest tests/test_depaqueter_marlin.py tests/test_marlin_port_a_sec.py tests/test_gemv_marlin.py tests/test_marlin_prefill_p1.py tests/test_regime_noyaux.py -q -p no:cacheprovider` → fichier seul : 9 passed, 2 skipped (carte), 4,2 s ; les cinq fichiers : 35 passed, 25 skipped (carte). À part : `tests/test_cadrage_perplexite.py::test_la_liste_des_variables_lues_ne_derive_pas` est rouge et l'était DÉJÀ au HEAD 86e7097 (vérifié sur un extrait de HEAD : DOUBLE_DISPOSITION_DIAG, DUMP_MOE, GEMV_LAYOUT, PREFILL_A8, PREFILL_A8_FMT absents de la garde cli.py:55-70 — aucune variable de C2 ; C2 n'en ajoute pas).

## État
* Fait : dépaquetage exact (torch + Triton), repack torch de référence, chemin c2 + tampon, régime, tests ci-dessus.
* Reste : rien à sec.
* Non vérifié : (a) repack_torch = op CUDA `gptq_marlin_repack` (preuve indirecte seulement par l'émulation du GEMV ; test direct écrit, `test_sur_carte_repack_torch_egale_l_op_cuda`, à lancer sous le verrou) ; (b) le noyau Triton COMPILÉ (REGLES § 7 : l'interpréteur ne prouve rien) ; (c) vitesse et PPL ; (d) VRAM réelle sous le Plan (le témoin grouped_mm allouait déjà 384 Mio transitoires par projection ; c2 en garde un seul).

## Première fenêtre carte (≤ 30 min, sous `outils/carte.sh`)
1. `pytest tests/test_depaqueter_marlin.py tests/test_gemv_marlin.py -q` sur carte : (a), (b), et `chemins` — rien ne se mesure avant leur vert.
2. ABAB prefill 2 047 Coder, `certifie`, 3 bras : marlin (défaut) / c2 / grouped_mm (témoin) ; verdict au seuil ≥ 19 500 ; si faux, profil d'un pas c2 (part `_depaqueter_kernel` contre `aten::mm`) avant toute retouche.
3. PPL 3 tranches c2 contre 1,0144 ± 0,001.
