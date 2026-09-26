# Chantier C11 — chemin étroit int8 par canal (i8c) pour 2 ≤ M ≤ 16 : la VUE g128 (à sec, 40 min) ; à mesurer : capture godets + b=12 ABAB

Objectif (poste7, `poste7-p2-dec-c11-c6-19-09`) : P2-déc b=12 FAUX (1 062,6 t/s < 1 334, J × 1,30) parce qu'à 2 ≤ M ≤ 16 un poids i8c (groupe = K = 2 048) n'avait que le repli déquant : `gemm_etroit.eligible` refuse G > 128 (tuile K = groupe, 393 Kio de shared à G = K), `torch._int_mm` exige M > 16, `narrow_gemm` CUDA non pris.

Prédiction scellée (poste7, recopiée) : b=12 i8c **≥ 1 334 t/s ET J ≤ 1,02 × A** (A = classé au défaut) → tenu → P2 au défaut ; prédiction : tenu, même coût que le g128 du classé.

## Le geste (le plus court des deux de poste7 : ni noyau, ni ptxas)

`acvram/kernels/__init__.py` `vue_g128(t)` : un INT8Tensor symétrique par canal est VU comme un g128 — même `qweight` (partagé, aucune copie), échelle de la ligne répétée sur K/128 groupes (`scales.expand(N, K/128)`, 128 Kio pour q_proj), zéros à 128 ; construite une fois par tenseur (`t.__dict__["_g128"]`), identité pour un g128 ou un affine. `int8_matmul` la prend pour tout appel 2 ≤ n ≤ seuil GEMV, avant le choix Triton/CUDA — les deux chemins étroits servent alors l'i8c exactement comme le g128 du classé (`y += (x_g·q_gᵀ − Σx_g·z_g)·s_g` avec s_g = s pour tous les g : même arithmétique, sommes par groupe de 128 puis accumulation fp32). `_dequantize_int8(vue) == _dequantize_int8(t)` au bit (test). Compteurs `CHEMINS_INT8["etroit_triton" | "narrow_cuda" | "gemv"]` ajoutés : la preuve du chemin pris se lit, elle ne se déduit pas.

## Preuve à sec

`tests/test_prefill_int8_cublas.py::test_c11_vue_g128_du_poids_par_canal` : codes partagés, échelles répétées, zéros 128, déquant identique, cache, éligibilité étroite (vue oui, original non), identité pour g128/affine.

**Non vérifiable à sec** : `gemm_etroit` sous `TRITON_INTERPRET` rend des valeurs fausses (×10⁷) — vérifié aussi sur un g128 classé, ce n'est pas la vue : REGLES § 7 (« un Triton vert à sec ne prouve rien du noyau »). Test carte écrit : `test_c11_gemm_etroit_sur_la_vue_egale_la_reference_sur_carte` (vue ≈ référence fp32 à 1 %, vue == vrai g128 au bit, témoin cassant échelles décalées).

## Première fenêtre (poste2, ≈ 25 min)

1. `pytest tests/test_prefill_int8_cublas.py -k c11 -q` sous verrou (30 s).
2. Capture godets {1, 2, 8, 16} sur le converti i8c (`scratchpad/capture-godets-17-09.py`, `ACVRAM_MODELE_MESURE=…-qkvo-i8c`) : 4/4.
3. b=12 ABAB harnais égal, A = classé défaut, B = i8c (`ACVRAM_PREFILL_INT8=cublas` pour le préfill, décodage sans variable) ; `regime_ligne()` + `CHEMINS_INT8` en fin de bras (attendu B : etroit_triton > 0, dequant = 0 hors tête). Scellé ci-dessus.
