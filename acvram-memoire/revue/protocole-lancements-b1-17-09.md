# Protocole — lancements par pas et part des petits noyaux, Coder-30B b=1 régime classé 0.6.8 (E triton, C mixte = cuda à b=1, B0)
instrument : `scratchpad/profil-pas-coder-17-09.py b1` (fenêtre pure, graphes, 40 pas profilés, noyaux seuls ; ajouté : lancements par pas total et par poste) — `scratchpad/lancements-b1-17-09/`.
commit : arbre laure (= main, C mixte défaut) ; régime classé `ACVRAM_MOE_MMA=0 ACVRAM_MOE_DECODE_MMA=0 ACVRAM_NARROW_GEMM=1`, défauts E/C/B0.
scellé (Sage) : ≥ 600 lancements par pas ; petits noyaux (glue MoE + norme/rope/act + élémentaire) ≥ 28 % du pas (1,17 ms / 30 % lus sur mon profil du 16/09 à 4,04 ms).
mes prédictions : 900-1 100 lancements (16/09 : int8_gemv 96, gemv_grouped 96, attention 96, route 48, rmsnorm 97, rope 48, copies ≈ 200, scatter/compare ≈ 150, + E réduit l'attention à 96 mais ne change rien au reste) ; pas GPU 3,5-3,8 ms (E : 0,85 → ≈ 0,5) ; petits noyaux 1,1-1,2 ms = 30-34 % ; dense_gemm 1,0 (int8_gemv, cuda à b=1) ; experts 0,9.
falsification : < 600 lancements → Sage réfutée, F n'a pas sa cible ; petits < 28 % → idem ; un poste unique > 50 % des lancements → c'est lui la fusion à faire en premier.
