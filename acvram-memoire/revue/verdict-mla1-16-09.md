# Verdict — MLA commit 1 (d352fe2, `ACVRAM_MLA_BATCH=2`) : pas b=12 43,7 → 21,4 ms (−51 %), 15 534 → 2 541 lancements, logits et PPL bit-identiques ; ≤ 22 ms TENU, ≤ 2 500 dépassé de 41 → marche suivante RoPE + cat

- **instrument** : tests pytest (3 fichiers) ; nsys `--cuda-graph-trace=node`, `nsys-rejeu-b12-15-09.py` 50 pas b=12 ctx 256→306, analyse par poste ; équivalence `ties-moe-decode-15-09.py` (12 × 256 greedy, logits 64 × 12 = 768) + `compare-ties-ulp` (poste1, 5 ulp) ; PPL décodage `ppl-narrow-b12` (12 × 2 047, 3 tranches, `PPL_MODEL`) ; sorties `scratchpad/mla1-16-09/`
- **commit** : arbre mesuré **travail/poste3-qa @ d352fe2** (poste4) ; scripts travail/poste3 c4161fd ; témoin **BATCH=1 rejoué sur le même arbre** (à code égal)
- **régime** : `-k48`, prefill W4A16, décodage MMA=1 MIN_T=5 (PREUVE lue : `_MLA_BATCH_lu` 2 / 1), graphes actifs, lot 12, une carte
- **scellé** : poste7 : lancements ≤ 2 500, pas ≤ 22 ms ; poste4 : 2 200-2 600, 24-28 ms ; moi : 2 300-2 800, 22-27 ms, top-1 ≥ 99,5 %, PPL B/A 1,000 ± 0,002
- **mesuré** : tests **16/16** ; **21,44 ms** (témoin 43,66 ; 670cd63 43,87) ; **2 541 lancements/pas** (témoin 15 534) ; trou 5,16 → **1,17 ms** ; équivalence **768/768, max |Δlogit| = 0, 12/12 séquences bit-identiques** ; PPL 3 tranches **B/A = 1,000000** (8,000319 / 9,733779 / 8,178649 identiques au 10⁻¹²)
- **verdict** : pas **TENU** (21,44 ≤ 22), lancements **dépassé de 41** (2 541 > 2 500) → par la règle de poste7, marche suivante nommée : **RoPE + cat en un noyau** ; numérique **identique** (le commit ne change pas un bit de sortie) ; poste4 et moi réfutées sur le pas (trop pessimistes), tenues sur les lancements ; commits 2 et 3 peuvent partir

## 1. Postes avant / après (ms/pas, même arbre, même instrument)
```
poste          BATCH=1   BATCH=2   lecture
elementwise     12,58     1,35     les 12 boucles par séquence sont parties ; reste ~2 000 lancements : add résiduel 498, binaires 425, copies 830, gather 265 (RoPE, cat, résidu)
denses          11,47     5,85     projections MLA au lot 12 : int8_gemv<4,12> 388/pas 4,67 ms + cuBLAS 1,0 ; 2,0 Go int8 à 435 Go/s : LE poste suivant (narrow_gemm int8 M ≤ 16 = ce que 0.6.7 met ON sur Coder, −16 %)
mla              6,40     6,45     scores 2,75 + reduce 3,64 (inchangés) + mla_ecrit_latent 0,06 (remplace 1 104 copies)
moe_gemm         5,18     4,87     inchangé
normes           1,69     0,37     rmsnorm 2 123 → 406/pas
trou             5,16     1,17     lancement : 15 534 → 2 541
total           43,66    21,44     vLLM même lot : 15,1
```
- Ce qui sépare encore de vLLM (6,3 ms) : denses 5,85 (int8 GEMV lent au lot 12 — `ACVRAM_NARROW_GEMM=1` à essayer sur GLM : 3 min de nsys, non fait ici, hors ordre), élémentaires 1,35 + trou 1,17 (la marche RoPE + cat), attention MLA 6,45 (scores + reduce : le noyau lui-même, commit 2 « attention à une passe »).
- Contexte : ces 21,4 ms sont à ctx ≈ 300 ; en rondes 256→2048 le pas était 56,1 contre 43,9 (+12 ms de KV) — la remesure `certifie` b=12 (10 min) donnera le chiffre publiable du duel après fusion.

## 2. Instrument PPL sur GLM : 4 séquences sur 12 tronquées, silencieusement
`n_jetons_notes` = **22 664 / 24 564** sur les six passes : s1 1 488, s4 1 360, s6 1 632, s10 1 808 (multiples de 16 : blocs KV épuisés), `kv_max_tokens` 25 344 ≥ 24 576 pourtant vérifié — l'assertion de poste2 ne suffit pas sur un MLA (le KV latent n'a pas la géométrie que le plan compte). Le rapport B/A reste valide (mêmes 22 664 jetons des deux côtés, identiques), la PPL absolue ne l'est pas. À ajouter au script : `assert all(n == 2047)` qui rend « faux » au lieu d'un chiffre ; cause à trouver par poste4 (plan KV MLA) avant toute PPL GLM publiée avec cet instrument.
