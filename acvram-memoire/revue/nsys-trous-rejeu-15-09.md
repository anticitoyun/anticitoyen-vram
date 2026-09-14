# nsys sur le rejeu b=12 Coder-30B : trou 0,78 ms, et les projections int8 pèsent 40 % du pas

poste3, 15/09/2026, 20:28-20:31 (unité nsys-rejeu-poste3, carte vide 34 °C).
Ordre : poste7 § 5 [`poste7-reprise-15-09-b.md`](poste7-reprise-15-09-b.md).
Protocole scellé : [`protocole-nsys-trous-rejeu-15-09.md`](protocole-nsys-trous-rejeu-15-09.md)
(poste3 5539846). Moteur : main 77b5def (v0.6.5 ; `[PREUVE] MMA=True MIN_T=9
ROUTE_PACK=True`, graphes actifs, lot 12 tenu). Incident : premier lancement
à 19:59 sans trace — `/tmp/nvidia` appartient à root (session ncu), nsys n'a
pas pu écrire ; `NSYS_TMPDIR` posé dans mon scratchpad, relancé.

## En-tête (REGLES §3)

    instrument   nsys 2026.3.2, `--cuda-graph-trace=node -t cuda,nvtx` ; une carte
    fenêtre      50 pas de décodage purs b=12 (12 invites de 256, ctx 2048),
                 30 pas de chauffe avant ; `synchronize` par pas (la plage NVTX
                 = pas GPU + retour hôte, publié à part)
    analyse      par plage `pas` : union des intervalles de noyaux (pas de
                 double compte des flux) ; trou = plage − union ; médiane sur 50
    données      scratchpad/nsys-rejeu-15-09/{rejeu-b12.nsys-rep,rejeu-b12.sqlite,trous.txt}

## Résultat

    pas (médiane, avec sync)     12,295 ms      (chrono hôte : 12,308 ms/pas)
    union des noyaux             11,477 ms      (= somme : aucun recouvrement, un seul flux)
    TROU (pas − union)            0,783 ms      min 0,737, max 0,980 ; 1 505 noyaux/pas → 0,52 µs par noyau

    poste           ms/pas   part    noyau dominant
    projections     4,599   40,0 %   int8_gemv_kernel<4,16> (q/k/v/o int8, M=12)
    moe_gemm        3,897   33,9 %   nvfp4_gemm_grouped_mma2_kernel<16,4,128> (3 GEMM)
    attention       0,953    8,3 %   paged_attn_partial (0,80) + kv_write_int8 + reduce
    route+pack      0,685    6,0 %   moe_route_pack (0,34) + moe_route (0,29) + scatter
    elementwise     0,568    4,9 %   copies/direct_copy/compare torch (≈ 4 par couche)
    normes          0,227    2,0 %   rmsnorm_bf16
    lm_head         0,165    1,4 %   cutlass wmma bf16 + splitK reduce
    quant           0,164    1,4 %   nvfp4_quant_act (×2 par couche)
    autre           0,137    1,2 %   rope_inplace
    moe_glue        0,094    0,8 %   moe_reduce_trie + moe_act

## Contre les seuils

poste7 : trous ≥ 1,5 ms → PDL ; < 0,8 ms → fusion normes/routage/quant.
Mesuré **0,783 ms** : sous 0,8 de 17 µs — **la fusion, pas PDL**, mais au
bord : la médiane est sous le seuil, le maximum (0,98) au-dessus. Ma
prédiction (0,9-1,3) : **réfutée** par le bas — le creux par nœud est de
0,52 µs, pas 0,61, et il n'y a pas de dépendance visible en plus (union =
somme, un seul flux, aucun recouvrement).

## Ce que le tableau dit de plus, et qui compte davantage que le trou

**Les projections int8 (q/k/v/o, `int8_gemv<4,16>` à M=12) sont le
premier poste : 4,60 ms, 40 % du pas — devant les 3 GEMM MoE (3,90 ms).**
C'est le poste que `narrow_gemm` (poste4 5802e1b, 12,04 → 9,52 ms)
remplace ; le gain annoncé (−2,5 ms) est cohérent avec ce chiffre
(4,60 → ≈ 2,1). Le trou de 0,78 ms est le troisième levier, derrière
narrow (2,5 ms) et la MoE (3,9 ms, port b12x ≤ 75 µs × 48 ≈ 3,3 ms).
Fusion normes+quant+glue : 0,49 ms de noyaux courts, au mieux −0,3 ms.

## Ce que je ne conclus pas

Rien sur b=1 ni sur GLM ; le `synchronize` par pas ajoute ≤ 13 µs (12,308
chrono contre 12,295 plage) — négligeable. Les noms de noyaux sont dans
`trous.txt` pour vérifier le classement par poste.
