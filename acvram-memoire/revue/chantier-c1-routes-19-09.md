# Chantier C1 — deux routes W4A8 chiffrées à sec contre le scellé T_experts ≤ 27,2 ms (0,55 × 49,5 ms Marlin, nsys poste2) : (i) CUTLASS int8 groupé après déquant · (ii) `kind::mxf8f6f4` sur la pile naturelle — **(ii) tel qu'énoncé (B e2m1) est impossible sans requantifier les poids (11,6 % d'erreur : mort) ; la forme viable de (ii) est B en e4m3 dépaqueté en registres (2,15 % d'erreur, 77 % des poids exacts) — plus lente que (i) au pic (FP8 acc. f32 = ½ int8) mais sans tampon transitoire ; aucune des deux ne tient 27,2 avec marge : poste7 tranche** (19/09 soir, ordre `poste7-mesure1-plafond-c17-c1-19-09` § 2)

Base : `chantier-c1-19-09` (branche `poste1-c1-w4a8` 0512a00 : référence torch W8 par ligne F = 127/12, boucle par groupes de 8 experts, point d'appel CUTLASS, squelette de tuile int8) ; `verdict-porte-a8-19-09` (A8 int8 par jeton +0,0006) ; porte W8r +0,0007 (poste2 f5a6545) ; A8 e4m3 tenue (−0,0002, MLA). Données : `donnees-c1-routes-19-09/erreur-b-mxf8f6f4.{py,json}` (72 tenseurs d'experts réels de Coder : couches 0/23/47 × 8 experts × gate/up/down, CPU, 40 s).

## 0. Ce que la MMA 8 bits accepte (PTX, `acvram_kernels.cu:2946-2948`, REGLES § 9)
`mma.sync.m16n8k32.kind::mxf8f6f4.block_scale` : A et B en {e4m3, e5m2, e3m2, e2m3, e2m1} **dans des conteneurs de 8 bits**, échelles **UE8M0 par bloc de 32** pour A et pour B, accumulation f32 seule. Nos poids sont E2M1 × **UE4M3 par bloc de 16** : l'échelle E4M3 (mantisse à 3 bits, 1.mmm) n'a **aucune place** dans cette instruction — ni dans le facteur UE8M0 (puissance de 2), ni dans A (échelle de A = par ligne de A et bloc de k, pas par colonne de B). Deux issues : la replier dans les valeurs de B (requantifier) ou passer B en 8 bits.

## 1. Erreur relative des poids réels sous chaque format de B (Frobenius contre la NVFP4 d'origine, échelle globale fp32 par tenseur laissée à l'épilogue)
| format de B | erreur méd [min-max] | éléments exacts | lecture |
|---|---|---|---|
| (ii-a) **e2m1 requantifié**, UE8M0/32 = 2^⌈log₂(amax₃₂/6)⌉ | **11,6 %** [11,5-12,2] | 21 % | une seconde quantification 4 bits par-dessus la première (la NVFP4 elle-même vaut 9,5 % contre bf16) : PPL prédite ≥ +0,010 (l'A4 à 9 % coûtait +0,010) — **morte**, et elle changerait aussi le décodage (disposition unique) |
| (ii-b) **e4m3**, UE8M0/32 = 2^⌈log₂(amax₃₂/448)⌉ | **2,15 %** [2,09-2,20] | **77 %** | m·(1+f) exact quand m ∈ {½, 1, 2, 4} (1 bit significatif × 1.mmm = 4 bits = e4m3) ; m ∈ {1½, 3, 6} (1.1b) × 1.mmm = 5 bits → arrondi ≤ 2⁻⁵ sur 23 % des éléments ; l'exposant relatif des deux blocs de 16 passe dans l'exposant e4m3 (exact) |
| (i) **int8 par ligne** F = 127/12 (fiche C1) | **0,85 %** [0,77-1,13] | 0 % | ce que la porte W8r a mesuré (+0,0007) |

(ii-b) est 2,5 × l'erreur de (i) et du même ordre que l'A8 e4m3 des activations (1,2-2,1 %, porte tenue). **Prédiction PPL (ii-b)** : +0,001 à +0,004 contre le défaut (scellé ≤ 1,020 pour une base 1,0148 : marge +0,005 avec l'A8) — issue qui me gênerait : > +0,004, alors la disposition unique naturelle se paie en qualité et (ii) meurt comme (ii-a). À mesurer par fausse-quant (une variante e4m3/UE8M0 de `fake_quantize_w8_row`, 20 lignes, carte 15 min, poste2) AVANT toute tuile.

## 2. Vitesse : les deux routes contre 27,2 ms (Coder prefill 2 047 jetons, experts 7,4 T-op, 16,3 Go de poids)
Pics 5090 (fiche C1 et REGLES § 9) : bf16 acc. f32 209,5 TFLOPS (plancher mesuré), **int8 838 TOPS (acc. s32)**, **FP8 acc. f32 = 419** (la MMA mxf8f6f4 n'accumule qu'en f32 : demi-débit GeForce), Marlin mesuré 7,4 T-op / 49,5 ms = 150 TFLOPS = **72 % du plancher bf16**.

| | (i) CUTLASS int8 groupé après déquant | (ii-b) mxf8f6f4 A e4m3 × B e4m3 dépaqueté en registres |
|---|---|---|
| GEMM au pic | 8,9 ms | 17,7 ms |
| GEMM à 72 % (l'efficacité de Marlin) | 12,3 ms | **24,5 ms** |
| tampon transitoire | int8 par ligne : 0,56 Gio écrits + 0,32 lus par couche → **12-15 ms** si le L2 sert la relecture par groupes de 8 experts (ncu ≤ 1,2 × tampon), **25-30 ms** sinon | **aucun** : B lu en quartets depuis la pile naturelle, déplié en e4m3 dans la tuile |
| coût du dépliage B | dans le tampon (ci-dessus) | par élément : extraction du quartet, niveau (LUT 8), × 2^(e−e_max), cvt e4m3 ≈ 4-5 instr ; par MMA m16n8k32 : 8 éléments/fil → ~36 instr **amortis sur BM/16 tuiles de M** : à BM = 128 (8 tuiles) ≈ 4,5 instr/MMA contre ~33 cycles de créneau MMA à 419 T-op/s → **+15 % d'émission**, tenable ; à BT = 16 (queues de routage) le dépliage domine (×8) |
| quantification de A (e4m3 ou int8 par jeton, UE8M0/32 pour (ii-b)) | 2,2 G éléments par prefill (16 376 lignes × (2 048 + 768) × 48 couches) = 6,6 Go de trafic → **≈ 5-7 ms** non fusionnée, ≈ 1 ms fusionnée dans `moe_act`/le rassemblement | idem |
| **prédiction T_experts** | **24-28 ms** (L2 tenu) / **37-44** (L2 faux) | **30-34 ms** |
| contre 27,2 | tenu de justesse ou faux : **ncu du tampon décide** | **faux de 10-25 %** sauf si la tuile FP8 dépasse 85 % du pic (Marlin n'y est pas) |
| registres (jumeau) | squelette int8 ≈ 112 regs (fiche), à confirmer par ptxas | jumeau `mma2_kernel<128,4,64>` compilé sm_120f : **118 regs, 0 déversement, 24 Kio de shared par 4 étages** (`cuobjdump`) ; en e4m3 les fragments A doublent par k (2 MMA k32 par k64) → **≈ 135-150 regs**, 1 bloc/SM au lieu de 2 sauf BT = 64 ; B reste en quartets en shared (2 Kio par étage de 32 k) |
| ce qui la gênerait | trafic du tampon > 1,2 × (le L2 n'isole pas 8 experts sous 2 047 jetons de A en vol) → 37-44 ms, faux | ΔPPL > +0,004 (§ 1) ; ou l'émission : si les 4,5 instr/MMA ne s'amortissent pas (BT ≤ 32 sur les experts peu servis) la tuile tombe sous 60 % du pic → > 34 ms |
| ce qu'elle achète si elle tient | rien de structurel : Marlin reste au décodage, tampon par couche | **la disposition unique devient naturelle** (mma2 au décodage : ×0,88/×0,75 mesurés, mxf8f6f4 au prefill), C17 meurt sans une ligne, le repack Marlin et son cache disparaissent |

## 3. Échelles de A (route ii-b) : forme et coût
`A_scale[m][k/32]` UE8M0 = 2^⌈log₂(amax₃₂(ligne m)/448)⌉, un octet par 32 k, rangé `scale_vec::1X` par fragment (le même rangement que `sfa` de `mma_mxf4nvf4`, blocs de 32 au lieu de 16) ; quantification : amax par bloc de 32 sur la ligne (réduction warp), `cvt.rn.satfinite.e4m3x2` — même noyau que `nvfp4_quant_act` avec `blk = 32`, sortie 8 bits au lieu de 4 : **2 × les octets écrits de l'A4** (2,2 Go par prefill), ≈ 2-3 ms si séparé. **L'A8 e4m3 mesurée (porte MLA) n'est pas ce format** (échelle E4M3/16 contre UE8M0/32) : la porte des experts se rejoue au format exact de l'instruction — même script de fausse-quant que (ii-b) côté B, un seul passage carte pour les deux.

## 4. Jumeau torch et ptxas
* Jumeau torch de (ii-b), au bit du groupement : `donnees-c1-routes-19-09/erreur-b-mxf8f6f4.py` (`route_b` = B, `ue8m0_32` + `q_e4m3` = A avec plafond 448) ; la GEMM groupée du jumeau = par expert `A_e4m3 · B_e4m3ᵀ` en fp32 × S_A × S_B × g — non écrite (aucune ligne avant la route tranchée) ; référence de (i) : `acvram/kernels/w4a8_experts.py` (fiche C1).
* ptxas : (ii-b) n'a pas de tuile ; le chiffre engageant est celui du jumeau `<128,4,64>` (118/0/24 Kio) ; la tuile FP8 devra montrer ≤ 168 regs (1 bloc de 256 fils/SM), 0 déversement, avant toute carte.

## 5. Ce que je propose (poste7 tranche)
La fausse-quant **avant** la tuile, pour les deux routes dans un seul passage carte (15 min) : W8 int8/ligne (fait : +0,0007), **W8 e4m3/UE8M0-32 (ii-b)** et **A8 e4m3/UE8M0-32** — si (ii-b) sort > +0,004, la route est (i) et C17 reprend ; si ≤ +0,004, (ii-b) vaut sa tuile malgré les 30-34 ms prédits, parce qu'elle achète la disposition unique naturelle (le décodage y gagne déjà 12-25 % mesurés) et que 27,2 est un scellé sur les experts seuls, pas sur le pas.
