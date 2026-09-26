# Verdict — pièce 203 : profil du chemin experts servi, Coder-30B-A3B, b=1 et b=8 — le GEMM tensor est AU PIC HBM à b=8, le GEMV par paire à 62 % à b=1 ; les trous entre noyaux pèsent plus que les auxiliaires (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p203-25-09/prise-nsys.sh` (chariot 116 : `frontiere-pas.py` 20 pas sous nsys
  `--cuda-graph-trace=node`, graphes, alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c), `classer-203.py` (fenêtres de couche
  [route_fusee(i), route_fusee(i+1)) du lot mesuré seulement — grille du marqueur [B×1×1] —, médiane par noyau nom + grille),
  experts distincts : `ACVRAM_TRACE_ROUTAGE_PT` (appels eager de la même prise, 88 appels à 64 paires). Résultats
  `classes-b{1,8}.{txt,json}`, `b{1,8}/frontiere.json` ; traces hors git : b1/graphe.nsys-rep 49fd64e8d69da5a8…, b8/ 8a9ecc53aa250b43….
* **commit** : 3b0aa2345 (poste6-203 = origin/main 9d646c197 + scellé + instruments).
* **régime** : carte 0, horloge libre, graphes on (repli_eager=0), llama-server 4242 (5,6 Go, tiers) présent début = fin ;
  b=1 `chemin_moe=mma-a4(atteint=gemv_marlin+gemv_v1)+tensor(b≥8, repli: 4 couches)`, b=8 `…(atteint=gemv_marlin+marlin_tensor+
  gemv_v1+decode_mma)+tensor(b≥8, repli: 4 couches « up_proj : échelles sous-normales non représentables en Marlin » (157))`.
* **scellé** : `scratchpad/poste6-p203-25-09/scelle.md` (3b0aa2345, avant les prises).
* **mesuré** : b=1 pas GPU 2 940 µs (noyaux 2 508, 52,3 µs/couche) ; b=8 pas GPU 4 996 µs (noyaux 4 473, 93,2 µs/couche) ;
  experts distincts par couche à 64 paires : moy 30,5, méd 29, p10 23, p90 40 (prédit 26-31 : tenu) ; 8 à b=1.
* **verdict** : (1) GEMM experts : b=1 GEMV par paire 19,2 µs/couche = **62 % du pic HBM** (gain max 0,26-0,36 ms/pas, 9-12 %,
  hors bit) ; b=8 tensor 45,9 µs pour ≈ 81 Mo = **1,76 To/s = 98 % du pic 1,79** → falsificateur (a) DÉCLENCHÉ, prédiction
  72-88 % FAUSSE, rien à gagner sur le noyau à b=8 ; (2) trous entre noyaux sous graphe : **0,43 ms/pas à b=1 (14,7 %), 0,52 à
  b=8 (10,5 %)**, au bit, gain max ≈ la moitié (PDL / fusions) ; (3) auxiliaires MoE à b=8 : 9,9 µs/couche = 0,47 ms/pas (9,5 %),
  fusion route + routeur + aligneur ≈ −0,2 ms (hors bit sur les logits fp32), act/reduce dans les épilogues ≈ −0,1 (hors bit).
* **durée** : prévu 2 × ≤ 5 min ; tenu 232 s (b=1, dont chargement) + 65 s (b=8) ; 4 min de dépouillement à sec par trace.

## 1. Ce qui est servi (à sec, fichier:ligne)
`moe.py:1306` : tensor si T ≥ 8 (`regime.py:114`) → b=1 = GEMV Marlin PAR PAIRE (`nvfp4_gemv_marlin_kernel` gate·up bf16 [12×8×4]
puis down fp32 [32×8×2], `moe_reduce`) ; b=8 = `gemm_experts_tensor` (`moe.py:1981`) : `moe_aligner_petit_kernel` (1 bloc),
`marlin_moe_wna16` w13 gate·up fusionné puis down (même grille [510×1×1]), `moe_act_kernel`, `moe_reduce_kernel` ; routage :
`_route_fusee_kernel` + GEMM du routeur (cutlass wmma bf16 [8×1×8]). 4 couches (up_proj sous-normal, 157) restent hors
tensor à b=8 : `decode_mma` (`nvfp4_gemm_grouped_mma2_kernel` 26,6 + 21,9 µs, `moe_route_pack` 7,1, `nvfp4_quant_act` 2,1,
`moe_reduce_trie` 1,1 = 58,7 µs/couche contre 50,5 pour le tensor : +8 µs × 4 = 0,03 ms/pas, nommé, négligeable).

## 2. Par couche (µs, médiane des fenêtres du lot plein ; × 48 = ms/pas)
| poste | b=1 µs | b=1 ms/pas | % noyaux | b=8 µs | b=8 ms/pas | % noyaux |
|---|---|---|---|---|---|---|
| GEMM experts | 19,23 (12,00 + 7,23) | 0,923 | 36,8 | **45,92** (2 × 23,14) | 2,204 | 49,3 |
| routage (route_fusee) | 2,62 | 0,126 | 5,0 | 2,69 | 0,129 | 2,9 |
| GEMM du routeur (cutlass) | — (dans route_fusee) | | | 2,56 | 0,123 | 2,7 |
| aligneur | — | | | 2,43 | 0,117 | 2,6 |
| act + reduce | 0,93 | 0,045 | 1,8 | 1,38 + 0,83 | 0,106 | 2,4 |
| **MoE total** | **22,8** | **1,09** | **43,6** | **55,8** | **2,68** | **59,9** |
| projections étroites int8 | 17,18 | 0,825 | 32,9 | 20,93 | 1,005 | 22,5 |
| attention / rope-kv / normes | 5,66 / 3,52 / 2,78 | 0,57 | 22,9 | 7,14 / 4,13 / 2,94 | 0,68 | 15,3 |
| noyaux | 52,26 | 2,508 | 100 | 93,18 | 4,473 | 100 |
| **pas GPU (frontière)** | 61,2 | **2,940** | | 104,1 | **4,996** | |
| **trous entre noyaux (pas − noyaux)** | 9,0 | **0,43** | 14,7 % du pas | 10,9 | **0,52** | 10,5 % du pas |
Octets contre l'idéal (2,65 Mo par expert : 3 × 786 Ko de codes + 3 × 98 Ko d'échelles) : b=1 8 experts = 21,2 Mo en 19,23 µs
= **1,10 To/s (62 % du pic 1,79, 71 % du plancher étroit 1,55)** ; b=8 30,5 distincts = 80,8 Mo en 45,92 µs = **1,76 To/s
(98 % du pic)** — réserve : le compte distinct vient des 88 appels eager à T=8 de la même prise (les pas sous graphe ne
tracent pas le routage) ; à 29 distincts (médiane) 1,67 To/s, 93 %. Le 66 µs de la 62-A4 à b=12 (33,4 distincts, 1,35 To/s)
est un autre lot : à b=8 le noyau porté de vLLM est au pic, ce que la 61/0 (60 µs pour 88 Mo, 1,47) laissait prévoir.
Auxiliaires : b=1 3,55 µs = 6,8 % (116 : 6,5 %, piste morte confirmée) ; b=8 9,9 µs = 10,6 % des noyaux, 9,5 % du pas.

## 3. Les trois postes (gain max, au bit ou non)
1. **GEMM experts à b=1 : GEMV par paire à 62 % du pic** — 0,92 ms/pas ; au pic 0,57, au plancher étroit 0,66 : **gain max
   0,26-0,36 ms/pas (9-12 % du pas b=1)**. Hors bit dès que l'ordre des sommes change (autre noyau) ; par expert distinct ne
   rend rien (8 paires = 8 distincts). À b=8 : rien (98 % du pic) — le levier b=8 n'est PAS le noyau.
2. **Trous entre noyaux sous graphe** : 0,43 ms/pas à b=1 (14,7 %), 0,52 à b=8 (10,5 %) — ≈ 0,8-0,9 µs par lancement
   (11 lancements/couche à b=1, 13 à b=8). **Au bit** (ordonnancement seul) : PDL (194 § a, borné 0,15-0,3 ms sur Qwen3.8,
   à banc-er ici), fusion de lancements (act + reduce + aligneur : −3 lancements/couche = −0,12 ms). Gain max ≈ 0,2-0,3 ms/pas.
3. **Auxiliaires MoE à b=8** : 0,47 ms/pas (9,5 %) — route_fusee 2,7 + routeur GEMM 2,6 + aligneur 2,4 (7,7 µs pour < 1 Mo :
   latence pure) en UN noyau : gain max ≈ −5 µs/couche = **−0,24 ms/pas (4,8 %)**, hors bit sur les logits du routeur (somme
   fp32 d'un GEMM maison ≠ cutlass ; top-k stable sauf égalité) ; act dans l'épilogue w13 et reduce dans l'épilogue down :
   −0,1 ms, hors bit (reduce). À b=1 : mort (6,8 %, 116).
Hors postes : projections étroites int8 22-33 % (195 les sert déjà en canal : ce profil est PRIS APRÈS 195 b ; `_etroit_reduit`
[80×4] et [32×11] à b=8 = les formes 5120… non : ici K 2048, hors table 195 → noyau à tranches : à mesurer, autre pièce).

## Suite (à chef)
(i) b=1 : un GEMV experts à 90 % du pic vaut 0,3 ms (10 %) — hors bit, scellé KL comme 195 ; (ii) b=8 : PDL + fusion
route/routeur/aligneur (0,2-0,5 ms, 4-10 %) ; (iii) le noyau tensor est au pic : ne plus y chercher ; (iv) les projections
int8 du Coder (K 2048) ne sont pas dans la table 195 : à mesurer (banc-canal, 5 formes) avant d'étendre la table.
