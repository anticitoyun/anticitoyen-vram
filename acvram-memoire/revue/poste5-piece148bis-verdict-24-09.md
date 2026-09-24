# Verdict — pièce 148 bis : le pas b=8 de l'alias mixte décomposé, face à NInfer (poste5, 24/09)

* **instrument** : `outils/gpu/mesure/frontiere-pas.py` (en processus, régime servi, 300 pas, SANS profileur) ; nsys
  `-t cuda --cuda-graph-trace=node` (60 pas) → `scratchpad/poste5-p148bis-24-09/familles-148.py` ; NInfer : nsys sur
  `ninfer-serve` sous le banc chat b=8, 12 s → même grille (`familles-ninfer.txt`)
* **commit** : e3d70897 (acvram, `acvram.__file__` = worktree poste5), 46b43c0a (NInfer)
* **régime** : alias `Qwen3.8-27B-unsloth-mixte-i8c`, PROJ_MARLIN (config 142), b=8, invite 256, ctx 2048, -lgc 2700,
  cpu-safe 100, `régime NOMINAL`, graphes 6 captures / 339 rejeux ; NInfer `qwen3_8_27b_nvfp4.ninfer` (W4A4/W8A8)
* **scellé** : `revue/poste5-piece148bis-scelle-24-09.md` (écrit avant)
* **mesuré** : acvram en processus **21,50 ms/pas** (graphe 21,47, échantillon 0,004, trou hôte 0,023 en médiane,
  moyenne 0,43), somme des noyaux sous nsys 20,87 (écart 3 % : instrument tenu) ; NInfer somme des noyaux **15,98 ms/pas**
  (671 pas), banc 17,27
* **verdict** : prédictions GEMV **TENUES** (int8 8,13 dans 7,5-9 ; nvfp4 5,74 dans 5-7). Issue (c) **TENUE** : 5,5 ms entre
  banc et processus, plus que les 2-4 prédits. Frontière hôte sous la fourchette (0,02 contre 0,3-1). GDN sous la
  fourchette (1,20 contre 1,5-4). **NON PRÉDIT : 2,85 ms/pas de GEMM bf16 alpha/beta.**
* **durée** : prévu ≤ 10 + 5 min / tenu 11:17:52-11:19:25 et 11:22:11-11:23:27 (`carte.sh`)

## Grille commune (b=8, durées de noyaux, ms/pas)

| famille | acvram | NInfer | Δ |
|---|---|---|---|
| couches 8 bits (int8 W8A16 / fp8 W8A8-A16) + tête | 8,13 | 7,59 | +0,54 |
| MLP nvfp4 (W4A16 Marlin / W4A4 + quantification des activations) | 5,74 | 6,53 | **−0,79** |
| **GEMM bf16 alpha/beta des 48 couches GDN** | **2,85** | 0 (fusionnées dans `gdn_projected_conv`) | **+2,85** |
| glue torch (1 331 lancements/pas chez nous) | 1,45 | ~0,06 | +1,39 |
| GDN (récurrence, conv, gating) | 1,20 | 1,24 | −0,04 |
| normes / copies / attention / rope | 1,41 | 0,53 | +0,88 |
| **somme des noyaux** | **20,87** | **15,98** | **+4,89** |
| service et hôte (banc − noyaux) | 6,1 | 1,3 | **+4,8** |

## Où sont les 9,7 ms d'écart (27,0 contre 17,3)

1. **Service et hôte : ≈ +4,8 ms.** NInfer passe de ses noyaux à son banc HTTP pour 1,3 ms ; nous pour 6,1, alors que la
   frontière entre pas ne coûte que 0,02 ms en processus. Le surcoût est dans le SERVICE (HTTP/SSE, ordonnanceur,
   admission des lots successifs du banc chat, détokenisation, préfill par lot), pas dans le moteur. Non décomposé ici.
2. **GEMM bf16 alpha/beta : +2,85 ms.** Poids bf16 [48 × 5120] (0,5 Mo), deux par couche GDN. À M = 8, cuBLAS choisit
   `cutlass_80_wmma … 16x16_128x1`, grille de **4 blocs**, **30,8 µs par appel** (≈ 16 Go/s). Un GEMV étroit prendrait
   ~2-3 µs. Remède direct : empiler alpha et beta ([96 × 5120]) et les servir par le chemin étroit, ou les fusionner à
   in_proj_qkv/z comme NInfer. Gain attendu ≈ −2,6 ms/pas (**+12 % à b=8**), sans changer la sortie au-delà de l'ulp.
3. **Glue torch : +1,4 ms** (1 331 petits lancements/pas) ; normes et copies : +0,8. Chantier de fusions, à la C15.
4. **Poids : quasi égalité.** Nos 8 bits sont 0,5 ms plus lents, notre nvfp4 Marlin est 0,8 ms plus rapide que leur W4A4.
   Cela confirme la clôture du port C (148) : le noyau 8 bits n'est pas le levier.

## Proposition (ordre de gain)

(i) Pièce alpha/beta : pile + chemin étroit, au bit ou ± ulp, prédit −2,6 ms/pas. Elle vaut sans doute aussi pour
`Qwen3.8-27B-nvfp4` (mêmes couches GDN en clair), à vérifier. (ii) Décomposer les 4,8 ms de service (banc chat contre
`frontiere-pas` à charge égale). (iii) Fusions de la glue du pas hybride.
