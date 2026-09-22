# Écart TRT-LLM (2) : réconciliation avec la table nsys de poste2 — leur MoE est plus lent que le nôtre, leur avance est dans le bloc d attention (22/09, poste1, à sec)

* sources : `verdict-nsys-trtllm-serve-b12-22-09` (poste2 bb999ec0 : pas mur 6,513 = noyaux 6,364 + trou 0,149, fenêtre 128 jetons/séquence sous nsys ; MoE 5,011 ; « étroites/attention » 1,036 ; reste 0,238), `verdict-nsys-familles-22-09` (nous : 6,749), cellules poste3 (trtllm 1 998 = 6,01 ms/pas à 1 024 jetons ; acvram 1 638 = 7,33), poste2 : vLLM réel 1 782 (6,73 ms/pas).
* correction de lecture : dans la table trtllm, `routeur_gemm 0,498` est ma famille regex `cutlass::Kernel2` — chez trtllm ce sont les **projections d attention FP8** (qkv et o), pas un routeur (le leur est `customMoeRouting 0,096`) ; et `cutlass 40×1 0,426` (un lancement par pas, 40 tuiles) est **la tête** [12 × 151 936] bf16 (622 Mo à 1,79 To/s = 0,35 ms ; la nôtre int8 : 0,132). Une fois ces deux lignes replacées, les blocs se comparent :

## 1. Poste à poste (ms/pas, noyaux seuls)
| bloc | acvram (6,75) | trtllm (6,36) | écart | lecture |
|---|---|---|---|---|
| MoE : GEMM experts | Marlin 3,937 | grouped 2,880 + cuda_core_gemm 0,539 = **3,419** | +0,52 | 5,42 Go/pas dans les deux cas : 1,38 To/s chez nous (94 % de 1,55), **1,58 chez eux** — ils ne dépassent pas notre plancher de 1,55 que de 2 %, le reste est notre 6 % sous le plancher |
| MoE : routage + glue | route 0,44 + moe_act 0,05 = **0,49** | expand 0,305 + strides 0,254 + quantize 0,206 + activation 0,183 + routing 0,096 + build 0,080 + interleave 0,042 = **1,166** | **−0,68** | notre `_route_fusee` + pack en 2-3 lancements contre leurs 7 : leur W4A4 paie sa quantification d activation et ses tables à chaque pas |
| **MoE total** | **4,43** | **4,585** | **−0,16** | **leur MoE est plus lent** — la question « lisent-ils moins d octets » est close : non |
| tête | 0,132 (int8) | 0,426 (bf16) | −0,29 | la nôtre est meilleure |
| projections q/kv/o | **1,335** (int8 W8A16, 0,68 To/s) | **0,498** (FP8 W8A8 CUTLASS, 18,8 Mo/couche en 10,4 µs = **1,81 To/s**) | **+0,84** | **LE poste** : mêmes octets, ×2,7 sur le temps |
| attention | 0,459 (ctx 256+) | 0,374 (ctx 128) | ≈ 0 | non comparable (longueurs), à 128 jetons les deux sont à la latence |
| rope + kv_write | 0,234 | 0,164 (QKNormRope 0,097 + updateKV 0,067) | +0,07 | fusion 3a réfutée chez nous |
| normes | 0,241 | 0,181 | +0,06 | |
| glue + copies + trou | 0,05 + trou 0,03 (frontiere-pas) | 0,056 + trou 0,149 | −0,13 | leur trou sous nsys est plus grand que le nôtre |
| **service hors noyaux** (cellule 1 024 jetons) | 7,33 − 6,75 = **0,58** (cellule d avant le défaut epingle) | 6,01 − 6,36 < 0 : le pas de la cellule est PLUS COURT que celui de nsys (128 jetons + profileur) → **≈ 0** hors noyaux | **+0,5** | notre levier 2 en défaut ramène ceci à ≈ 0,2 (queue de pas lents) : à lire dans la cellule officielle 0.6.35 |
Somme des écarts : +0,84 + 0,07 + 0,06 + 0,5 − 0,16 − 0,29 − 0,13 ≈ **+0,9 ms ≈ l écart des cellules (1,32) moins l incomparable (attention, longueurs)**. **Réconcilié : les 22 % sont le bloc d attention (projections ×2,7) et notre service d avant le levier 2 ; le MoE et la tête jouent pour nous.** Ma note (1) plaçait 0,3-0,6 ms sur Marlin : faux, c est ≤ 0,45 et ce n est pas là qu ils gagnent.

## 2. Leviers réordonnés (objectif vLLM 1 782 = 6,73 ms/pas = −0,60 ms sur 7,33 ; puis trtllm 1 998 = −1,32)
| # | levier | plafond | au bit ? | mesure qui tranche |
|---|---|---|---|---|
| 0 | service : levier 2 en défaut (2bf61522) | 0,3-0,4 | oui (tenu) | cellule officielle 0.6.35 : prédit **1 700-1 720** (−0,35 ms) |
| **1** | **projections int8 étroites** : 1,335 → 0,60-0,70 (≥ 1,3 To/s, trtllm prouve 1,8 sur ces formes exactes) | **0,6-0,75 = 9-11 %** | **NON** (split-K : ordre fp32 des tranches ; ± 1 ulp par noyau + PPL à 2 SE) — opt-in tant que la règle tient | `--detail` fait (qkv 10,3 µs, o 11,8) ; banc à sec du noyau par (BN, tranches) ≤ 1 min carte ; puis ABBA |
| 2 | Marlin : 94 % → 100 % du plancher 1,55 | 0,25-0,45 | oui si tuiles et ordre intra-tuile inchangés (occupation, prefetch), non si les tuiles changent | `banc-horloge-decodage` gate·up à D = 42 contre 1,55 et 1,79 To/s |
| 3 | rope + kv_write en un (3a) | 0,07 | oui (mêmes ops) — réfuté a3f1b7e (corruption sous graphe), à rouvrir avec le défaut nommé, pas avant | test carte de capture |
| 4 | normes | 0,06 | oui | après 1-2 seulement |
**Arithmétique** : 0 + 2 (au bit) = −0,6 à −0,8 ms → 6,55-6,75 ms = **1 780-1 830 t/s = vLLM 1 782, à la limite, sans toucher un bit**. Avec le levier 1 (± 1 ulp) : −1,2 à −1,5 → 5,8-6,1 ms = **1 970-2 070 = trtllm**. Sans le levier 1, trtllm est hors de portée : c est un choix qui appartient à la chef et à l utilisateur (une règle « au bit » ou un opt-in documenté à ± 1 ulp).

## 3. Ce que la table trtllm apprend d autre
* Leurs 7 noyaux auxiliaires (1,17 ms) sont inhérents au chemin CUTLASS grouped-GEMM W4A4 (expand des lignes par expert, strides, quantification FP4 des activations, tables de block-scale) — c est exactement la glue que notre `mma-a4` porte aussi (nvfp4_quant_act, route_pack, reduce_trie) et qui a fait perdre C17 : **le chemin GEMV Marlin W4A16 à b=12 est le bon**, la table le confirme de l extérieur.
* Leur tête bf16 (0,43) coûte 3 × la nôtre : un point pour la tête int8 (`poste7-duel-verdict § 14`).
* Leur trou 0,149 sous nsys > le nôtre 0,03 : leur ordonnanceur n est pas meilleur, il est masqué par une cellule de 1 024 jetons où les pas sont longs.
