# nsys sur trtllm-serve b=12 — pas TENU, MoE bien au-dessus de la fourchette prédite — 22/09 (Manon)

* instrument : `nsys profile -t cuda --cuda-graph-trace=node -o trtllm`, script dédié `scratchpad/trtllm-cellules-22-09/nsys-trtllm-b12.sh` (nsys DANS la prise carte.sh, leçon `verdict-nsys-familles-22-09`), `trtllm-serve` checkpoint hub, `--backend pytorch --max_batch_size 12` ; client `banc-llamacpp-16-09.py decode` (BANC_SLOTS=12, BANC_JETONS=128, BANC_FENETRE_S=5) — **faux départ nommé sans en faire un échec** : 1er essai avec `BANC_MOTEUR=trtllm` a pris le mauvais endpoint HTTP (`/completion` natif llama.cpp au lieu de `/v1/completions` OpenAI, car `MOTEUR` pilote aussi le choix d'endpoint dans le client, pas seulement le label) → 0 jeton décodé, archivé `nsys-sortie-echec-endpoint-07h50/`, non exploité ; 2e essai avec `BANC_MOTEUR=acvram` (label trompeur mais endpoint correct, comme le fait `cellule.sh` de Laure) → 9 216 jetons décodés, 1 781,5 t/s, TENU
* commit : main à jour (5d541f8a au lancement)
* méthode de segmentation : `outils/gpu/mesure/familles-noyaux.py` attend un marqueur 1×/couche (acvram : `_route_fusee_kernel`) — absent chez trtllm. Substitut : `FusedAddRMSNormKernel` (2×/couche, `--couches 96`), 1 706 pas jugés (têtes/queue exclues) — cohérent avec l'estimation indépendante (~1 672 passes avant/pendant/après la fenêtre mesurée)
* scellé (Océane, `oceane-ecart-trtllm-22-09.md` § 3) : pas 6,0 ± 0,3 ms ; MoE 3,1-3,5 ; étroites/attention 0,9-1,2 ; reste 0,4-0,6 ; trou ≤ 0,1 ; réfuté si MoE < 2,9 ou pas < 5,5
* mesuré :

| poste | ms/pas | prédit | verdict |
|---|---|---|---|
| **pas (mur)** | **6,513** (noyaux 6,364, trou 0,149) | 6,0 ± 0,3 | limite haute, non réfuté |
| **MoE** (GEMM groupée 2,880 + cuda_core_gemm_nvfp4 0,539 + cutlass 40×1 0,426 + expandInputRows 0,305 + computeStrides 0,254 + quantize_bloc 0,206 + doActivation 0,183 + customMoeRouting 0,096 + fusedBuildExpert 0,080 + block_scale_interleave 0,042) | **5,011** | 3,1-3,5 | **très au-dessus, hors fourchette (+43 % sur la borne haute), pas une alarme codée (seul MoE<2,9 l'est)** |
| étroites/attention (routeur_gemm 0,498 + kernel_mha 0,374 + fusedQKNormRope 0,097 + applyBiasRopeUpdateKV 0,067) | 1,036 | 0,9-1,2 | **dans la fourchette** |
| reste (normes 0,181 + glue_torch 0,028 + copies 0,028 + routage 0,001) | 0,238 | 0,4-0,6 | en dessous |
| trou | 0,149 | ≤ 0,1 | légèrement au-dessus |

* verdict : **pas TENU au sens littéral** (aucune des deux alarmes codées — MoE<2,9 ou pas<5,5 — ne se déclenche), **mais la décomposition contredit la prédiction sur sa répartition interne** : le poste MoE mesuré (5,011 ms) est 43-62 % au-dessus de la fourchette prédite (3,1-3,5), et le poste « reste » est sous sa fourchette (0,238 contre 0,4-0,6) — la somme reste proche du total prédit (compensation), mais la lecture par poste que le scellé voulait obtenir (« la GEMM MoE sort plus/moins de bande que notre plancher ») n'est pas ce qui s'est passé : **7 noyaux auxiliaires MoE non anticipés** (expand/strides/quantize/activation/routage/build/block-scale, 1,933 ms cumulés, 30 % du pas) portent l'essentiel de l'écart — trtllm paie une bureaucratie de routage MoE par couche que le modèle de plancher (bande GEMM seule) ne comptait pas. Ce n'est pas mesurable chez nous par `banc-horloge-decodage` (qui ne mesure que la GEMM) : comparaison directe GEMM-contre-GEMM biaisée en faveur d'acvram si on ignore cette bureaucratie côté trtllm.
* durée : ~10 min (2 essais, chargement trtllm ~80 s chacun, capture 5 s, post-traitement nsys ~2 min sur 1,94 M lignes)

## Suite
La question initiale (« MoE < 2,9 = bande > notre plancher ») ne se pose pas : MoE est plus LENT que prédit, pas plus rapide. À nommer par Océane : les 7 noyaux MoE auxiliaires sont-ils compressibles (fusion routage+expand, comme le fait déjà `_route_fusee_kernel` côté acvram) ou inhérents au chemin CUTLASS grouped-GEMM de trtllm ? Carte rendue à Laure (courbe débit(b)).
