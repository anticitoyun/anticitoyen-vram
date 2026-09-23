# Pièce 59 — où sont les 8 % (1 634 contre 1 782 à b=12) : sur les noyaux vLLM mène de 1,47 ms/pas, dont **1,37 sur le GEMM des experts** (2,55 contre 3,92 ms pour les mêmes 96 lancements) ; sur l'hôte vLLM en rend 0,9 ; ma prédiction « ce sont les projections » est réfutée — 23/09 (Gaelle, à sec + 141 s de carte)

* instrument : trace nsys de vLLM (`scratchpad/gaelle-p59-23-09/chaine.sh` : `decode-vllm-17-09.py`, Coder `Qwen3-Coder-30B-A3B-Instruct-FP4-a16` (modelopt W4A16_NVFP4, 16,85 Gio, même classe de format que notre alias 17,05 Gio à 4,5 bpw), b = 12, ctx 2 304, invite 256, 1 024 jetons par séquence, fenêtre 5 s, mêmes réglages que la cellule b=12 vraie), `familles-vllm.py` (classement par nom, fenêtre = dernier train continu de pas repéré par `moe_align_block_size` (1 par couche), 806 pas) ; acvram officiel = trace du service de la pièce 39 (`nsys-tete-sampler-22-09`, 105 pas) reclassée par `familles-noyaux.py` (`familles-acvram-officiel.json`) ; cellules : acvram 1 634 (7,34 ms/pas), vLLM 1 782 (6,73)
* commit : a7885517 ; vLLM sous nsys : 1 670 t/s (le profileur lui coûte 6 %, comme à nous)
* régime : une prise de 141 s (`tenue=141s`), eco 2700, compute-apps début = fin (llama-server seul)
* scellé (gaelle.md a7885517, avant) : famille dominante prédite = projections qkv/o (+ 0,5 à + 0,8), MoE ≈ 0 ± 0,3 ; réfuté si la dominante est le MoE (vLLM ≥ 0,5 ms plus rapide sur les experts) ou le hors-noyaux
* mesuré (ms/pas, noyaux, b = 12) :

| famille | acvram (officiel) | vLLM | acvram − vLLM |
|---|---|---|---|
| **GEMM experts** (nous : `nvfp4_gemv_marlin` gate·up + down ; eux : `marlin_moe_wna16::Marlin`), 96 lancements/pas des deux côtés | **3,921** | **2,551** (26,6 µs par lancement) | **+ 1,37** |
| projections qkv/o (nous : int8 W8A16 `_etroit`, 145 l. ; eux : `marlin::Marlin` 4 bits, 96 l.) | 1,345 | 0,771 (8,0 µs) | + 0,57 |
| attention (nous `_partiel_reduit` ; eux `kernel_unified_attention` + `reduce_segments`) | 0,453 | 0,537 | − 0,08 |
| routeur GEMM + tête (eux : `cutlass` 49 l. + splitK) | 0,131 + 0,132 | 0,546 | − 0,28 |
| routage + glue MoE (nous `_route_fusee` + `moe_reduce` ; eux topkGating, align, moe_sum, act_and_mul, count_sort) | 0,310 + 0,052 | 0,386 | − 0,02 |
| normes + rope + kv (eux : fusions Triton `triton_red_fused_*` + `reshape_and_cache`) | 0,243 + 0,233 | 0,309 + 0,060 | + 0,11 |
| glue torch + copies + échantillonnage | 0,048 | 0,038 + 0,046 | ≈ 0 |
| **Σ noyaux** | **6,736** | **5,264** | **+ 1,47** |
| hors noyaux (cellule mur − Σ noyaux) | 7,34 − 6,74 = **0,60** | 6,73 − 5,26 = **1,47** | **− 0,87** |
| **mur (cellule)** | **7,34** | **6,73** | **+ 0,61 = les 8 %** |

* verdict : **prédiction RÉFUTÉE** — la famille dominante est le **GEMM des experts : + 1,37 ms/pas**, les projections viennent en second (+ 0,57) ; nous rendons 0,28 sur la tête et le routeur (leur tête bf16 cutlass), et vLLM perd 0,87 ms/pas SUR L'HÔTE (son pas réel est 1,47 ms plus long que ses noyaux : ordonnanceur Python, pas un noyau) — c'est ce qui ramène son avance de 1,47 à 0,61 ms. **Le fait dérangeant** : pour les mêmes 96 lancements sur le même modèle à 4,5 bpw, leur Marlin MoE fait 26,6 µs quand le nôtre en fait 40,8 (gate·up 53 + down 29 par couche) ; à 5,42 Go/pas (chiffre de la pièce 41, qui nous plaçait à 94 % du plancher 1,55) 2,55 ms voudrait dire 2,1 To/s, au-dessus de la HBM : **donc soit vLLM lit moins d'octets par pas que nous (experts distincts par couche, ou tuiles inutiles lues chez nous : nos grilles `[12×128]`/`[32×128]` lancent un programme par expert, 128 par couche), soit le chiffre 5,42 est faux et notre Marlin n'est pas à 94 % du plancher mais à ≈ 75 %.** Dans les deux cas, la pièce 41 (« leur MoE est plus lent, ce n'est pas là qu'ils gagnent ») valait pour TRT-LLM W4A4, pas pour vLLM W4A16 Marlin.
* durée : 141 s de carte (trace vLLM, autorisée : aucune n'existait), 40 min à sec ; carte LIBRE

## Pièce qui vise la famille dominante (proposée, non jouée)
**Octets réels du MoE par pas, des deux côtés, avant tout noyau** : (1) acvram, 1 min de carte : relever `_usage_routage` (experts distincts par couche et par pas) sur la même fenêtre b=12 et recalculer Go/pas et To/s effectifs de `nvfp4_gemv_marlin` ; (2) vLLM, à sec : ses experts distincts sont les mêmes (même routeur, mêmes jetons) — donc si (1) confirme ≈ 42 experts/couche (5,4 Go), leur 2,55 ms est inexplicable par les octets et il faut regarder leur Marlin MoE (tuiles, préchargement, format des échelles) ; si (1) donne ≈ 30-36, notre kernel lit ce qu'il faut mais à 1,1-1,2 To/s et le levier est dans le noyau (les 128 programmes par couche dont ~90 vides, la réduction). **Prédiction** : (1) rend 30-36 experts/couche (routage concentré sur des jetons tirés au hasard), Marlin acvram à 70-78 % du plancher → gain accessible − 0,8 à − 1,1 ms/pas (1 634 → 1 830-1 900, devant vLLM) ; **réfuté** si (1) rend ≥ 42 (nous sommes à 94 %, l'écart est dans leur kernel et il faut le lire). Second levier, indépendant : projections en 4 bits (− 0,5 ms, la 42 corrigée par la 47).
