# Verdict (à sec) — diff des deux chemins de prefill MoE : les experts routés sont les mêmes, l'expert partagé passe en W8A8 sous extension

poste4, 17/09, arbre `poste4`, suite de verdict-bissection-w4a16-suite-17-09 (poste3 6df34fb : tout-torch 1,0104 contre noyaux 1,0280, ≥ 1,2 % non localisé) et de poste7-bissection-w4a16-verdict-17-09 pt.3. Lecture seule + un instrument processeur ; aucune carte.

## Ce que fait chaque régime au prefill (t = 2048, GLM-4.7-Flash-vllm-direct, ACVRAM_MOE_MMA=0)

| étape | noyaux CUDA | tout-torch (`ACVRAM_DISABLE_KERNELS=1`) | différence |
|---|---|---|---|
| experts routés : déquant de la pile | `_pile_bf16` → `ext.nvfp4_dequant(gscale_rows)` (`model.py:863-883`, `kernels/__init__.py:457-465`) | même `_pile_bf16` → repli `dequantize_nvfp4(global_scale_rows)` (`kernels/__init__.py:444-456`, 9b5b428) | même arithmétique : s = e4m3(bloc)·g fp32, code·s, un arrondi bf16 (`acvram_kernels.cu:196-206`) — bras noyau de `outils/dequant-trois-references-17-09.py` chez poste3 |
| experts routés : GEMM | `torch._grouped_mm` bf16 (`model.py:1081-1084`) | **identique** (même branche : `mma` faux, `direct` faux car 128 j/expert > `_MOE_GEMM_MAX` 48) | aucune |
| experts routés : act | `ext.moe_act` fp32 silu, un arrondi bf16 (`acvram_kernels.cu:2842-2876`) | silu fp32 × u fp32 → bf16 (`model.py:1030-1032`) | ulp fp32 (fast-math), `tests/test_moe_glue.py` ; poste3 : −0,19 % en masquant act+swiglu |
| experts routés : réduction top-k | `ext.moe_reduce_trie` somme fp32 pondérée, un arrondi (`acvram_kernels.cu:2905-2918`) | `d.float() × topw` → `sum` fp32 → bf16 (`model.py:1091-1092`) | ordre fp32 ; masqué par poste3 : 1,0275 |
| **expert partagé (47 couches) et couche dense 0 : gate/up/down** | backend `cuda-fusionne` (`kernels/__init__.py:753`) → `nvfp4_matmul`, n = 2048 > 32 → `ACVRAM_PREFILL` défaut `a8` (`kernels/__init__.py:579-593`) → `fp4_gemm.nvfp4_mm_w4a8` (`fp4_gemm.py:257-281`) : **poids déquantifié requantifié E4M3 par ligne (`:277`), activations E4M3 par ligne (`:276`), `torch._scaled_mm`** | backend `reference-cuda` (`kernels/__init__.py:816`) → `dequantize_nvfp4` bf16 → `F.linear` exact | **la seule différence porteuse d'erreur** : double quantification FP4 bloc 16 → E4M3 ligne entière, plus activations FP8 |

Le régime « W4A16 » n'en est un que pour les experts routés. L'expert partagé sert chaque jeton de chaque couche ; sous extension il tourne en W8A8 avec un poids requantifié.

Ce qui ne diffère pas non plus : `ACVRAM_PREFILL` ne dépend pas de l'extension (`os.environ.get`, `kernels/__init__.py:583`) mais `nvfp4_matmul` n'est atteint que par le backend `cuda-fusionne`, dont `available = get_extension() is not None` (`:720`) — d'où l'écart entre les deux régimes de poste3. Les masques `PPL_MASQUE_NOYAUX` (proxy sur `_EXT`) ne pouvaient pas le voir : le chemin a8 n'appelle aucun symbole de l'extension, il est en torch (`_scaled_mm`) derrière une porte qui teste l'extension.

## Mesure à sec (processeur, poids réels, même arithmétique que `fp4_gemm.py`)

`outils/w4a8-expert-partage-17-09.py GLM-4.7-Flash-vllm-direct 0,1,10,30 512` — erreur relative RMS contre le produit fp32 exact (activations synthétiques, gaussienne et t₃ ; le bras « W8 poids » n'en dépend pas) :

| tenseur | W8 poids seul | A8 seul | W8A8 (= noyau) |
|---|---|---|---|
| couche 0 dense gate/up/down | 0,026 / 0,028 / 0,031 | 0,026 | 0,037 / 0,038 / 0,041 |
| partagé c.1 | 0,029 / 0,027 / 0,029 | 0,026 | 0,039 / 0,038 / 0,039 |
| partagé c.10 | 0,028 / 0,029 / 0,029 | 0,026 | 0,039 / 0,039 / 0,039 |
| partagé c.30 | 0,027 / 0,027 / 0,025 | 0,027 | 0,038 / 0,038 / 0,036 |

Soit ≈ 3,8 % d'erreur relative sur la sortie de chaque projection de l'expert partagé, à chaque couche, uniquement sous extension. Le repli bf16 du même poids : 0,14 % (`tests/test_prefill_w4a8_double_quantification.py`, processeur, 1 passé).

## Prédiction scellée (carte, poste3, 3 tranches privé, même instrument)

`ACVRAM_PREFILL=bf16` avec les noyaux, tout le reste par défaut (MOE_MMA=0) :

- **≤ 1,014 × bf16** (≈ tout-torch 1,0104 ± le bruit de routage 0,3 % vu sur les masques) → l'expert partagé en W8A8 est le 1,2-1,7 % ; l'expert routé n'y est pour rien.
- entre 1,014 et 1,024 → une part, pas tout ; reste à chercher dans rmsnorm (−0,35 %, poste3) et la glue.
- **≥ 1,024** → cette piste est réfutée ; l'écart est alors dans `nvfp4_dequant` noyau contre repli (bras noyau de dequant-trois-references) — seule différence restante des experts routés.

Coût attendu du bras : le chemin bf16 matérialise le poids et fait cuBLAS bf16 ; l'expert partagé est petit (1536 × 2048), le prefill ne devrait pas bouger de plus de quelques %.

## Témoins (commit poste4)

- `tests/test_prefill_w4a8_double_quantification.py` : (1) processeur — double quantification E4M3 par ligne 2,8 % contre bf16 0,14 % sur un NVFP4 synthétique à blocs étagés ; (2) carte — `nvfp4_matmul` à 64 lignes, bras `a8` > 1,5 % et bras `bf16` < 0,5 % contre fp64 ; (3) carte, **xfail strict** — le défaut vaut le bras bf16 : casse tant que le défaut reste `a8`, à retirer le jour où poste7 change le défaut.
- `outils/w4a8-expert-partage-17-09.py` : la table ci-dessus, rejouable sur tout converti.

## Hors périmètre, noté

- `ACVRAM_PREFILL=a8` est aussi le défaut au prefill de tous les modèles denses NVFP4 : toute PPL « W4A16 » publiée sous extension avec plus de 32 jetons par appel est en réalité W8A8 (poids double-quantifié) sur les projections non groupées. À relire dans les tables publiées.
- `nvfp4_mm_tensorcore` (`a4`) est le repli du a8 s'il rend None (`kernels/__init__.py:590-592`) ; sur sm_120 avec `_scaled_mm` disponible il n'est pas atteint.
