# Banc de prefill MoE : GEMM groupée vs déquant+grouped_mm

Date : 12/09/2026 — poste4, chantier 2.
Modèle : Qwen3-Coder-30B-A3B-nvfp4, RTX 5090, 7 répétitions, médian.

## Résultats

| L    | GEMM grp (j/s) | ±σ  | grouped_mm (j/s) | ±σ  | ratio |
|-----:|----------------:|----:|------------------:|----:|------:|
|   64 |           2 332 |  18 |             2 340 |  27 |  1,00 |
|  128 |           4 655 |  59 |             4 621 |  51 |  1,01 |
|  256 |           9 369 |  89 |             9 286 |  78 |  1,01 |
|  512 |          18 682 | 222 |            18 796 | 198 |  0,99 |
| 1024 |          37 153 | 475 |            37 301 | 321 |  1,00 |
| 2048 |          72 977 | 663 |            73 105 | 356 |  1,00 |

Référence bout en bout : 6 420 j/s (même modèle, CLI acvram, 512 j).

## Dénominateurs

- **j/s** = longueur / médian des 7 durées d'un `Engine.step()` complet
  (forward du modèle entier : embeddings, attention MLA, MoE, tête).
- **σ** = écart-type sur les 7 mesures converties en j/s.
- 2 passes de chauffe non mesurées avant chaque série.
- `torch.cuda.synchronize()` entre chaque passe.
- Le ratio 18 682 / 6 420 = 2,91 au L=512 vient de ce que le 6 420 de
  référence inclut la tokenisation, le transfert, et la latence du serveur ;
  ce banc ne mesure que le forward GPU.

## Conclusion

Le noyau NVFP4 de GEMM groupée (`nvfp4_gemm_grouped`) n'apporte **aucun gain
mesurable** au prefill par rapport au repli déquant+`torch._grouped_mm`, sur
aucune des longueurs testées (ratio ~1,00 partout, σ < 2 %).

**Explication** : au prefill, le temps du GEMM MoE est noyé dans le reste du
forward (attention MLA paginée, projections denses, embeddings). Le noyau NVFP4
économise de la bande passante mémoire (4 bits vs 16 bits pour les poids), mais
au prefill les lots sont assez grands pour que le GEMM soit compute-bound, pas
memory-bound — la déquantification bf16 suivie de `grouped_mm` (qui utilise les
tensor cores bf16 nativement) atteint le même débit effectif.

Le noyau NVFP4 reste utile au **décodage** (lots petits, memory-bound), où il
évite de lire 4× plus d'octets. Le banc de décodage MoE est un chantier
distinct.
