# Dossier — pièce 147 : effacer le surcoût constant de PROJ_MARLIN au préfill (à sec, aucun code, poste6, 24/09)

Entrée : 145 — +47-51 ms par requête sur gemma4 31B, +38-41 ms sur Qwen3.8-27B, quelle que soit la longueur (512 → 4 096) ;
+13-15 J par requête ; les deux bras préfillent au plafond de 400 W.

## 0. Ce que les deux bras font vraiment par requête (fichier:ligne)
* **A (disposition naturelle)** : `nvfp4_matmul` (`kernels/__init__.py:569`) → à n > 32 et `prefill_regime() == "bf16"` :
  `w = nvfp4_dequant(t, bf16)` (`:691`, noyau CUDA `ext.nvfp4_dequant`, `:472-491`) puis `F.linear` cuBLAS (`:692`). **A
  matérialise donc aussi tout le modèle en bf16 à chaque requête** : gemma 28,7 G paramètres denses (60 couches × 479 M :
  q/k/v/o 132 M + MLP 347 M) → 57 Go écrits + 14 Go lus = **72 Go, ≥ 48 ms à 1,5 To/s** ; Qwen3.8 21,8 G → 54 Go, ≥ 36 ms.
  (Le mode `PREFILL=w4a16`, `gemm_groupe.nvfp4_linear`, `:687-690`, évite la matérialisation mais n'est pas le défaut.)
* **B (disposition unique Marlin)** : `_marlin_seul` (`:1043-1052`) → `MP.depaqueter_marlin` (`marlin_port/__init__.py:267`),
  noyau Triton `_depaqueter_kernel` (`:538-571`, un programme par tuile 16 k × 64 n) puis le même `F.linear`. Même trafic que A
  (72 Go) mais **écrit par segments de 32 octets** (16 k bf16 par ligne n, 64 lignes par programme, `base = out + n·K + kt·16`) :
  ≈ la moitié de la bande d'un `nvfp4_dequant` qui écrit des lignes entières → ≈ 95 ms contre ≈ 48. **Le +47 ms de la 145 est
  la différence d'efficacité de deux dépaquetages qui font le même travail, pas un travail en plus.** Les vues q/k/v de la
  pile ajoutent des `.contiguous()` (`:593`) — petits.
* Conséquence pour l'énergie : 72 Go de trafic HBM ≈ 20 J par requête dans les DEUX bras (≈ 35 pJ/bit, 99) — un quart des
  84 J d'un préfill de 512 sur gemma. Le +13 J de B est le temps en plus au plafond (47 ms × 395 W ≈ 18 J, moins le repos).

## 1. Les trois leviers demandés, plus le quatrième que le § 0 impose
| levier | où | gain prédit (B, ms/requête) | sortie | VRAM | verdict |
|---|---|---|---|---|---|
| **L1 cache dépaqueté** (garder W bf16, ou une copie naturelle nvfp4) | `_marlin_seul` : ne pas refaire `depaqueter_marlin` | −95 (B < A de 48) | au bit | **+57 Go** (bf16) ou **+14 Go** (nvfp4 naturel = les deux dispositions) sur gemma ; +11 Go sur Qwen3.8. Libre au chargement (145, ctx 4 608) : gemma A 5,56 Gio / B 4,26, Qwen3.8 11,2 / 9,0 | **MORT** : n'entre pas, et la 146 (capacité KV, requêtes jamais closes quand le KV manque) rend chaque Gio de KV plus précieux |
| **L2 GEMM de préfill sur la disposition Marlin** (`MP.gemm_dense`, `marlin_port/__init__.py:447` → `ops.marlin_gemm`, réduction fp32 déterministe) | rendre le chemin d'avant la 134 (`_marlin_seul` `:1053-1058`, `CHEMINS_NVFP4["marlin_dense_seul"]`) | **−95 pour B et le même −48 pour A** si les deux y passent (aucune matérialisation : le noyau lit les codes fp4 dans la tuile) ; −20 J/requête | **PAS au bit**, par construction : Marlin accumule Σ code·x sur la tuile puis × échelle de bloc (fl(s·g) appliquée après la somme partielle), le défaut arrondit chaque poids bf16(code·fl(s·g)) AVANT le produit — l'ordre des arrondis diffère, pas seulement l'ordre des sommes. C'est l'excès de la 134 : **KL 0,00545 au pas 0** (seuil 0,00491), 0 aux pas suivants. Corrigible au bit : **non** (il faudrait reproduire l'arrondi bf16 du poids dans la tuile, ce que le noyau vLLM ne fait pas) ; corrigible en qualité : régime **« ± 1 ulp » opt-in** (REGLES § 1 : PPL avec erreur-type + KL, cellule étiquetée) — la KL est 11 % au-dessus du seuil de la 134, un scellé PPL/KL dédié tranche | 0 (moins qu'aujourd'hui : plus de W bf16 transitoire) | **le vrai levier**, pour les DEUX dispositions ; pièce à part, opt-in, jamais par défaut sans le scellé ± 1 ulp |
| **L3 dépaquetage par couche dans un tampon réutilisé, recouvert par la couche précédente** | boucle des couches `model.py:162-176` (`layer.prefetch()` `:172` existe déjà pour les poids en flux, `couches.py:458`) : flux CUDA annexe + événement, dépaqueter la couche i+1 dans un tampon double pendant la couche i | −45 (tout sauf la couche 0 : 95/60 = 1,6 ms de dépaquetage contre 4,4 ms de calcul par couche à 512 jetons, 15,8 ms à 2 048 — recouvert même à 512) | au bit (mêmes valeurs, seul le moment change) | tampon double d'une couche bf16 : gemma 2 × 0,96 Go = **+1,9 Go** (−1,5 Go net après le transitoire actuel de 0,46 Go) → KV −27 % sur gemma (5,56 → ~4 Gio) ; Qwen3.8 2 × 0,68 = +1,4 Go | possible mais **mauvais rapport** : le trafic (et les 20 J) restent, la VRAM se paie sur le KV de la 146, et la contention HBM à 512 jetons rogne le recouvrement |
| **L3' dépaquetage au débit de `nvfp4_dequant`** (noyau CUDA qui lit les tuiles Marlin par les tables `_indices()` `marlin_port/__init__.py:222-246` et ÉCRIT des lignes entières, comme `nvfp4_dequant`) | remplacer `_depaqueter_kernel` sur la carte (`noyau="triton"` → "cuda"), la version torch reste le juge | **−47 ± 5 → B = A** (Δ TTFT ≈ 0 à toute longueur ; J/requête ≈ A, le temps gagné au plafond rend ≈ 18 J) | au bit (mêmes fp32, même arrondi RNE, prouvé contre `depaqueter_marlin(noyau="torch")` sur les 3 formes + piles, comme la 134) | 0 | **recommandé en premier** : ferme la question « PROJ_MARLIN coûte-t-il au préfill » sans toucher au régime du défaut |

## 2. Recommandation chiffrée
1. **L3'** (≈ 1 jour, code noyau) : B rejoint A à ± 5 ms sur les six cases de la 145 ; c'est ce qui manque à la décision « PROJ_MARLIN
   par défaut » — après quoi le préfill n'est plus un argument contre, et les gains de décodage (b=8 +57-60 %, b=1 −1 à −7 % de J)
   restent seuls en balance. Preuve : rejouer `prise.sh gemma 1` et `qwen38 1` (deux prises de 7 min) ; seuil scellé d'avance :
   |Δ TTFT| ≤ 10 ms à 2 048 et 4 096, ≤ 15 ms à 512, 5/5 lots.
2. **L2 en pièce à part, ± 1 ulp, pour les deux dispositions** : −48 ms et −20 J par requête même sur le défaut actuel (A) — le
   seul levier qui touche l'énergie du préfill ; il exige le scellé PPL (SE) + KL de REGLES § 1 et n'entre jamais par défaut.
3. **L3 : non**, sauf si L3' échoue (VRAM contre KV, énergie inchangée). **L1 : non** (n'entre pas en VRAM).
4. Ce qui me gênerait : que L3' plafonne à −30 ms parce que la lecture des tuiles Marlin (512 octets par tuile, permutés) coûte
   plus que la lecture linéaire de `nvfp4_dequant` — alors B resterait +15 ms au-dessus de A, et la décision « défaut » devrait
   l'assumer explicitement (à 2 048 jetons, +1,6 %).

## Scellé L3' (feu chef, 11 h 4x, AVANT compilation et mesure)
* Code : `depaqueter_marlin_kernel` (acvram_kernels.cu, avant PYBIND) — un bloc par (8 tuiles k, tuile n, expert), tuiles lues
  en shared (coalescé), décodage avec EXACTEMENT l'arithmétique de `_depaqueter_kernel` (fp32 : bitcast((sb<<20)+0x34800000),
  0 si sb = 0, (s_dec·g)·2⁻¹¹⁹, × E2M1 signé (−0,0 conservé), arrondi bf16 par le même calcul entier `b + 0x7FFF + ((b>>16)&1)`),
  écriture par lignes entières (8 o par fil, 256 o contigus par warp). `depaqueter_marlin(noyau=auto)` → cuda si l'extension
  l'a ; `ACVRAM_DEPAQUETAGE` (Variable) force cuda | triton | torch ; compteur `DEPAQUETAGES` exposé dans `/metrics`.
  **Rien ne bouge sous PROJ_MARLIN=0** : `_marlin_seul` seul appelle le dépaquetage (test `test_le_defaut_n_appelle_pas_le_depaquetage`).
* Test au bit : `tests/test_depaqueter_cuda_p147.py` — 11 formes (gemma q/kv/o/gate_up/down, Qwen3.8 idem, pile MoE E=8), vue à
  échelle par colonne ; cuda = triton = torch à `torch.equal` sur les motifs int16 ; bras cassant : un octet de code ou d'échelle
  changé → sortie différente (et le juge torch voit le même écart).
* Prédiction : dépaquetage ≈ 48 ms sur gemma (72 Go à ~1,5 To/s) contre ≈ 95 aujourd'hui → **TTFT B − A : |Δ| ≤ 10 ms à 2 048 et
  4 096, ≤ 15 ms à 512, 5/5 lots** (seuils de chef), sur `prise.sh gemma 1` et `qwen38 1` (A B B A A), mêmes instrument et régime
  que la 145 (les lots A sont rejoués aussi : même séance). FAUX si Δ > 15 ms à 2 048/4 096 ou > 25 ms à 512 → le noyau reste
  sous la bande (lecture des tuiles permutées) et la décision « défaut » l'assume. Ce qui me gênerait : Δ < −10 ms (B plus rapide
  que A) — alors `nvfp4_dequant` lui-même n'est pas à la bande, et c'est le défaut qui a un levier.
