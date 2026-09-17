# GEMM W4A16 dense à petit M — noyau, juge et micro-banc livrés à sec (poste4, 17/09) ; la porte est le banc de poste3

Commande : poste7-hybrides-etape1-close-gemm-dense-17-09 § 2. Le poste : à
b > 1, `nvfp4_gemv` relit les poids une fois par séquence (Qwen3.8 b = 12 :
20 Go à 0,23 To/s, 86,6 ms sur 93,9 ; plancher de bande 11 ms).

## Noyau : `acvram/kernels/gemm_dense_etroit.py`

- Tuile d'activation [BM = 16 ou 32, BK] en registres (M rembourré), poids
  NVFP4 balayés UNE fois par pas et décodés en registres par les tables de
  B1' (E2M1 → bf16, E4M3 → bf16, produit code × échelle exact en bf16),
  `tl.dot` sur la tuile, échelle globale (scalaire ou par ligne) dans
  l'épilogue. Grille (tuiles N, tranches K) : `_tranches` vise ≥ 2 programmes
  par SM (q/k/v/o : N = 5 120 → 80 tuiles de 64, donc 4-5 tranches K) ;
  partiels fp32 réduits par `y.sum(0)` (déterministe, même ordre par tuile).
- Réglages exposés au banc (BN, BK, warps, stages) ; défauts 64/128/4/3,
  `ACVRAM_DENSE_ETROIT_*` hors régime.
- Intégration derrière `ACVRAM_DENSE_NVFP4 = gemv (défaut, témoin) | triton`
  (`kernels/__init__.py nvfp4_matmul`, 2 ≤ n ≤ 32, bf16) ; M = 1 garde la
  GEMV. Variable de régime (regime.py, cli.py).

## Juge (même commit, règle 9) : `tests/test_gemm_dense_etroit.py`

2⁻⁷ × Σ|x·w| contre la déquantification, M ∈ {2, 12, 16, 17, 32}, formes
N = K et N > K, entrée plus courte que `padded_in` + échelle par ligne,
plusieurs tranches K forcées ; **bras cassant de poste7** : échelle de bloc
décalée d'un rang → rouge. 13 tests, interpréteur Triton en fp16 sans carte.

## Porte : `outils/banc-gemm-dense-etroit-17-09.py` (poste3, ~5 min de carte)

Formes Qwen3.8-27B (q 6144×5120, kv 2048×5120, o 5120×6144, gdn_qkv
10240×5120, gdn_out, gate_up 34816×5120, down 5120×17408), M ∈ {2, 12, 32},
8 configurations Triton, témoins `gemv_boucle` (attendu ~0,23 To/s) et
`narrow_gemm` ; rejeu de graphe ; chaque bras jugé exact (hors 2⁻⁷ = 0) ;
JSON `scratchpad/banc-gemm-dense-etroit-17-09.json` avec le verdict :
minimum sur les formes de la meilleure config exacte à M = 12 — ≥ 1,3 To/s
OUVRE, < 0,9 FAUX (noyau CUDA, décision séparée), entre : à poste7.

Prédiction (avant le banc) : q/k/v/o et gdn_qkv 1,0-1,4 To/s (N petit :
la réduction des tranches K et le rembourrage M = 12 → 16 coûtent), gate_up
et down ≥ 1,3 (grilles larges) ; la boucle GEMV 0,2-0,3 ; narrow_gemm
0,5-0,8. Issue qui me gênerait : ≤ 0,9 partout parce que `tl.dot` à BM = 16
et le décodage par table laissent la bande inoccupée — alors BK plus grand
(256) et `num_stages` 4 avant de conclure au noyau CUDA.

## Aussi dans ce commit

`cli.VARIABLES_LUES` : `ACVRAM_PREFILL_A4` (21b6476 sur main l'avait mis dans
regime.VARIABLES mais pas dans la liste de cadrage : `test_cadrage_perplexite`
rouge en suite complète).

## Palier 2 (poste7-gemm-dense-porte-fermee-palier-17-09 § 3) — à sec, second commit

Banc de poste3 (21033b7, M = 12) : q 0,72 · kv 0,47 · o 0,86 · gdn_qkv 1,01 ·
gdn_out 0,83 · gate_up 0,94 · down 1,09 To/s — le taux suit N : la porte
palier 1 (min ≥ 1,3) fermée ; ma prédiction (q/k/v/o 1,0-1,4) réfutée sauf
sur down.

Fait vérifié avant d'écrire : **l'empilement q/k/v existe déjà**
(`Attention.fuse` → `stack_nvfp4_linears`) mais il est REFUSÉ sur calibA —
`_scaler_commun` : act_scale q ≠ k ≠ v (max |q − k| = 19,4 sur la couche 3),
qkv ≠ gate du GDN (0,53). Une conversion calibrée par projection ne s'empile
pas sans requantifier ; c'est un choix du convertisseur (poste2), pas du
moteur. D'où, sans toucher aux poids :

1. **`MultiProjection`** (`kernels/gemm_dense_etroit.py`) : plusieurs
   projections NVFP4 de même entrée en UN lancement, chacune avec SON scaler
   (x / s calculé en fp32 dans le noyau puis arrondi — le même arrondi que
   `ChannelScaler.apply` en torch), table de pointeurs (poids, échelles de
   bloc, scalers), tuiles par projection, échelle globale concaténée.
   Branchée : `Attention.fuse()` la pose quand l'empilement refuse
   (`qkv_multi`, servie par `_proj` à 2 ≤ t ≤ 32 sous
   `ACVRAM_DENSE_NVFP4=triton`) ; `GatedDeltaNet.fuse()` (nouveau, appelé
   par le chargeur) sert qkv + gate + α + β (N = 16 480) en un lancement.
2. **Tranches K** : `_PROGRAMMES_PAR_SM` balayé par le banc (2 / 4 / 8) sur
   toutes les formes ; **réduction fusionnée** (`_reduire_kernel` : somme des
   partiels + sortie bf16 en un lancement, contre deux en torch) — sur kv à
   13 µs, un lancement compte.
3. Banc : formes réelles du pas (qkv_multi 6144+1024+1024 avec scalers
   distincts, gdn_multi, o, gdn_out, gate_up, down, × couches 17/48/65),
   témoin `separees_triton` (palier 1) et gemv/narrow sur q et kv ; juge =
   **taux pondéré par les octets du pas** (meilleure config exacte par
   forme) : ≥ 1,0 OUVRE, < 0,85 FAUX. Le JSON porte `pondere_To_s` et
   `ms_gemm_pas` (la somme des GEMM du pas à M = 12).

Juge à sec : `test_gemm_dense_etroit.py` +4 — multi = projections séparées
(M 2/12/32, scalers distincts), sans scaler + échelle par ligne, bras
cassant : scalers échangés entre projections → rouge ; intégration GDN
(`fuse()` puis forward sous `triton` = forward sous `gemv`). Suite 811.

Prédiction scellée (banc poste3, M = 12) : qkv_multi 0,95-1,05 (le gdn_qkv
1,01 en est le témoin, même N), gdn_multi 1,0-1,1, o et gdn_out 0,90-1,0
avec 4-8 programmes/SM (pas ≥ 1,0 : la réduction fp32 grandit avec les
tranches), gate_up 0,95-1,05, down 1,05-1,15 ; **pondéré 0,95-1,05**. Faux
si < 0,85 : alors la sous-occupation n'est pas la seule cause (latence du
décodage par table dans la boucle K) et c'est le noyau CUDA (§ 4 de poste7).
En situ ensuite (poste3) : pas GEMM ≈ 16-19 ms ⇒ 500-600 t/s.
