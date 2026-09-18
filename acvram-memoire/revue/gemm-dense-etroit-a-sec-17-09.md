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

## Défaut, tête et témoin (troisième commit) — poste7-gemm-dense-palier2-non-ouvert-17-09

- **Défaut `ACVRAM_DENSE_NVFP4=triton`, bascule `ACVRAM_DENSE_NVFP4_MIN_M=4`**
  (poste3 0690bd4 : b=12 128 → 349 t/s, J/j ÷ 2,7, b=1 et ppl-decode-kv
  inchangés ; à M = 2 la GEMV gagne au banc, 1,50 contre 1,12 To/s).
- **Tête NVFP4 couverte** (`model.py _tete`) : x bf16, M ≥ 4, sans biais ni
  flux → `gemm_dense_etroit(…, sortie_fp32=True)` (scaler appliqué avant
  s'il y en a un) ; les logits sont accumulés en fp32 depuis des produits
  bf16 × bf16 exacts — la GEMV fp32 relisait 0,6 Go par séquence (18 % du
  pas). Témoin : `ACVRAM_TETE_FP32_ENTREE=1` ou `DENSE_NVFP4=gemv`.
  Test : logits fp32 contre x fp32 @ déquant fp32 (le chemin GEMV) — 2⁻⁷ ×
  borne, écart max 10⁻⁵ relatif ET même argmax sur chaque ligne.
- **Palier 2 = témoin nommé** : `ACVRAM_MULTI_PROJ=1` (défaut 0) pour la
  multi-projection ; jamais empruntée par défaut (0,88 pondéré, qkv_multi
  0,52 ≈ kv seule 0,47 : la sous-occupation n'explique pas, cause inconnue).
- Scellé de poste7 pour la tête (poste3, après Nemotron) : b=12 349 → ≥ 400
  t/s (faux < 380 : relire le profil), ppl-decode-kv ± 0,0005, b=1 ± 3 %.
  Ma prédiction : 5,6 ms de tête à 0,23 To/s → ~1,2 ms à 1,1 To/s
  (N = 248 320 : grille large, le régime « down » 1,09) ⇒ pas 34,4 → ~30 ms,
  **≈ 400 t/s** — juste au scellé ; faux si < 380.

## Dernier tour : réduction fusionnée (épilogue « dernier bloc ») — poste7-lm-head-392-verdict-17-09

`_dense_etroit_kernel` : les T tranches K d'une tuile N écrivent leur partiel
fp32 puis incrémentent un compteur (`tl.atomic_add`, acq_rel) ; la dernière
arrivée somme les T partiels **dans l'ordre t = 0..T-1** (lectures
`volatile`, même somme que `_reduire_kernel` : déterministe, bit à bit
rejouable), écrit la sortie (bf16 ou fp32 pour la tête) et remet le compteur
à zéro — un lancement au lieu de deux, aucun memset, rejouable sous graphe.
T = 1 : écriture directe, pas de tampon. Compteurs : un tampon int32 par
appareil, jamais réalloué en dessous, les anciens gardés vivants (un graphe
capturé les tient). Test : plusieurs tranches → exact, compteurs revenus à
zéro, second appel identique au bit. Suite 816 passed. La multi-projection
(témoin) garde `_reduire_kernel`.

Scellé de poste7 (poste3) : ≥ 455 t/s NU tenu / < 435 faux, J/jeton ≤ 0,95× ;
tenu ⇒ défaut, faux ⇒ on laisse. Ma prédiction : la réduction séparée était
12 % du pas (≈ 3,4 ms sur 28,9 nu à 415 t/s) ; l'épilogue en rend ~2,5 ms
(la somme reste, mais dans le noyau et sans second lancement ni relecture
du workspace par un autre noyau) ⇒ ~26,4 ms ⇒ **≈ 450-460 t/s nu** — au
bord du scellé ; faux si < 435 (alors la somme par le dernier bloc sérialise
la fin des tuiles : les T-1 autres attendent… non, elles sortent ; c'est le
volatile qui coûterait).
