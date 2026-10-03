# Sonde : nommer le produit cuBLAS bf16 hors des six sites qui fait dépendre le moteur du découpage — scellé AVANT la prise (poste6, 03/10 03 h 4x, ordre chef)

Ordre : chef 03/10 03 h 3x — « scelle-la d'abord (prédiction écrite : quel produit, quel noyau, à quelle forme), puis UNE
prise carte.sh ≈ 1 min — sdpa_kernel(FLASH) forcé + profileur au préfill et au décodage ». Arbre : poste6-gemma-anneau
après e98bd8114 + retrait de `tranches`.

## Ce qu'on sait (mesuré)

* Drapeau global : B = A1 au bit (C1, 02/10) et A1 change (C3, 02/10). `etroite` (six `F.linear` des poids quantifiés au
  préfill, exacts au bit — B3) : A1 = A1 `reduite` au bit (C5, 03/10) et B ≠ A1 dès le premier jeton (C1, 03/10).
* Donc le produit cherché : (i) est lu par le drapeau global → un GEMM / GEMV cuBLAS bf16 ; (ii) n'est pas l'un des six
  sites ; (iii) agit DANS LE PRÉFILL (le premier jeton de B diffère de celui de A1) ; (iv) dépend du nombre de lignes M.
* À sec (`poste6-bf16-prefill-cublas-scelle-03-10.md`) : le SDPA du préfill est flash par la source (B1 : admissible,
  True pour les deux formes), `_BIAIS_MORCEAUX` = 1 par défaut (`attention.py:68`), le seul tenant passe par
  `is_causal=True` (`layers.py:1174`). Les couches int8 de Devstral (k 11, q 9, o 5, gate 5, down 13 sur 40) passent au
  préfill par `prefill_int8=cublas` = W8A8 entier (accumulation int32, exacte et indépendante de M), sauf repli.

## Instrument (une prise, deux processus sous garde de chaîne, ≈ 1 min)

`scratchpad/poste6-bf16/sonde-cublas.py` : charge Devstral, puis — (1) préfill de l'invite S1 (7 865 jetons,
`max_tokens=1`) sous `torch.profiler` avec formes : toutes les opérations `aten::{linear, mm, addmm, bmm, baddbmm, matmul,
_scaled_dot_product_*}` sur des entrées bf16 CUDA, comptées par (op, forme) ; en regard, le nombre d'appels de
`kernels.linear_prefill` (compteur) ; (2) le même préfill sous `sdpa_kernel([FLASH_ATTENTION])` forcé : réussite ou
`RuntimeError` ; (3) 8 pas de décodage sous profileur, même relevé ; (4) top-10 logprobs du premier jeton, `etroite`
contre `exacte` (drapeau global), dans le même processus. Processus A : seul tenant (`ACVRAM_PREFILL_MORCEAU_AU_DELA=0`) ;
processus B : morceaux (défaut, 4 096). Aucun texte lu : formes, comptes, logprobs.

## Prédictions

| # | grandeur | prédit | faux si |
|---|---|---|---|
| S1 | SDPA au préfill, A et B | `_scaled_dot_product_flash_attention` : 40 (A), 80 (B) ; aucun `bmm` / `_scaled_dot_product_attention_math` ; flash forcé : réussit | un `bmm` [têtes, M, 128] × [têtes, 128, T] ou l'erreur « No available kernel » : **l'attention est le produit cherché** (math : scores M = lignes, sortie N = 128, K = T) |
| S2 | GEMM bf16 cuBLAS au préfill hors `linear_prefill` (comptes `aten::mm / addmm / linear` bf16 − compteur `linear_prefill`) | **≥ 1 par couche NVFP4 ou int8**, forme lue dans le relevé — c'est le produit cherché. Candidat nommé : un repli bf16 des couches int8 (`kernels/__init__.py:1781`, `:1786`, déquantification → `F.linear`), formes [M, 5 120] × [5 120, 1 024] (k / v int8, 11 couches) ou [M, 32 768] × [32 768, 5 120] (down int8, 13 couches) | 0 : aucun GEMM bf16 cuBLAS hors des six sites — alors S1 doit être faux, sinon la cause n'est pas un GEMM et je le dirai |
| S3 | premier jeton, top-10, `etroite` contre `exacte`, processus A (préfill seul) | **différents** (le produit cherché agit au préfill) | égaux : il n'agit qu'au décodage, contradiction avec C1 (03/10) |
| S4 | décodage (8 pas) : `batched_decode_attention` à masque explicite | `bmm` bf16 cuBLAS [têtes, 1, 128] × [têtes, 128, T] présent (math) — c'est un second produit lu par le drapeau, N = 128, K = T, qui explique « mêmes ids, logprobs différents » (C3 tranches) mais pas le premier jeton | absent : le décodage est hors cuBLAS |

Décision, fixée ici : le produit nommé est celui de S1 (si l'attention) ou de S2 (forme lue) ; le correctif n'est PAS codé
dans cette pièce — il est proposé au chef avec sa forme : flash forcé / biais pour l'attention, ou `linear_prefill` étendu
au repli int8. Issues qui me gêneraient : S1 et S2 tous deux « rien » (le drapeau agirait par un chemin que je n'ai pas
listé — `torch.mv` / `addmv` du GEMV bf16, `cumsum`…) ; S3 égaux.
