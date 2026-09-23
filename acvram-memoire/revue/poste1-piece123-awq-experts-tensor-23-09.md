# Pièce 123 — dossier : les −17,5 % de S1b à b=12 ne viennent pas de la division AWQ, mais du REFUS du chemin tensor — à sec, aucun code — 23/09 (poste1)

Référence : la cellule 118 de poste2. S1b donne 1 646 t/s à b=12, contre ≈ 1 995 pour i8c ; la seule différence est
dans les experts (AWQ calibré, alpha commun gate/up).

## 1. Où l'échelle AWQ d'entrée des experts est appliquée, et ce qu'elle coûte
* **Décodage à b < 8 (GEMV sur disposition Marlin)** : l'échelle est **fondue dans le noyau**.
  * `moe.py:1343-1344` (gate/up) et `:1410` (down) passent la table à `nvfp4_gemv_marlin(…, xscale)`.
  * Le noyau la lit en bf16, lignes contiguës (`acvram_kernels.cu:2486-2492`).
  * Il n'y a ni lancement ni lecture de plus au-delà de la table [E, K]. **À b=1, rien à gagner.**
* **Préfill groupé** : division torch avant la GEMM (`moe.py:846-850`, `:1119-1128`).
* **Décodage à b ≥ 8 (chemin tensor, Marlin MoE du port)** : **REFUSÉ**.
  * `MoEBlock._raison_tensor` (`moe.py:423-437`) passe `awq_unite = (gate, up, down sans table)` à `forme_tensor_refus` (`moe.py:1899-1913`).
  * Là, `if not awq_unite: return "tables AWQ d activation par expert non unité…"` (`:1910-1911`) renvoie la couche au GEMV à tous les godets.
  * **Preuve dans la 118** (`scratchpad/poste2-piece118-attribution-23-09`) : les 12 lignes de régime des alias à experts AWQ portent `chemin_moe=…+tensor(b≥8,repli:tables AWQ d activation par expert non unité…)`. i8c porte `tensor(b≥8)` sans repli.
* **Les −17,5 % sont donc la perte du chemin tensor**, qui valait +12,2 % de t/s et −30 % de J/jeton à b=12 (pièce 62 A5), et non un coût de la division. Le chemin tensor ne sait pas appliquer une échelle **par (jeton, expert)**.
  * Il appelle `gemm_moe(xm, …, top_k, T, …)` (`moe.py:1985`, `:1991-1992`) sur les **T lignes de jetons**.
  * Le Marlin MoE lit la ligne `paire / top_k` : la même ligne x pour les top_k experts d'un jeton. Or chacun devrait la diviser par une échelle différente.

## 2. Comment la fondre sans coût
**Au rassemblement, dans l'aligneur (recommandé).**
* `ext.moe_aligner_petit` (`acvram_kernels.cu:4692-4760`) parcourt déjà les G = T·top_k paires en UN lancement, au début du chemin tensor (`moe.py:1962`).
  * Il écrirait en plus `xs[g] = bf16(x[g / top_k] / s_gate[eid[g]])` : G × K bf16, 0,39 Mo par couche à b=12.
  * La GEMM gate·up (w13 : alpha commun gate/up dans A et S1b, `up_distinct` faux) serait alors appelée sur `xs` avec **top_k = 1** et M = G, comme l'appel down le fait déjà (`moe.py:1988`, `:2000` : `…, 1, G, …`). Les paires deviennent des « jetons ».
  * Coût : zéro lancement de plus ; ≈ 0,4 Mo d'écriture et de relecture L2 par couche, soit ≈ 0,3-0,5 µs à la bande L2.
  * Les fantômes (eid < 0) donnent une ligne nulle, comme le clamp actuel.
* **down** : `moe_act` accepte DÉJÀ la table AWQ de down et `e_sorted` (`acvram_kernels.cu:4763-4768`, appel `moe.py:861-862`). Il suffit de la passer dans le chemin tensor, au lieu de `awq_d=None` (`moe.py:917` pour le W4A4).
* **Tables gate/up distinctes** (pièce 47, pas de w13) : l'aligneur écrit deux `xs`, pour deux GEMM ; ce n'est pas le cas de A ni de S1b.
* Hors table unité, la garde `forme_tensor_refus(:1910)` devient « accepté si l'aligneur porte l'échelle ». La rotation Hadamard du down reste refusée.

Alternative écartée : l'échelle dans le **prologue du Marlin MoE**, en multipliant la tuile A par 1/s[e] en mémoire partagée. Elle modifie le template porté (NOTICE, cp.async → ldmatrix) sans rien gagner sur l'aligneur, qui touche déjà chaque paire.

## 3. Justesse
* **« Au bit contre le chemin actuel » est impossible par construction.** Le chemin servi de S1b à b ≥ 8 est aujourd'hui le GEMV. Tensor contre GEMV n'est jamais au bit : autre GEMM, autre découpe de K (2⁻⁷, pièces 82/82 ter), et le GEMV divise en fp32 dans le noyau alors que `xs` est arrondi en bf16.
* Critère proposé, celui de la mise au défaut du tensor (82 ter) :
  * **(a) `xs` de l'aligneur au bit de `bf16(x / s[e])` torch.** Division IEEE `__fdiv_rn` obligatoire, sous `--use_fast_math` (même leçon que la 104). Test sur fantômes, T = 8 / 12 / 16.
  * **(b) Chemin tensor avec AWQ au bit d'un témoin tensor recevant `xs` calculé en torch** : même GEMM, même entrée.
  * **(c) KL Coder S1b b=12 ≤ témoin (GEMV actuel) + 0,025, et b=1 identique** : le chemin tensor ne sert pas à b=1.
  * **(d)** Capture aux godets 8 et 16, sans repli (ligne de régime : `tensor(b≥8)` sans « repli »).

## 4. Gain prédit (avant mesure) et réfutation
* **b=12 : 1 646 → 1 850-1 990 t/s (+12 à +21 %)**, soit le retour au chemin tensor. Le prix est l'écriture `xs` (≈ 0,02-0,05 ms/pas) contre le +12,2 % de la 62. J/jeton : −20 à −30 %.
* **b=1 : inchangé** (GEMV, AWQ déjà fondu).
* **Réfuté** si S1b à b=12 gagne moins de 8 % en cellule servie, ou si la ligne de régime montre encore un repli.
* **Ce qui me gênerait** : que le GEMV actuel porte une partie de l'écart autrement, par exemple une lecture de la table AWQ non coalescée dans `nvfp4_gemv_marlin`. Dans ce cas, le gain du tensor serait moindre que +12 %. La cellule le dira.

## 5. Coût
* Aligneur + garde + down dans le chemin tensor : **2-3 h**, avec les tests (a) et (b) à sec et sur carte.
* Prise : KL + capture ≈ 10 min, puis la cellule servie de poste2 (S1b contre i8c).
