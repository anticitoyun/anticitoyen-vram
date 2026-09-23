# Pièce 68 — routage b = 1 : ce qui existe, et ce qui peut rester AU BIT (conception à sec, prédictions avant la carte) — 23/09 (poste1)

## Ce qui existe déjà (cherché avant d'écrire une ligne)

* `acvram/kernels/route_prep.py` — **`_route_fusee_kernel`** (Triton) : le noyau
  servi. `route_fusee()` le lance sur la grille `(T,)` : **à b = 1, T = 1, donc
  UN bloc** — et `route_logits_fusee()` le lance à `num_warps=1` (C15-3c), soit
  **un seul warp sur 170 SM**. Les 6,3 µs de la pièce 66 sont ce warp.
* `_route_prep_kernel` (même fichier) : déjà multi-blocs (`BLOC=256` sur `T·k`),
  mais il ne fait que la mise en forme après sélection — il ne sélectionne rien.
* `moe_route_pack_kernel` (`acvram_kernels.cu:4716`) : empaquetage après
  routage, pas la sélection.
* pièce 3 du 20/09 : le routeur GLM « en un nœud » (`_route_x_kernel`, 14,8 µs
  par lancement contre 6,0 pour les trois noyaux remplacés) — **réfuté**, et la
  cause était la sélection top-k sérielle. La même cause revient ici.

## Le fait qui commande tout : à T = 1, il n'y a pas de parallélisme de jetons

La grille est `(T,)`. À b = 1 un « routage multi-blocs » ne peut découper que
les **E = 128 experts** ou les **K = 8 passes**. Or :

* les K passes sont **séquentielles par nature** (chaque passe retire l'élu de
  `sel` avant la suivante) ;
* découper E sur plusieurs blocs change l'ordre de `tl.sum(ex, 0)` du softmax
  (`scoring="softmax"` pour ce modèle, `moe.py:70`) : les probabilités bougent
  au dernier bit, `topw` avec elles, et des experts basculent sur les égalités.
  **Ce n'est pas au bit, et le 19/09 l'a déjà coûté** (verdict-c15-niveau3-coder,
  versions 0ca85673 et 3e62a9b5 retirées pour exactement cette raison).

Ce qui, en revanche, **est exact quel que soit le découpage** : `probs` en
sigmoïde (élément par élément) ; `bi` (comparaisons d'entiers, départage par
indice le plus bas = un ordre total) ; `pj = tl.sum(tl.where(i == bi, probs, 0))`
— 127 zéros et une valeur, et `0.0 + x` est exact en fp32. Seuls le `tl.sum` du
softmax et l'accumulation `somme += pj` (ordre j = 0..K−1) portent l'exactitude.

**Donc le levier au bit n'est pas « plus de blocs » : c'est la sélection.**
Remplacer les 8 passes × 3 réductions de warp par une seule passe qui rend le
même top-k dans le même ordre laisse `tl.sum` du softmax intact, et retire 24
réductions sérialisées.

## Ce que je mesure AVANT d'écrire ce noyau (et pourquoi)

6,3 µs pour 128 flottants, c'est de la latence, pas du calcul : je refuse de
réécrire avant de savoir laquelle. Micro-banc (≤ 3 min de carte, prise courte) :

| bras | ce qu'il isole |
|---|---|
| `vide` | un noyau Triton qui ne fait rien, même grille, même `num_warps` : le **plancher de lancement** |
| `probs` | logits → probs, sans sélection : le coût du softmax seul |
| `servi` | `_route_fusee_kernel` tel quel (num_warps=1) : la référence, 6,3 µs attendus |
| `warps` | le même à num_warps=4 et 8 : **témoin de coût**, PAS un candidat (il change l'ordre de `tl.sum`, donc pas au bit — mesuré pour borner le gain possible) |

## Prédiction chiffrée et issues, écrites AVANT la mesure

* **prédiction nominale** : `vide` 1,5-2,5 µs, `probs` 2,5-3,5 µs, `servi`
  6,0-6,6 µs → **la sélection coûte 3,0-4,0 µs**, et c'est le seul gisement.
  Une sélection en une passe viserait 0,5-1,0 µs, soit **−2,5 à −3,5 µs par
  couche = −120 à −170 µs/pas** — au-dessus de la fourchette de chef
  (−80 à −110), parce qu'il compte 48 couches × 2,3 µs ; je prédis mieux et je
  m'y tiens.
* **R1 — réfutation par le plancher** : si `vide` ≥ 4,5 µs, le temps est le
  **lancement**, pas le noyau : aucune réécriture ne rend les 114 µs, et la
  pièce se ferme sur « fusionner le routage dans un noyau voisin, ou rien ».
* **R2 — réfutation par le softmax** : si `probs` ≥ 5,0 µs, c'est la réduction
  du softmax qui coûte, et elle est **intouchable au bit** : la pièce se ferme
  sur « pas au bit, donc pas ».
* **R3 — réfutation par les warps** : si `warps` (4 ou 8) ne gagne pas plus de
  1 µs sur `servi`, alors même en s'autorisant de casser l'exactitude il n'y a
  rien à prendre, et une version au bit en prendra encore moins.
* **A1 — alarme** : si `servi` mesuré s'écarte de plus de 20 % des 6,3 µs de la
  pièce 66, je ne mesure pas le même objet (godet, capture de graphe, horloge)
  et je le dis avant toute conclusion.
* **ce qui me gênerait et que je nomme quand même** : R1 et R2 ferment la pièce
  sans un seul noyau écrit, après que j'ai annoncé mieux que la fourchette du
  chef. Un échec est un résultat.

## Contrat d'exactitude du noyau, s'il est écrit

Même experts, mêmes poids, même départage : `sha256` de `(topw, topi, eid,
usage)` identique au chemin servi sur les mêmes logits, sur au moins 200 tirages
dont des égalités exactes construites exprès. Test cassant, et la faute
réintroduite une fois (départage vers l'indice le plus **haut**) doit le faire
échouer — sinon ce n'est pas un test d'équivalence.
