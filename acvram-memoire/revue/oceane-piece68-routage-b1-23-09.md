# Pièce 68 — routage b = 1 : ce qui existe, et ce qui peut rester AU BIT (conception à sec, prédictions avant la carte) — 23/09 (Océane)

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
  couche = −120 à −170 µs/pas** — au-dessus de la fourchette de Jerome
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

## Mesure et correctif — 23/09, quatre prises courtes en alternance avec Gaelle

### D'abord une faute d'instrument, dite avant les chiffres

Le premier banc chronométrait **un** lancement par graphe : le bras `vide`
rendait 4,77 µs et le bras `probs` sortait **au-dessus** du noyau complet, ce
qui est impossible. `Event … g.replay() … Event` mesurait la latence de rejeu,
pas ce que le pas paie — dans le pas, le noyau est un nœud parmi d'autres d'un
graphe déjà lancé. Corrigé par 200 lancements **dans** le graphe, divisés :
le plancher tombe à 0,38 µs et tout redevient cohérent. Les chiffres ci-dessous
sont ceux du banc corrigé.

### Décomposition (T = 1, E = 128, k = 8, 100 rejeux × 200 lancements)

| bras | µs | lecture |
|---|---|---|
| `vide` (nœud de graphe) | **0,38** | le plancher : **R1 RÉFUTÉE** |
| `probs` (softmax + somme) | 0,75 | +0,37 : **R2 RÉFUTÉE**, le softmax ne coûte rien |
| `servi` (num_warps=1) | **4,54** | la sélection pèse **3,79 µs, 83 % du noyau** |
| `warps4` / `warps8` | 5,40 / 5,56 | **R3 TENUE** : plus de warps est PIRE |

Puis deux témoins qui isolent la cause **dans** la sélection :

| témoin | µs | écart |
|---|---|---|
| référence (atomiques et stores en boucle) | 4,56 | — |
| **sans les k `atomic_add` de la boucle** | **2,00** | **−2,56 µs** |
| stores groupés hors boucle | 4,55 | −0,02 (rien) |

**La cause est nommée : les k `tl.atomic_add` étaient dans la boucle de
sélection, donc sérialisés derrière elle — 2,56 µs sur 4,56, 56 % du noyau.**
Les stores, eux, ne coûtent rien et restent en place.

### Correctif, au bit par construction

Un seul atomique vectorisé après la boucle (`route_prep.py:105`) : les indices
élus sont gardés en registres (`ti`), et `tl.atomic_add(usage_ptr + ti, 1,
mask=mj & (v != 0))` les compte d'un coup. **L'addition entière est commutative
et associative, et les k experts d'un jeton sont distincts par construction**
(chaque passe retire l'élu de `sel`) : aucune collision ne se joue sur l'ordre.
`probs`, la sélection, `topi`, `topw` et `eid` sont inchangés, ligne pour ligne.

| | avant | après | écart |
|---|---|---|---|
| noyau servi (banc) | 4,54 µs | **2,32 µs** | **−2,22 µs, −49 %** |
| sur 48 couches | — | — | **−107 µs/pas** |

**A1 s'est déclenchée et elle avait raison** : le banc rend 4,54 µs là où la
trace nsys du service en donne 6,3 (−28 %) — pas de contention, cache chaud,
200 lancements du même noyau. La **décomposition** reste valide (les quatre
bras partagent ce biais), mais **l'absolu ne l'est pas** : si le rapport se
conserve, le gain dans le service est **≈ −148 µs/pas** ; seule une trace nsys
b = 1 après correctif le dira. Je publie les deux et je ne choisis pas le plus
flatteur. La fourchette de Jerome (−80 à −110) est tenue au banc ; la mienne
(−120 à −170) ne l'est qu'à l'échelle nsys, et je le dis.

### Le test d'équivalence ne mordait pas — trou trouvé et bouché

La faute annoncée au contrat (départage vers l'indice le plus **haut**) ne
faisait tomber **qu'un seul** test, et pas celui de l'équivalence : les égalités
fabriquées de `test_f2_route_fusee_egale_moe_route` sont **hors du top-k**,
donc elles ne jugeaient rien depuis le 17/09. Deux tests ajoutés où les ex-aequo
**sont** les plus grands (six experts à égalité parfaite, k = 4 ; puis cinq
ex-aequo par jeton sur cinq jetons, avec biais et sigmoïde).

Vérification que les tests peuvent rendre « faux », deux fautes posées une à une :

* **masque `jj < K` oublié** dans l'atomique groupé — la faute la plus
  silencieuse (elle ne change ni `topi` ni `topw`, seulement le compte
  d'usage) : **16 échecs sur 39** ;
* **départage vers l'indice le plus haut** : **3 échecs** (1 seul avant que je
  bouche le trou).

39 verts sur l'arbre propre, 0,96 s, sous le verrou de la carte.

## Reste

Une trace nsys b = 1 du service après correctif, pour le chiffre publiable du
pas (la chaîne existe : `scratchpad/gaelle-p66-23-09/chaine.sh`, `familles-b1.py`).
Elle ne m'appartient pas : Gaelle ou Manon la jouent dans une fenêtre qu'elles
tiennent déjà, et le chiffre du banc (−107 µs) sert de prédiction à réfuter.
