# zzs — cœur MLA du préfill : troncature causale des clés (poste5, 01/10) — borne, prédiction et critère d'équivalence SCELLÉS avant le code

Ordre de chef (bd zzs, suite de `poste5-ht9-verdict-01-10.md`). Base : dac452617 (origin/main), branche poste5-mla-causal.

## Ce qui change

`mla.py:571-581` : chaque morceau de 256 requêtes [d0, d1) calcule `einsum(q_eff[d0:d1], C32)` sur les `total` clés, puis
masque celles qui sont au-delà de `passe + d1` (`:578`), puis fait softmax et produit par V sur `total`. La pièce limite
les deux produits, le masque et le softmax à `C32[:passe + d1]`. La règle des 2 048 clés (`cles = passe + d1`), les
dtypes, la taille des morceaux et le chemin flash ne changent pas. `ACVRAM_MLA_CAUSAL=0` rend l'ancien chemin (témoin).

## Borne (Kimi-Linear-35B, 7 couches MLA, 32 têtes, rang 512 + rope 64)

Travail du cœur proportionnel au nombre de clés traitées, S, par morceau : 2 × 256 × 32 × (576 + 512) × S FLOP, et
256 × 32 × S éléments pour le masque, l'échelle et le softmax.

| t | morceaux fp32 (> 2 048 clés) : Σ S tronqué / complet | morceaux tf32 (≤ 2 048) | ensemble |
|---|---|---|---|
| 8 192 | 125 952 / 196 608 = **64,1 %** | 9 216 / 65 536 = 14,1 % | 51,6 % |
| 32 768 | 2 104 320 / 3 932 160 = **53,5 %** | 9 216 / 262 144 = 3,5 % | 50,4 % |

Mesuré au ht9 (p81:prefill8k, 1 415,9 ms GPU) : sgemm fp32 491,3 ms, tf32 90,4 ms, masque 76,0, échelle 75,9 et softmax
76,2 ms (224 lancements = 7 × 32 morceaux). Borne du gain à 8 k, à rendement de noyau égal :
−176,6 (fp32) − 77,7 (tf32) − 110,4 (élémentaire) = **−364 ms, soit −25,7 % du GPU du préfill**.
À 32 k (extrapolation : cœur quadratique, le reste linéaire), préfill ≈ 15,7 s dont cœur fp32 ≈ 9,8 s et élémentaire
≈ 3,6 s ; borne du gain ≈ −6,7 s, soit **−43 %**.

## Prédiction (scellée)

* 8 k, nsys p81 (`p81:prefill8k`, GPU) : **−280 à −365 ms, soit −20 à −26 %** (le bas de la fourchette paie la perte de
  rendement des GEMM courtes : les morceaux tf32 voient 256 à 2 048 clés).
* 8 k, mur du préfill (instrument ci-dessous) : **−18 à −26 %**.
* 32 k, mur du préfill : **−35 à −48 %**.
* Décodage b=1 et b=12 : inchangés (le chemin t = 1 n'est pas touché), 0 ± 1 %.
* **FAUX si** le gain du GPU à 8 k est inférieur à 150 ms (−10,6 %), ou si le gain au mur à 32 k est inférieur à 20 %.
* Issues nommées : (i) cuBLAS choisit, pour les S courts, des noyaux moins efficaces ou un découpage en K, ce qui ronge
  le gain des morceaux tf32 (le bas de la fourchette) ; (ii) le softmax PyTorch change d'algorithme ou de taille de bloc
  selon la longueur de ligne (non vérifié dans le source de la version installée), ce qui changerait l'ordre de
  somme pour les morceaux courts et coûterait le « au bit » ; (iii) la gênante : le mur à 32 k est dominé par autre
  chose que je n'ai pas mesuré (mémoire des scores de 1 Go par morceau, allocateur), et le gain au mur est loin de la
  borne GPU.

## Critère d'équivalence (scellé, dans cet ordre)

* **E1 — au bit, visé.** (a) Test sur carte : la sortie du préfill MLA (y, [t, nh·dv]) est `torch.equal` entre
  ACVRAM_MLA_CAUSAL=1 et =0, pour t = 8 192 sans passé et pour t = 1 000 avec passe = 3 000 (préfill par morceaux),
  au régime servi (tf32 ≤ 2 048 clés, fp32 au-delà). (b) Bout en bout : ids gloutons de 64 jetons après une invite de
  8 192 et une de 32 768, A contre B IDENTIQUES, avec les témoins A1 = A2 et B1 = B2.
* **E2 — si E1 tombe : tolérance dérivée de l'ordre de somme, sous témoin (comme lic, 2 × le témoin).** Le témoin T est
  l'écart de l'ANCIEN chemin à lui-même sous un autre ordre de somme légitime : morceaux de 128 au lieu de 256
  (`ACVRAM_MLA_MORCEAU`, diagnostic), ce qui change la longueur des GEMM et des softmax par morceau, comme la
  troncature. Exigence : max |y_B − y_A| ≤ 2 × max |y_A128 − y_A| par couche dans le test sur carte, et pour le bout en
  bout, |Δ logprob| des 64 jetons gloutons ≤ 2 × le témoin, les ids pouvant diverger seulement après le premier jeton
  où le témoin lui-même diverge.
* Le verdict dit lequel de E1 ou E2 tient. Si ni l'un ni l'autre : FAUX en équivalence, rien n'est servi.

## Tests cassants (même commit que le code)

* Sans carte (CPU, CUDA_VISIBLE_DEVICES="" et TRITON_INTERPRET) : la forme. Chaque morceau ne traite que `passe + d1`
  clés (einsum espionné). Le test casse si l'on revient à `total`, ou si un décalage de un fait perdre la diagonale.
  Écart au chemin complet ≤ 2 × celui du témoin à morceaux de 128, contre une référence fp64.
* Sur carte : E1(a) au bit, ou E2 selon le verdict, avec le bras cassant « troncature à `passe + d1 − 1` » qui doit
  rendre faux.

## Amendement du 01/10 (~15 h 30), AVANT toute mesure sur carte : le témoin de E2

Le témoin « morceaux de 128 » était dégénéré. Dans le chemin complet, chaque produit et chaque softmax porte sur
`total` clés quelle que soit la taille des morceaux. Changer 256 en 128 ne change donc aucune longueur de somme : sur
le processeur, A128 = A au bit (mesuré à sec, 300 + 700 jetons). Un témoin nul ne peut rien borner. Il est remplacé par
**l'erreur d'arrondi du chemin servi contre une référence fp64** (même module, même entrée, tout en fp64) :

* E2, test : max |y_B − y_A| ≤ 2 × max |y_A − y_ref| ET max |y_B − y_ref| ≤ 2 × max |y_A − y_ref|. À sec en fp32
  (forme réduite) et sur carte en fp32, aux formes de Kimi avec un passé (3 000 + 1 000), au régime servi.
* E2, bout en bout : inchangé pour les ids (64 jetons gloutons après 8 k et 32 k, témoins A1 = A2 et B1 = B2). Une
  divergence est publiée avec son premier jeton, et la décision revient au chef ; aucune tolérance n'est inventée après coup.
* `ACVRAM_MLA_MORCEAU` est retiré : il ne servait que ce témoin.
* Déjà constaté à sec : sur le processeur, B n'est PAS au bit de A (|B − A| = 1,0e-7 pour max |A| = 0,28), et
  |B − ref| = |A − ref| = 1,4e-7 : la troncature est aussi juste que le chemin complet. E1 se joue sur la carte, où
  cuBLAS et le softmax de PyTorch décident.
* Risque nommé (duck.ai via poste4, `poste4-duckai-01-10-c.md`) : vLLM (accumulate_mla_context_chunk) et FlashMLA
  tronquent déjà causalement par morceau et fusionnent par LSE, sans gain chiffré publié pour la troncature seule.
  L'issue vLLM #27491 signale des NaN en MLA avec préfill par morceaux. Ici, chaque ligne de scores garde au moins sa
  diagonale (clé passe + i ≤ passe + d1 − 1), donc aucun softmax tout −inf. Le test de forme et le bras cassant
  (diagonale perdue) couvrent ce bord ; la prise relèvera tout NaN.
* Tests liés (71 fichiers : MLA, régime, registre) sans carte, avec TRITON_INTERPRET posé par le conftest : 713 passés,
  139 sautés, 1 xfail (15:01:15-15:03:22). Chevauchement : la campagne a pris la carte à 15:01:20 (services deepseek-coder).

## Amendement 2 du 02/10 (13 h 25), après la prise de 13:11-13:18, AVANT toute mesure à 16 k

Ce que la prise a rendu, sans rien changer à la prédiction ci-dessus :

* **Étape 0** : E1(a) tombe sur carte (8 192 sans passé : 262 473 éléments ≠, max 3,9e-3 ; 3 000 + 1 000 : 17 357 ≠,
  max 2,0e-3), E2 vert : le verdict d'équivalence se juge en E2, comme prévu par le critère.
* **nsys p81, préfill 8 k au GPU** (le critère FAUX principal, régime du pilote à 8 448 de contexte, sans exil) :
  A 1 406,7 ms → B 1 064,0 ms, soit −342,7 ms, −24,4 % (prédit −280 à −365 ms, −20 à −26 % ; FAUX < 150 ms). Décodage
  b=1 −0,04 %, b=12 +0,04 % (prédit 0 ± 1 %).
* **ABBA en service : invalide, faute de l'instrument.** À `--max-model-len 32896`, le plan exile 5 MLP sur 27 en RAM
  hôte (graphes coupés, « 17 562 tiendrait sans exil »), et le KV de 1 024 blocs refuse l'invite de 32 768 (« 0 jetons,
  6 ms », sans erreur HTTP). Le comparateur a refusé (rc 5). Aucun mur n'est publié.

Changement, posé avant de mesurer : **32 768 ne tient pas en régime résident sur la carte.** La longueur longue devient
**16 384** (`--max-model-len 16512`). La borne se calcule comme plus haut : Σ S tronqué / complet = 50,8 % (fp32 : 57,0 %,
tf32 : 7,0 %). En extrapolant depuis le ht9 (cœur quadratique, le reste linéaire), le préfill fait ≈ 4,45 s, dont un gain
borné à ≈ −1,60 s, soit **−36 %**. Prédiction au mur à 16 k : **−25 à −36 %**, et **FAUX si le gain au mur à 16 k est
inférieur à 15 %**. Ce critère remplace celui de 32 k, qui reste écrit pour une carte où 32 k tiendrait. L'instrument
prouve désormais le régime résident avant toute requête (`/metrics` : graphes, repli_eager, kv_max_tokens ≥ L + 64), et il
refuse toute réponse privée de ses 64 jetons (rc 7).

**Ordre de rédaction (demandé par chef) : l'amendement 2 a été écrit APRÈS avoir lu le −24,4 % du nsys à 8 k.** Le haut
de la bande (−36 %) n'en dépend pas : c'est la borne à rendement égal, extrapolée depuis la décomposition du ht9 (1 415,9 ms,
mesurée la veille) par la même règle que les bornes de 8 k et 32 k. Le bas (−25 %) n'est PAS indépendant du chiffre lu. À 8 k,
le GPU a réalisé 342,7 ms sur une borne de 364, soit 94 %, et le mur garde des parts hors GPU que le nsys ne voit pas (pilote,
échantillonnage, HTTP). J'ai donc posé le bas à environ 70 % de la borne (0,7 × 36 ≈ 25), comme à 8 k où la bande au mur
(−18 à −26 %) valait 70 à 100 % de la borne GPU. Le seuil FAUX (15 %, soit 42 % de la borne) reste en dessous de tout ce que
le −24,4 % laisse attendre. Il juge le mécanisme, pas la précision de ma bande.

## Amendement 3 du 02/10 (14 h 1x), après la prise de 13:53, AVANT toute mesure à 14 k ou 12 k

La prise de 13:53 (`poste5-zzs-b`) a rejoué l'étape 0, identique (E1(a) tombe, E2 vert). Puis le nouveau contrôle a refusé
le bras A1 : à `--max-model-len 16512`, le plan exile encore 1 MLP (19,3 Gio de poids et 7,16 Gio d'activations réservées
pour 29,5 libres), et les graphes sont coupés. Aucun chiffre. Le « 17 562 tiendrait sans exil » du premier serveur était une
borne basse, fausse ici. L'arbre importé était bien le worktree (ligne `ARBRE` du journal).

**Preuve à sec, par le planificateur, sur le vrai manifeste** (`outils/gpu/mesure/zzs-plan-a-sec.py`). Le rig est lu par
nvidia-smi ; `_plan_from_manifest(max_concurrent_seqs=1)` tourne avec `torch.cuda.mem_get_info` remplacé ; aucun contexte
CUDA n'est créé. **Calibration, qui peut rendre faux** : la mémoire libre F est la seule inconnue. F = 30 208 Mio reproduit
au mot près les DEUX plans observés : à 32 896, « 6 MLP de plus… 15,7 Gio pour 29,5 libres » (exil 5) ; à 16 512, « 1 MLP de
plus… 19,3 pour 29,5 ». Avec F = 30 720 ou plus, la simulation n'exilait rien à 16 512 : elle aurait été fausse.

| contexte | F = 30 208 (calibré) | F − 0,5 Gio | F − 1 Gio |
|---|---|---|---|
| 14 464 (L = 14 336) | **résident** (0 MLP, 0 expert partiel) | 1 couche à experts partiels | 1 MLP |
| 12 416 (L = 12 288) | **résident** | **résident** | 1 couche à experts partiels |

14 336 tient, avec une marge inférieure à 0,5 Gio. 12 288 tient avec une marge de 0,5 à 1 Gio. Ordre de la prise, comme
convenu avec chef : 14 336 d'abord, puis 12 288 aussitôt si le contrôle de régime refuse A1. Bandes (écrites avant toute
mesure à ces longueurs ; même règle qu'à l'amendement 2, bas = 0,7 × la borne, borne extrapolée du ht9) :
**14 336 : −23 à −33 % (borne −33,3 %) ; 12 288 : −22 à −31 % (borne −31,3 %) ; FAUX si le gain au mur est < 15 %**.
Le contexte du serveur suit L (max(L) + 128).
