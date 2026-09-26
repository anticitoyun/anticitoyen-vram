# Verdict — pièce 217 (poste3, 25/09, ordre chef) : bilan de l'après-midi, A=be837ca1 → B=origin/main (4d9a4e40f)

* **instrument** : `outils/carte.sh` + `scratchpad/poste3-p217-25-09/cellule-217.sh` (gabarit
  `scratchpad/poste2-p190-25-09/cellule-190.sh`, ABAB×5, `-lgc 2700`, serveur neuf par passe) ;
  worktrees `travail/poste3-p217-A` (be837ca1), `travail/poste3-p217-B` (origin/main, 4d9a4e40f, 216
  inclus) ; capacité KV relevée via `/metrics` (`kv_max_tokens`) au démarrage de chaque bras.
* **scellé** : `revue/poste3-piece217-scelle-25-09.md` (avant), addendum daté après la 3e cellule
  (deux suppositions fausses nommées, prédictions non réécrites).
* **régime** : plein (reprise après pause de groupe, ordre chef) ; `ACVRAM_ATTENTE=5400`.
* **mesuré** : tableau ci-dessous. 4/5 cellules TENUES dans la bande prédite ; 1 FAUX net (Coder
  b=8), remonté immédiatement à chef pendant la mesure, isolation confiée à poste6 (220).

## Tableau (ABAB×5, médianes)

| cellule | débit A→B | débit Δ | prédit | J/jeton A→B | J Δ | prédit | KV A→B (jetons) | verdict |
|---|---|---|---|---|---|---|---|---|
| mixte-i8c b=8 | 397,0 → 420,9 t/s | **+6,02 %** | +4 à +9 % | 0,8051 → 0,7613 | **−5,44 %** | −3 à −8 % | 32 768 → 32 768 (saturé par la requête, 8×4096) | **TENU** |
| Qwen3.8-27B-nvfp4 b=8 | 498,2 → 496,6 t/s | **−0,32 %** | 0 ± 2 % | 0,637 → 0,6404 | **+0,53 %** | 0 ± 2 % | 32 768 → 32 768 (saturé) | **TENU (falsificateur confirmé neutre)** |
| Coder-30B-A3B-nvfp4 b=8 | 1648,6 → 1501,7 t/s | **−8,91 %** | +4 à +7 % | 0,1327 → 0,1601 | **+20,65 %** | −6 à −11 % | 32 768 → 32 768 (saturé) | **FAUX — falsificateur net, sens opposé** |
| Coder-30B-A3B-nvfp4 b=1 | 328,3 → 329,1 t/s | **+0,24 %** | 0 à +2 % | 0,6015 → 0,5949 | **−1,10 %** | ±3 % | 4 096 → 4 096 (saturé) | **TENU** |
| gemma-4-31B-it-nvfp4-vision b=8 | 406,0 → 406,2 t/s | **+0,05 %** | 0 ± 2 % | 0,7818 → 0,7819 | **+0,01 %** | 0 ± 2 % | **13 727 → 10 936 (−20,3 %)** | **TENU (falsificateur confirmé neutre) — KV cohérente avec 201 (−21,4 % publié à 8k ctx, ici 4k ctx, même ordre de grandeur)** |

## Lecture

Les deux falsificateurs du scellé (nvfp4 et gemma31, b=8) sont restés dans 0 ± 2 % — l'attribution
« 194/195b/209/201/212 ne touchent aucun chemin commun aux quatre autres cellules » tient. La
capacité KV de gemma31 confirme, à une autre longueur de contexte, le même mécanisme que la 201
(vision + MTP comptés dans la borne).

**Coder-30B-A3B-nvfp4 b=8 est un FAUX net et inattendu, dans le sens opposé à la prédiction** :
−8,91 % de débit et +20,65 % de J/jeton, alors que la 209 prédisait un gain. Journaux serveur
vérifiés : le chemin bascule bien naturel→marlin-w13 comme attendu (0 refus en B, 4 refus
« sous-normales » en A) — ce n'est pas un raté d'alias. Deux suppositions fausses trouvées et
documentées dans l'addendum du scellé :
1. La 209 a été scellée et mesurée sur l'alias **Coder qkvo-i8c**, jamais sur ce
   **Qwen3-Coder-30B-A3B-nvfp4** pur — et l'ampleur des sous-normales diffère de plus d'un ordre de
   grandeur (67 477 sur un seul up_proj ici, contre « 8 piles réelles / 70,8 M éléments » testés
   par la 209).
2. La 195b (int8 par canal) est ACTIVE sur cet alias (`etroites=serie+canal(table)` au journal),
   alors que le scellé la supposait inerte sur Coder.

Coder b=1 reste TENU (+0,24 %/−1,10 %) : à b=1 le chemin GEMV naturel domine déjà (209 le note
elle-même), donc l'effet marlin-w13/b=8 ne s'y applique pas — cohérent avec le mécanisme suspecté
(surcoût du facteur par ligne concentré au chemin tensoriel b≥8).

**Suite** : pièce 220 (poste6, isolation PAR_LIGNE=0/CANAL=0/les deux, processus séparés) doit
départager si c'est la 209 seule, la 195b seule, ou une interaction, qui régresse cet alias.

* **durée** : scellé 22:0x, cellules 21:14→22:42 (~90 min avec les files d'attente derrière la
  publication GitHub de chef et la mesure p213 de poste5), audit 219 et son tri en parallèle
  pendant les files d'attente.

## Addendum du 26/09 01h0x (poste3, ordre chef, sur constat poste6 226) — la cellule Coder b=8 est gardée telle quelle, sa lecture est corrigée

La cellule Coder-30B-A3B-nvfp4 b=8 ci-dessus (**−8,91 % débit, +20,65 % J**) N'EST PAS RETOUCHÉE —
la mesure elle-même n'a rien de faux, seule l'interprétation « la 209 régresse cet alias » l'était.
La 226 (poste6, `revue/poste6-piece226-verdict-26-09.md`) a montré la cause : le banc de la 217
(`scratchpad/banc-llamacpp-16-09.py`, fonction `invite()`) génère ses invites en JETONS TIRÉS,
sans texte réel — en génération libre sur un modèle MoE, cela produit des sorties dégénérées
(répétition d'un seul caractère) qui DIVERGENT entre les bras A et B, donc le ROUTAGE des experts
diffère entre les deux : les deux bras ne mesurent plus le même chemin de calcul, indépendamment de
la 209. À invites RÉELLES (banc chat), le même alias donne **+12,80 % de débit et −18,4 % de
J/jeton** avec la 209 — un gain, pas une régression. Les deux suppositions fausses nommées plus
haut (alias qkvo-i8c ≠ nvfp4 pur, 195b active sur Coder) restaient vraies mais ne portaient pas la
bonne conclusion : l'ampleur mesurée ici n'était pas un effet du volume de piles basculées, c'était
l'artefact du banc.

Corrigé en pièce 227 (poste3, branche `poste3-227`) : `scratchpad/banc-llamacpp-16-09.py` force
désormais des invites réelles (texte fixe du banc chat de la 102) pour tout modèle MoE en
génération libre, avec test (`tests/test_banc_moe_invite_227.py`) ; et `experts_layout`
(`acvram/engine/runner.py:_couverture_experts`) compte désormais par disposition
(`marlin-w13×44+naturel×4`) au lieu de retomber sur le seul mot « naturel » quand aucune couche
n'est littéralement `"marlin"` — c'est ce second défaut qui avait fait lire B « naturel » dans mes
journaux du 25/09 alors que 44 couches sur 48 étaient déjà en `marlin-w13`.

**Aucune cellule de la 217 n'est rejouée ici** — la 217 reste la preuve que le banc mentait sur ce
cas précis, la 226 reste la mesure corrigée qui sert de référence pour la 209.
