# zzs — verdict : troncature causale du cœur MLA au préfill (poste5, 02/10)

Scellé : `poste5-zzs-scelle-01-10.md`, avec ses amendements 1 à 3 et la décision de chef (12 288 seul). Tous ont été écrits
avant les mesures qu'ils jugent. L'amendement 2 l'a été après la lecture du nsys à 8 k, ce qui y est dit. Code : `mla.py`
`_cles_vues`, témoin `ACVRAM_MLA_CAUSAL=0`. Modèle : Kimi-Linear-35B-kda-nvfp4, RTX 5090 (carte 0), b=1.

## Prises (verrou `carte.sh`, carte 0 à 15 Mio occupés au départ de chacune)

| prise | heure | ce qui compte |
|---|---|---|
| poste5-zzs | 13:11:15-13:18:07 | étape 0 et **nsys p81 A/B** (valides) ; ABBA en service **invalide** (exil, refusé par le comparateur) |
| poste5-zzs-b | 13:53:16-≈13:54 | étape 0 rejouée ; A1 refusé par le contrôle de régime (16 512 exile 1 MLP) ; aucun chiffre |
| poste5-zzs-c | 14:12:28-14:15:55 | **ABBA en service à 8 192 et 12 288**, régime résident prouvé dans chaque bras |

Pendant la première prise, une simulation d'poste6 a tourné environ 40 s sur un cœur. Elle n'a aucun effet sur nsys, qui
mesure la durée des noyaux au GPU, et l'ABBA de cette prise est jeté. Aucun calcul étranger n'a été signalé pendant poste5-zzs-c.

## Équivalence : E1 tombe, E2 tient au niveau du module ; divergence de bout en bout à la décision du chef

* **E1(a), au bit sur carte** (`tests/test_mla_causal_zzs.py`) : **FAUX**, à l'identique aux trois prises. À 8 192 sans
  passé, 262 473 éléments ≠, max 3,906e-3 ; à 3 000 + 1 000, 17 357 ≠, max 1,953e-3. Le noyau tronqué n'est pas au bit du
  chemin complet sur la carte. C'est l'issue (ii) du scellé : cuBLAS et le softmax choisissent leurs noyaux selon la longueur.
* **E2, module** (la sortie fp32 contre la référence fp64, formes de Kimi avec passé) : **vert** aux trois prises. La
  troncature reste dans 2 × l'erreur d'arrondi du chemin complet.
* **E1(b), bout en bout** (64 jetons gloutons, serveurs neufs, témoins **A1 = A2 et B1 = B2 au jeton près**) : 4 invites sur
  6 identiques. **8 192 k=1 diverge au pas 4, et 12 288 k=102 dès le pas 0** (le premier jeton). Le serveur, en
  sampler=graphe, ne rend pas les logprobs : le Δ logprob n'est pas mesuré, et l'on ne sait pas si ces divergences tombent
  sur des quasi-égalités. Comme le scellé le prévoit, elles sont publiées avec leur premier jeton, sans tolérance inventée
  après coup : **la décision revient au chef.**

## Temps

**nsys p81, préfill 8 k au GPU** (pilote à 8 448 de contexte, témoin prouvé dans le pilote A et absent dans B) :

| | A (toutes les clés) | B (clés vues) | Δ | prédit | FAUX si |
|---|---|---|---|---|---|
| préfill 8 k | 1 406,7 ms | 1 064,0 ms | **−342,7 ms (−24,4 %)** | −280 à −365 ms (−20 à −26 %) | gain < 150 ms |
| décodage b=1 | 3,1543 ms/pas | 3,1529 | −0,04 % | 0 ± 1 % | |
| décodage b=12 | 8,5818 ms/pas | 8,5850 | +0,04 % | 0 ± 1 % | |

Pour contrôle, la base A retrouve celle du ht9 (1 415,9 ms) à −0,6 % près.

**Mur du préfill en service** (poste5-zzs-c ; `/metrics` : graphes=True, repli_eager=0, kv_max_tokens=12 416 dans les
quatre bras ; régimes égaux à `mla_causal=0(temoin)` près ; médiane de 6 mesures par chemin) :

| L | A | B | Δ | bande scellée | étendue du témoin A |
|---|---|---|---|---|---|
| 8 192 | 1 440,1 ms | 1 043,5 ms | **−27,5 %** | −18 à −26 % → au-delà (mieux) | 2,7 % |
| 12 288 | 2 891,3 ms | 1 989,4 ms | **−31,2 %** | −22 à −31 % (borne −31,3 %) → sur la borne | 1,8 % |

**Ordre de rédaction** : la bande de 12 288 (amendement 3) a été posée APRÈS la lecture du −24,4 % au nsys à 8 k.
Son bord haut (−31 %, la borne extrapolée du ht9) en est indépendant. Son bord bas (−22 %, 0,7 × la borne) ne l'est
pas. Le seuil FAUX (15 %) juge le mécanisme.

Mesures brutes (ms) : A1 8 k 1 446,7 / 1 424,4 / 1 425,2, 12 k 2 886,6 / 2 866,3 / 2 869,9 ; B1 1 063,8 / 1 040,5 / 1 040,8,
2 002,4 / 1 982,2 / 1 984,9 ; B2 1 066,8 / 1 043,3 / 1 043,7, 2 009,2 / 1 989,8 / 1 989,0 ; A2 1 463,5 / 1 440,0 / 1 440,3,
2 917,8 / 2 896,0 / 2 897,6. sha256 des sorties (16 premiers) : A1 ce607fbbab70d15f, B1 5606e46f83dc9345,
B2 a028dd91fa6bd43c, A2 e9b40ff2000c92ed.

**FAUX non atteint** : le gain GPU à 8 k vaut 342,7 ms, au-dessus des 150 ms, et le gain au mur à 12 k vaut 31,2 %,
au-dessus des 15 %.

**Ce que je n'avais pas prévu** : à 8 k, le mur gagne 396,6 ms, soit 54 ms de plus que les noyaux GPU (342,7). Le gain
au mur dépasse donc la borne GPU (−25,7 %). C'est le contraire de l'issue (iii) du scellé, qui craignait un mur moins
bon que le GPU. Hypothèse, non mesurée : la part hors noyaux (allocation des tenseurs de scores, environ 1 Go par morceau
dans le chemin complet, et synchronisations de l'allocateur) baisse avec la taille des scores. Une trace nsys avec
`cudaMalloc` et l'API CUDA le dirait.

**Borne fausse** : la longueur longue du scellé (32 768) ne tient pas en régime résident sur la carte. Le chargeur exile
5 MLP à 32 896 de contexte et 1 MLP à 16 512, et son conseil « 17 562 tiendrait sans exil » était faux. Prouvé à sec par
`outils/gpu/mesure/zzs-plan-a-sec.py`, calibré sur les deux plans observés. Les prédictions à 32 k et 16 k restent sans
mesure.

## Décision (chef, 02/10)

* Le gain est réel et dans la prédiction, ou au-delà : −24 % au GPU à 8 k, −27 % et −31 % au mur, décodage intact.
* La sortie change : E1 au bit faux, et 2 réponses gloutonnes sur 6 divergent, dont une dès le premier jeton. **La
  troncature entre donc en OPT-IN** : `ACVRAM_MLA_CAUSAL=1`, défaut 0. Elle est déclarée dans `cli.VARIABLES_LUES`, au
  registre (`regime.py`, ligne `mla_causal=1(opt-in)`), dans le CHANGELOG « En préparation (0.7.18) » et dans
  `DEFAUTS_PAR_VERSION["0.7.18"]`. `tests/test_mla_causal_zzs.py::test_la_troncature_est_opt_in_defaut_off` casse si le
  défaut bascule, au registre, dans le module ou dans un processus neuf (mutation vérifiée).
* Le défaut ne bascule qu'après la **garde de PPL de décodage à 8 192 + 512**, scellée à sec dans
  `poste5-zzs-garde-ppl-scelle-02-10.md`, avec 9 tranches, A1 B1 B2 A2, et une bascule si Δ géo ≤ +0,3 %, Δ géo + 2 SE
  ≤ +0,6 % et max |Δ| ≤ 2 %. poste2 la mesure.
* Instrument : `outils/gpu/mesure/mla-causal-abba.{sh,py}`, désormais avec A = défaut et B = `ACVRAM_MLA_CAUSAL=1`. Il
  prouve le régime résident avant de mesurer et refuse toute réponse sans ses jetons, après deux prises perdues (exil à
  32 896, puis à 16 512).
