# GEMV groupée des experts v2 — à sec (Laurine, 18/09) ; porte = micro-banc de Laure

Commande : sage-lecture-profils-coder-17-09 § 2 (priorité finale du circuit) :
décodage Coder b=12, GEMV d'experts 6,6 ms à 68 % de bande (8,1 Go/pas),
plancher 4,5 ms ; scellé **6,6 → ≤ 5,3 ms**, b=12 nu 1 114 → **≥ 1 300 t/s**
(faux < 1 200).

## Ce que fait v1 (`nvfp4_gemv_grouped_gateup_kernel`, `_warp_kernel`)

Un bloc par PAIRE (expert, jeton) et par tranche de 8 lignes : les poids de
l'expert sont relus une fois par paire — 96 paires pour ~69 experts
distincts à b=12 (× 1,39), les paires arrivant dans l'ordre des jetons (un
expert repris 3 jetons plus loin peut être déjà sorti du L2 : 183 Mo par
couche pour 96 Mo de L2) ; l'activation est étagée en mémoire partagée à
chaque bloc (4 Kio lus par bloc pour 16 Kio de poids).

## v2 (`acvram_kernels.cu` : `nvfp4_gemv_grouped_v2`, `_gateup_v2`)

- Les paires sont TRIÉES par expert côté hôte (`torch.argsort(eid,
  stable=True)`, déterministe, sous graphe) ; le noyau reçoit
  `eid_s`, `tok_s` et `ordre` (place d'origine de chaque paire) et ÉCRIT
  chaque sortie à sa place d'origine : act, down, `moe_reduce` inchangés.
- Bloc « meneur » : au début d'un segment d'expert, ou à un multiple de TPB
  depuis ce début (fil 0 remonte le segment, diffuse par la mémoire
  partagée) ; il sert jusqu'à TPB jetons (4 pour K ≤ 2 976, 2 jusqu'à
  5 952, 1 au-delà — l'étage tient en 48 Kio) : les poids de la ligne sont
  chargés UNE fois (uint4 par voie, comme v1) et dottés contre les TPB
  activations étagées ; les blocs non meneurs sortent aussitôt (coût : un
  bloc vide). Créneaux fantômes (e < 0) : zéro écrit, aucun poids lu.
- **Même arithmétique, même ordre** : `nvfp4_row_dot_warp_multi<TPB>` est
  `nvfp4_row_dot_warp` avec la boucle interne dupliquée par jeton (mêmes
  a0/a1/c0/c1, même `(…) * gscale`, même réduction par shuffles) → sortie
  identique AU BIT à v1 — le juge.
- Côté hôte : `ACVRAM_MOE_GEMV = v1 (défaut jusqu'au scellé) | v2`
  (regime.py, cli.py) ; `MoEBlock._forward_grouped` trie et appelle gateup
  v2 puis `_grouped(…, tri=)` pour down. Les chemins table/AWQ/distinct
  restent v1.
- Compilation contrôlée à sec : `nvcc -c` (C++20, sm_86 + sm_120) sans
  erreur ; pas de carte ici pour l'exécuter.

## Juge : `tests/test_gemv_experts_v2.py` (carte requise, 9 tests skippés à sec)

v2 = v1 `torch.equal` sur gate/up (K 2048, M 768) et down (K 768, M 2048),
routage aléatoire Coder, un expert qui reçoit les 12 jetons (sous-segments
> TPB), fantômes (zéros) ; gate/up fusionné v2 = v1 ; **bras cassant** :
`ordre` décalé d'un rang → différent ; tri stable vérifié.

## Porte : `outils/banc-gemv-experts-18-09.py` (Laure, ~5 min)

Coder b=12, 20 routages aléatoires (~69 experts distincts), gate/up + down,
rejeu de graphe, octets = experts distincts × (gate+up+down) ; chaque
routage jugé identique au bit ; JSON avec ms/pas (× 48 couches) et verdict :
**v2 ≤ 5,3 ms/pas ET identique** OUVRE.

Prédiction scellée : v1 ≈ 0,135-0,145 ms/couche (6,5-7 ms/pas, ≈ 1,2 To/s,
le 68 % de Sage) ; v2 ≈ 0,095-0,110 ms/couche (**4,6-5,3 ms/pas, 1,55-1,8
To/s**) — le gain vient des relectures évitées (÷ 1,39) et de l'étage
d'activation amorti ; faux si v2 > 5,3 (alors les blocs non meneurs coûtent
ou le segment remonté sérialise : mesurer TPB 2 vs 4 par
`ACVRAM_GROUPED_RPW` et un routage sans répétition, où v2 = v1 attendu ±
3 %). En situ ensuite (Laure, 20 min) : `ACVRAM_MOE_GEMV=v2` sur Coder
b=12 : ≥ 1 300 t/s nu (faux < 1 200), ppl-decode-kv identique (bit).

## Verdict (Laure ee2d12d) : porte FAUSSE — v2 plus lente de 23 %

v1 7,88 ms/pas (1,13 To/s, 63 %), v2 9,66 ms/pas (0,92 To/s), sur 20/20
routages, bit-exact 20/20. Mes deux prédictions dépassées (v1 6,5-7, v2
4,6-5,3). Lecture de Laure, que je retiens : à 65-75 experts distincts pour
96 paires, la relecture v1 (× 1,37) est servie par le L2 — elle ne coûtait
pas de bande ; v2 paie le tri, les blocs non meneurs (27 % de blocs vides
par tranche de lignes) et une occupation moindre (33 Kio d'étage par bloc
contre 8) pour un bus qui ne lisait déjà rien deux fois. Le « 54 % relus »
du profil était un compte d'octets demandés, pas d'octets sur le bus — un
chiffre juste, hors sujet comme décision (fiche « un chiffre juste peut
être hors sujet »). v1 reste défaut ; v2 reste témoin (`ACVRAM_MOE_GEMV=v2`).
Ce que dirait un pas suivant, si Sage le veut : le 63 % de v1 n'est pas
la relecture, donc c'est la latence par ligne (2 uint4 par voie en vol,
étage d'activation + __syncthreads par bloc de 8 lignes) — plus de lignes
par bloc (ACVRAM_GROUPED_RPW=2/4, déjà exposé, jamais mesuré au banc) et
des chargements de la ligne suivante avant la réduction ; à mesurer avant
d'écrire une ligne.

Bug de test corrigé : `_routage` définissait `eid` avant la branche
« fantômes » (UnboundLocalError, 2 cas) ; Laure l'a rejoué corrigé, 9/9.

## Prérequis de la campagne RPW (sage-gemv-experts-rpw-18-09) — fait à sec

- `regime.py` : `Variable("GROUPED_RPW", "1")` et `Variable("GROUPED_OLD",
  "", torch="1")` — lues par le `.cu` (`std::getenv`, figées au premier
  lancement : **un processus par valeur**) ; `regime_ligne()` les nomme
  quand elles diffèrent (`[régime] ACVRAM_GROUPED_RPW=2 …`, vérifié).
  Les six autres `getenv` du `.cu` (INT8_GEMV_WARP, INT8_TRANCHE, PA_CHUNK,
  PA_ETAPE, PAGED_ALLOC, PA_SANS_COMPTEUR) déclarées HORS_REGIME (témoins
  A/B jamais mesurés comme défaut) ; `test_regime_noyaux` scanne désormais
  aussi le `.cu` — jusqu'ici aucune variable de l'extension n'était dans
  une table.
- `outils/banc-gemv-experts-18-09.py` : ligne de régime en tête et dans le
  JSON (`banc-gemv-experts-18-09-rpw{N}.json`, un fichier par processus),
  verdict RPW imprimé : v1 ms/pas ≤ 6,7 et bit-exact v2/v1 sur tous les
  routages (le bit-exact v1(rpw)/v1(rpw=1) n'est pas dans le banc : les
  deux processus ne se voient pas — comparer les `v1_ms`/JSON, et
  `ppl-decode-kv` en situ si le scellé tient).
- Mesure : Laure, rpw = 2 puis 4, témoin rpw = 1 en fin ; scellé unique
  min(rpw 2, 4) ≤ 6,7 ms/pas ; faux ⇒ une passe ncu bornée avant toute
  ligne de noyau. Ma prédiction : rpw = 2 → 6,9-7,4 ms/pas (l'étage
  d'activation amorti sur 16 lignes, mais toujours 2 uint4 en vol par
  voie), rpw = 4 → 6,6-7,2 ; scellé **non tenu** de peu — faux si ≤ 6,7,
  et je le souhaite.

## Défaut rpw = 4 (sage-rpw-defaut-18-09) — fait à sec

Laure (in situ ABAB Coder b=12) : rpw = 4 → 1 262 t/s nu / 1 162 bridé /
0,344 J (rpw = 1 : 1 114) ; seuil 1 300 non tenu (−3 %), non rouvert ; ma
prédiction « non tenu de peu » sur le banc était juste de forme, et le
scellé du banc (≤ 6,7) est resté faux — le levier existe (+13 % en situ) sans
atteindre la bande visée. Fait : `acvram_kernels.cu` (six wrappers) et
`regime.VARIABLES` : `ACVRAM_GROUPED_RPW` défaut **4** ; banc et script ncu
alignés. `ppl-decode-kv` au défaut : à Laure (carte). Ce qui décide de la
fermeture : la passe ncu bornée rpw = 4 de Laure
(`outils/ncu_gemv_experts_rpw_18-09.sh`) — je lis avant d'écrire.

Aussi : `cli.VARIABLES_LUES` manquait `ACVRAM_VERROU_GLOB` (4b83a2f sur
main) — `test_cadrage_perplexite` rouge en suite complète.
