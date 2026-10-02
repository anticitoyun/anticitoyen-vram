# Réduction exacte ÉTROITE : seulement sur le `F.linear` bf16 du chemin NVFP4 naturel (k_proj / v_proj), pas sur le processus — scellé AVANT le code et la carte (poste6, 02/10 23 h 3x, ordre chef)

Ordre : chef 02/10 23 h 2x — « la piste k_proj / v_proj hors du F.linear bf16 est la bonne suite : plus étroite que le
drapeau global, elle peut tenir le seuil. Écris seulement le scellé (prédictions débit à 512 / 1 024 / 2 048 / 4 096 et
équivalence au bit, découpé contre seul tenant), sans carte. » Aucun code moteur, aucune mesure avant ce commit.
Arbre lu : poste6-gemma-anneau b1f7b73fa.

## Ce qui est acquis (mesuré, `poste6-bf16-reduction-verdict-carte-02-10.md`)

* Le drapeau GLOBAL `allow_bf16_reduced_precision_reduction=False` rend morceaux = seul tenant au bit (C1), coûte +0,22 /
  +0,35 / +0,81 / **+2,40 %** de préfill à M = 512 / 1 024 / 2 048 / 4 096 (C5) et change la sortie du seul tenant à
  7 865 lignes (C3) — alors que k / v à M = 7 865 ne changent PAS avec lui (C6 : 0 %). Il touche donc d'autres produits.
* Le drapeau est lu À CHAQUE APPEL : le test sur carte de 22:39 le bascule entre deux `F.linear` du même processus et lit
  2 532 511 éléments différents puis 0. Une pose le temps d'un appel est donc possible.
* Seules k_proj / v_proj dépendent du découpage (H4 : q, gate, up, attention : 0). Elles passent par
  `acvram/kernels/__init__.py:931` et `:935` (`F.linear` du repli GEMM de `nvfp4_matmul`, pris pour M > `_NVFP4_GEMV_MAX`
  = 32, `:693` — le décodage à b ≤ 32 passe par le GEMV maison, hors cuBLAS) parce que `_PROJ_MARLIN_MIN_N` = 2 048
  (`:1270`) les exclut de Marlin.
* À N = 1 024, cuBLAS prend DÉJÀ la réduction exacte, sans drapeau, à M = 64, 1 024, 7 865, 8 192, 10 240 (C6 : 0 %).

## Trois voies, une retenue

| voie | arithmétique | pourquoi oui / non |
|---|---|---|
| **P — portée de l'appel** : le drapeau est posé à `False` autour des seuls `F.linear` de `:931` / `:935` (et `:884`, le repli à échelle par ligne), puis rendu à sa valeur d'avant | celle du drapeau global pour k / v — déjà mesurée (C1, C6, C7) ; tous les autres produits bf16 gardent le défaut | **retenue** : aucune arithmétique nouvelle, rien d'autre ne bouge |
| **T — P + tranches de 1 024 lignes** quand M > 1 024 | la même que P au bit si la réduction exacte ne dépend pas de M (mesuré : 0 entre 7 865 et 4 096 + 3 769) | bras de DÉBIT seulement : à M = 1 024 cuBLAS prend l'exacte de lui-même, donc sans surcoût de sélection |
| F — produit fp32 étroit puis arrondi bf16 | PAS celle de cuBLAS exacte : 99,71-99,97 % d'éléments égaux (C7), donc le seul tenant changerait partout | écartée comme correctif ; chronométrée au noyau, pour mémoire |
| Marlin pour k / v (`ACVRAM_PROJ_MARLIN_MIN_N=1024`) | autre arithmétique, à tous les M | écartée : change toute sortie, et le GEMV Marlin à M = 1 perd 17-25 % sur ces rôles (pièce 129) |

Réglage : `ACVRAM_BF16_REDUCTION` = `reduite` (défaut, inchangé) | `exacte` (global, existant) | **`etroite`** (voie P ou T
selon la mesure). Le régime le dit (`reduction_bf16=etroite`) ; le défaut ne bascule que par le chef, après la fenêtre.

À lire avant le code (fichier:ligne, non fait ici) : le drapeau est global au PROCESSUS — un autre fil qui ferait un produit
bf16 pendant la fenêtre d'un appel le prendrait en réduction exacte, et sa sortie dépendrait de l'ordonnancement.
`runner.py:822` (`self._lock`), `graphs.py:277`, `vision.py:190` : vérifier qu'aucun produit bf16 cuBLAS ne tourne hors du
fil du moteur pendant un préfill ; sinon la pose se fait sous le verrou du moteur, et un test le dit.

## Ce qui change pour qui — dit avant

La voie étroite ne peut PAS laisser toute sortie inchangée : rendre k / v indépendantes du découpage, c'est changer leurs
valeurs à chaque M où cuBLAS prenait la réduction réduite (C6, N = 1 024 : M = 33, 128, 256, 512 — 41-42 % des éléments à
1 ulp ; 2 048, 3 769, 4 096, 6 000 — 27-36 %). Un préfill d'un seul tenant de ces longueurs change donc de sortie sous
`etroite`. Ne change pas : M = 64, 1 024, 7 865, 8 192, 10 240, le décodage (GEMV maison), tout produit hors du chemin
naturel. Au noyau, la valeur nouvelle est la plus proche du produit fp32 (C7) ; ce n'est pas une garde de qualité au modèle.

## Prédictions (une fenêtre carte, garde de chaîne, comparateur corrigé — même candidat)

| # | grandeur | prédit | faux si / seuil de décision |
|---|---|---|---|
| E0 | le drapeau après un préfill sous `etroite` | rendu à `True` (valeur d'avant) ; ligne de régime `etroite` | reste `False` : la portée fuit, c'est le drapeau global sous un autre nom |
| E1 | produit nu N = 1 024, K = 5 120 par `nvfp4_matmul` : 7 865 lignes contre 4 096 + 3 769, sous `etroite` | **0 élément différent** ; témoin `reduite` : 15-31 % des éléments (H4 : k 15 %, v 30 % ; produit nu 31 %) | > 0 ; ou témoin à 0 (le test ne prouve rien) |
| E2 | même produit, `etroite` contre `exacte` (global) | au bit, à M = 33, 512, 1 024, 2 048, 3 769, 4 096, 7 865 | un élément diffère : la portée ne donne pas l'arithmétique mesurée |
| E3 | voie T contre voie P, mêmes M | au bit | un élément diffère : T abandonnée, P seule jugée |
| E4 | chaîne dense S1 (Devstral, 7 865 jetons), `etroite` : B (morceaux 4 096) contre A1 | **Δ 0 sur toutes les valeurs, ids égaux** (au bit, comme C1) | Δ > 0 : un autre produit dépend du découpage sous le défaut de torch — le drapeau global ferait plus que k / v |
| E5 | seul tenant A1 `etroite` contre A1 `reduite`, 7 865 jetons | **ids égaux, Δ 0** (k / v inchangées à M = 7 865 ; rien d'autre n'est touché) | Δ > 0 : la portée touche autre chose que ce que j'ai nommé — C3 une seconde fois |
| E6 | témoin reprise sous `etroite` (même candidat) | 0,01-0,05, premier jeton possiblement basculé : la reprise dépend toujours de la longueur des clés (REGLES § 4) | — (non décisif, dit) |
| E7 | chrono au noyau, `F.linear` N = 1 024, K = 5 120, exacte / réduite | M = 512 : × 1,0-1,6 · 1 024 : × 1,00 ± 0,03 (même réduction) · 2 048 : × 1,1-1,8 · 4 096 : × 1,5-2,6 ; T à 4 096 : × 1,0-1,4 ; F (fp32) : × 2-5 | 1 024 hors ± 3 % : le drapeau change le noyau choisi même là où le résultat est le même |
| **E8** | **débit du préfill moteur, `etroite` (P) contre `reduite`**, ABBA, 6 passes par bras, médiane | 512 : +0,0 à +0,3 % · 1 024 : 0,0 ± 0,3 % · 2 048 : +0,2 à +0,8 % · **4 096 : +1,4 à +2,4 %** | **> 2 % à un M** |
| **E9** | **même mesure, voie T** | 512 et 1 024 : comme P (aucune tranche) · 2 048 : +0,0 à +0,5 % · **4 096 : +0,2 à +1,2 %** | **> 2 % à un M** |

Sur E8, je ne prédis PAS que P tient le seuil : k et v pèsent 1,9 % des multiplications des
projections d'une couche (10,5 M sur 556 M par ligne : q 4 096, k et v 1 024, o 5 120, gate et up 32 768, down 5 120). SI leur
part du temps vaut leur part des opérations — hypothèse, non mesurée —, les 27 ms de surcoût du drapeau global à M = 4 096
(1 159 contre 1 132 ms) s'expliquent par k / v seules pour une réduction exacte ≈ 2,2 × plus lente, ce que E7 dira. Deux issues : (i) le surcoût global venait
surtout d'AILLEURS (attention, tête) → P ≤ 1 % ; (ii) il venait de k / v → P ≈ +2,0-2,4 %, seuil franchi comme le global, et
seule T peut le tenir. T repose sur un fait mesuré une fois (C6 : à M = 1 024 l'exacte est le choix de cuBLAS) ; si, sous le
drapeau, cuBLAS quitte ce noyau, T ne gagne rien (E7 à 1 024 le dit avant le moteur).

## Décision, fixée ici

* Bras retenu = le moins cher de P et T parmi ceux qui tiennent **E0, E1, E2, E4 et le seuil de 2 % à chaque M** (seuil du
  chef, inchangé). T ne peut être retenu que si E3 tient.
* Si un bras est retenu : `etroite` devient le candidat au défaut — la bascule reste au chef, la section « Ce qui change
  pour qui » jointe (les préfills courts changent de sortie).
* Si aucun ne tient 2 % : `etroite` reste opt-in comme `exacte`, défaut inchangé ; je le dirai, chiffres à l'appui.
* E5 faux : `etroite` n'est pas plus étroite que je le dis ; pas de défaut avant d'avoir nommé le produit en cause.

Issues qui me gêneraient : E8 ET E9 > 2 % (la voie étroite ne paie pas) ; E4 > 0 (H4 n'avait pas tout vu, la couche 0 ne
valait pas pour les quarante) ; E5 faux ; un fil concurrent qui rend la pose de portée dépendante de l'ordonnancement.

## Fenêtre prévue (à l'ouverture par le chef, budget revenu sous +1)

Une prise, garde de chaîne (`poste6-bf16-chaine`, jouée et tenue à 22:39) : test sur carte (E0-E3) → chrono au noyau (E7)
→ débit moteur P puis T (E8, E9) → chaîne dense S1 sous `etroite` puis `reduite` (E4-E6). Durée estimée sur l'horodatage
de la fenêtre précédente (3 min 35 s pour 8 bras) : ≈ 5 min. Test d'équivalence dans le commit du code, avant la carte :
sans carte, la pose et le rendu du drapeau autour de l'appel (casse si la pose est retirée, casse si le rendu est oublié) ;
sur carte, E1-E3 avec leur témoin.
