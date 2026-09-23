# H2 (2) — table site → nœuds → ms/pas, à sec depuis le sqlite existant (0 min de carte) : les 336 lancements ne sont PAS de la glue MoE par couche, ce sont des noyaux à la FRONTIÈRE DE PAS (Laurine, 21/09)

instrument : `scratchpad/laurine-b12-21-09/ordre-noyaux.py`, sqlite déjà exporté (`nvtx-h2.sqlite`, prise de Manon 09:15-09:21), aucune nouvelle carte
méthode : segmentation par repère `_route_fusee_kernel` (une par couche) ; histogramme sur les 48 segments × 50 pas = 2 400 segments-couche

## Ce que le CSV du 19/09 ne pouvait pas voir : la répartition N'EST PAS uniforme par couche
Sur 2 400 segments-couche, une couche « normale » a **13 lancements fixes** (route_fusee, Marlin gate·up ×2, moe_reduce, rmsnorm, `_etroit_reduit` ×2, rope, kv_write, `_partiel_reduit`, rmsnorm, Kernel2 wmma, splitK). **Les kernels du CSV agrégé qui ne collaient à aucune hypothèse H2 (`reduce_kernel` 147, `vectorized_elementwise` 441, `elementwise` 49, `_scatter_gather_elementwise` 49, `indexSelectSmallIndex` 49) n'apparaissent QUE sur 49 des 2 400 segments — un par TRANSITION de pas** (couche 47 du pas n → couche 0 du pas n+1), jamais sur une couche normale (vérifié : 0 occurrence dans les 2 351 autres segments).

**Site nommé** : la fenêtre où ces 14 noyaux tombent commence après le `moe_reduce`+`rmsnorm` de la dernière couche (fin du calcul du modèle) et se termine avant le `rmsnorm` d'ouverture de la couche 0 suivante — **c'est le pas de sortie complet** : logits (tête), échantillonnage, et tout ce que `runner.py::_emit`/`_sample_only`/`_consommer` posent entre deux pas (candidats de code, non vérifiés par NVTX ici : `_sample_only` — argmax/softmax sur `[12, 151936]`, `indexSelectSmallIndex` correspond à un `.index_select`/`gather` typique d'un `logprobs[..., cible]` ou d'un filtre de vocabulaire ; `_scatter_gather_elementwise` à une écriture par indices ; les 9 `vectorized_elementwise` à la comparaison EOS/longueur + cast). **Durée mesurée du bloc : 0,426 ms/pas** (moyenne sur 49 transitions) — à comparer au « hors noyaux 0,49 ms (6,6 %) » du budget nsys du 19/09 et au **trou H1 (0,6-0,7 ms) du scellé initial : même famille, pas une glue MoE distincte.**

## Correction du cadre H2
Ma note du 21/09 09h31 cherchait une fusion « glue par couche » qui n'existe pas sous cette forme : **il n'y a pas de nœuds MoE non attribués par couche** (les 13 lancements/couche sont déjà tous nommés, dont `_route_fusee`/`rmsnorm`/`Kernel2` déjà comptés dans le budget nsys initial). **H2 devient : le poste de 0,43 ms/pas de la tête + échantillonnage, déjà repéré par H1 comme le « trou »** — ce n'est pas un chantier séparé, c'est la même pièce. Aucune fusion à proposer pour Océane sur une « glue MoE » : elle n'existe pas dans ce régime.

## Table (b=12, moyenne/49 transitions)
| noyau | n/transition | candidat runner.py |
|---|---|---|
| `reduce_kernel` | 3 | `_sample_only` : softmax/amax sur `[12, 151936]` fp32 (2-3 réductions) |
| `vectorized_elementwise_kernel` | 9 | comparaison EOS/`stop_token_ids`/longueur, cast fp32→argmax dtype, masquage |
| `elementwise_kernel` | 1 | assemblage jetons/logprobs |
| `_scatter_gather_elementwise_kernel` | 1 | écriture indexée (logprob à la cible, ou `output_ids.append` côté device) |
| `indexSelectSmallIndex` | 1 | `logits[..., cible]`/`index_select` |

durée : 0 min de carte
suite : Maîtresse : (1) H2 fermé — pas de glue MoE à fusionner ; (2) le poste 0,43 ms rejoint H1 (pipeline en service) : Océane peut viser directement `_sample_only`/`_consommer` (argmax device déjà en place selon `graphes-preparer-divergence-14-09`, mais softmax/amax `[12,151936]` fp32 en pleine précision pourrait se réduire à bf16/torch.compile) au lieu d'une piste MoE séparée ; (3) confirmation encore utile mais non bloquante : LISEZ-MOI v2 eager (envoyé à Manon) donnera les vrais noms de sites via NVTX, pas seulement l'ordre — à jouer sans urgence, cette note suffit pour trancher H2 close.

## Rejouable
`python3 scratchpad/laurine-b12-21-09/ordre-noyaux.py scratchpad/laurine-b12-21-09/sorties/nvtx-h2.sqlite`
