# Réserve de préfill du chemin Marlin résident (+0,74 Gio à 65 536 sur gemma-4-31B) : scellé AVANT calcul et code (poste6, 02/10 14 h, ordre chef)

Constat (carte, `poste6-g6r-stabilite-verdict-carte-02-10.md`) : en NOMINAL à 65 536 la chauffe mesure un pic de préfill de
4,09 Gio pour 3,35 d'activations en formule, aux trois prises ; elle le rattrape par un excès (+11 Kio/jeton) appliqué au
chargement SUIVANT. Le premier chargement et les suivants n'ont donc pas le même plan (réserve complète, puis allégée).

## Mécanisme, lu dans le code (hypothèse tant que la carte ne l'a pas confirmé)

* Au-delà du tenu d'un seul tenant, le préfill passe couche par couche (`model.py:337`, `forward_tranches`) sous
  `kernels.depaquetage_partage(seuil_partage=False)`. Dans cette portée, chaque poids à disposition Marlin seule est
  dépaqueté UNE fois en bf16 [N, K] (`kernels/__init__.py:1331`, `_w_partage` → `marlin_port.depaqueter_marlin`) et **reste
  vivant jusqu'à la fin de la couche** (`_W_PARTAGES`, `:738-778`) : q, k, v, o, gate, up, down ensemble.
  gemma-4-31B, une couche : 84 + 42 + 42 + 84 + 3 × 220,5 Mio = **0,89 Gio** si les sept sont en Marlin (280 poids
  « seuls » pour 60 couches sur ce converti « 4 sur 6 » : pas tous).
* La réserve n'en compte qu'UN, le plus gros : `_plus_grosse_nvfp4_marlin_bytes` (`loader.py:2282`) = 0,22 Gio — juste
  pour un seul tenant (chaque poids rendu après son appel), trop court de Σ(couche) − max sous la portée de partage.
* En DÉGRADÉ le chemin Marlin se replie (`marlin(repli:exil)`) : pas de portée de partage coûteuse, pic 1,93 Gio.
* La chauffe compare le pic à `activations_prefill_bytes` SEUL (`contexte.py:293`), sans les termes Marlin que le plan
  réserve pourtant : l'excès enregistré recompte 0,22 Gio déjà réservés.

## Correctif proposé (aucun code avant ce scellé)

(a) `_plus_grosse_nvfp4_marlin_bytes` → sous le régime par tranches (plafond de morceaux posé ET contexte au-delà) : la plus
grande somme, par couche, des poids nvfp4 2D denses (projections q/k/v/o, gate/up/down ; pas les experts) en bf16 ; hors de
ce régime : le plus gros seul, comme aujourd'hui. (b) la chauffe compare le pic à activations + termes Marlin (la réserve que
le plan tient pour ces transitoires). L'excès mesuré reste le filet.

## Prédictions

| | prédit | faux si |
|---|---|---|
| R1 à sec, terme (a) sur le vrai manifeste gemma-4-31B | 0,70-0,90 Gio (aujourd'hui 0,22) | hors intervalle |
| R2 à sec, 65 536, excès 0 (premier chargement) | réserve hors tampons ≥ 4,09 Gio ; plan = celui du 2e chargement d'aujourd'hui (anneau, table hôte, 0 exilé, tampons denses non comptés) ; excès 11 par-dessus : toujours 0 exilé | réserve < 4,09, ou ≥ 1 exilé |
| R3 à sec, témoin 4 096 (pas de plafond) | identique à aujourd'hui | un champ diffère |
| R4 à sec, 8 192 / 16 384 / 27 648 / 41 984 (plafond posé) | **changent** : réserve +0,5 à +0,7 Gio, donc KV ≤ aujourd'hui ou plafond plus bas ; 0 exilé partout ; mêmes choix plein / anneau | un exil, ou un choix plein/anneau qui bascule |
| R5 relecture des chauffes déjà mesurées (8 192 : pic 1,64 ; 4 096 : 0,81 ; 27 648 plein 01/10 : 1,42 ; 65 536 : 4,09) contre la nouvelle réserve | pic ≤ réserve partout ; sur-réserve ≥ 2 × aux petits contextes (déjà vraie aujourd'hui : 3,43 pour 1,64 à 8 192) | pic > réserve quelque part |
| Carte (1 chargement à 65 536, ≈ 6 min) | NOMINAL 0/60, ids 5aabb6d5…, **excès enregistré ≤ 2 Kio/jeton** (pic ≤ réserve dès le premier chargement) | excès > 2 Kio/jeton : le mécanisme lu n'est pas (tout) le bon |

Issues qui me gêneraient : (i) R4 — le correctif touche des plans qui tenaient (réserve plus grosse là où la formule
d'activations a déjà 2 × de marge) : s'il coûte du KV sans rien protéger, la bonne décision peut être de ne garder que (b) et
le filet de l'excès, qui est stable depuis ce midi ; (ii) le terme lu (Σ d'une couche) n'explique pas tout le pic et l'excès
reste > 2 Kio/jeton ; (iii) d'autres modèles nvfp4 denses du parc (Devstral, Llama) voient leur KV baisser sans mesure.
Décision après R1-R5 à sec : à chef, avant tout code moteur fusionnable.

## Résultats à sec (02/10 14 h 49, APRÈS le scellé ; rien ci-dessus n'a été retouché) — recommandation : ne PAS appliquer (a)

Prototype sans code moteur : `scratchpad/poste6-g6r/sim_g6r.py` (`NOUVELLE_RESERVE=1` remplace le terme Marlin dans
`_reserve_prefill_detail` ; `EXCES_KIO=0` neutralise le fichier de chauffe laissé par la prise de 13 h 46 — un premier
passage sans cette neutralisation comparait des plans chargés d'un excès, jeté). Vrai manifeste, VRAM de G1, 12 s, verrou
vide avant et après.

| | prédit | à sec | |
|---|---|---|---|
| R1 terme (a) | 0,70-0,90 Gio | **0,995 Gio** (min 0,892 ; les 7 projections sont en nvfp4 sur 50 couches, 6 sur 10) ; plus gros seul 0,215 | FAUX, au-dessus |
| R2 65 536, excès 0 | réserve hors tampons ≥ 4,09 ; plan du 2e chargement d'aujourd'hui | 3,35 + 0,99 = **4,35 Gio** ; anneau, table hôte, 0 exilé, tampons non comptés, plafond 5 120 ; avec l'excès 11 en plus : 0 exilé | tenu |
| R3 4 096 | identique | identique (4 810 509 577 o, 606 blocs) | tenu |
| R4 plafond posé | réserve +0,5 à +0,7, KV ≤, 0 exilé, mêmes choix plein / anneau | 0 MLP exilé, mêmes choix plein / anneau — MAIS : **8 192 : plus de plafond (seul tenant) et table de plongements en RAM hôte** ; **41 984 : table en RAM hôte**, plafond 4 096 → 8 192, 2 684 → 2 624 blocs ; 27 648 : plafond 7 168 → 5 120 ; 16 384 : 10 240 → 8 192 | tenu à la lettre, effet de bord NON prédit (la table sort) |
| R5 chauffes déjà mesurées | pic ≤ réserve partout | 65 536 : 4,09 ≤ 4,35 ; 8 192 : 1,64 ≤ 3,65 ; 4 096 : 0,81 ≤ 2,04 | tenu |

**Lecture.** Le mécanisme lu colle au chiffre : 0,995 − 0,215 = 0,78 Gio de poids dépaquetés non réservés, pour un excès
mesuré de 0,74. Mais le correctif (a) déplace tous les plans au-delà du plafond — plafonds, et la table de plongements qui
quitte la carte à 8 192 et 41 984 — pour ne servir qu'au PREMIER chargement à 65 536, qui est déjà NOMINAL et que le filet
(excès de chauffe + tampons denses non comptés) stabilise depuis 13 h 52. C'est l'issue (i) du scellé.
(b) seul (comparer le pic à activations + Marlin) ramènerait l'excès de 0,74 à 0,52 Gio et la réserve du second chargement
au pic exact : la marge de 0,22 Gio d'aujourd'hui vient justement du double compte. À ne pas faire sans (a).

**Recommandation** (décision : chef) : ne rien changer au moteur ; garder ce scellé comme explication du +0,74 Gio ;
rouvrir (a) seulement si l'identité des plans entre premier et second chargement devient une exigence — et alors avec une
prise carte à 8 192 et 41 984, pas seulement à 65 536. Aucun code moteur écrit.
