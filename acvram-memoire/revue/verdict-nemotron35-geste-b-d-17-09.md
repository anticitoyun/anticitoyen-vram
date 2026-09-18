# Verdict — Nemotron-3.5-30B-A3B : geste (b) repli, geste (d) plancher relatif RÉVISÉ (le premier était fautif), converti retiré

poste2, 17-18/09. Suite de `verdict-controle-parc-ratio-norme-17-09.md` :
la garde ratio_norme s'est déclenchée pour de vrai sur Nemotron calibA
(145 tenseurs hors bornes malgré `MIN_ECHANTILLONS_AWQ=8`, sur la
conversion réelle complète). poste7 (`poste7-awq-relu2-garde-repli-17-09.md`)
a identifié la cause dans le code, pas "peu routés" : l'entrée de
`down_proj` chez nemotron_h est ReLU²(up(x)) (`config.py:563`,
`model.py:622`) — creuse par construction, les canaux jamais activés
s'écrasent au plancher `clamp(min=1e-6)` (`calibrate.py:302`) contre ~1
ailleurs, étendue 1,7e6× mesurée, indépendante du nombre d'échantillons.

## Geste (b) : repli identité par tenseur (commit `982ad80`, tient)

`convert_checkpoint` refusait toute la conversion dès qu'un seul tenseur
sortait de `[0,80;1,25]`. Nouveau comportement : chaque tenseur fautif est
REPLIÉ à l'identité individuellement (RTN, pas d'AWQ), sous un plafond de
50 % du total de tenseurs vérifiés — au-delà, refus comme avant (plus une
réparation ciblée). Manifeste : `tenseurs_replies_identite` (nombre, part,
liste des noms). `cli.py` renomme le dossier de sortie en `-repliN`
(REGLES §4). Ce geste tient sans changement.

## Geste (d), PREMIÈRE VERSION (commit `982ad80`) — RETIRÉE, elle-même fautive

`1 % de la médiane du tenseur` comme plancher. Prédiction poste7 : « 0 hors
bornes Nemotron ». Mesuré : **135/5935** tenseurs toujours hors
`[0,80;1,25]` — prédiction RÉFUTÉE.

Ventilation demandée par poste7 sur les 135 : TOUS des
`mlp.experts.*.down_proj.weight`, aucun `up_proj`/`gate_proj` — l'
explication ReLU² n'était pas mise en défaut par cette ventilation-là.
Mais l'examen de 10 des 135 (alpha retenu + étendue `s_max/s_min` avant/
après, sur les vraies statistiques bras-A et les vrais poids source) a
montré un second défaut, plus grave que « insuffisant » :

```
                    avant (1e-6)    apres (1% mediane)
experts.18.down     1389            623      (ameliore)
experts.3.down      2774            639475   (x230 PIRE)
experts.30.down     501             867      (legerement pire)
experts.32.down     140             361      (pire)
experts.33.down     1038            331346   (x319 PIRE)
experts.37.down     511             162589   (x318 PIRE)
experts.5.down      914             857      (ameliore)
experts.7.down      1169            15732    (x13 pire)
experts.8.down      302             9901     (x33 pire)
experts.106.down    1591            852      (ameliore)
```

6/10 AGGRAVÉS, certains massivement. Cause confirmée (poste7,
`poste7-awq-plancher-median-faute-18-09.md`) : la formule `1e-2 × médiane`
suppose MOINS DE LA MOITIÉ des canaux quasi nuls. ReLU² viole cette
hypothèse par nature dès qu'une majorité de canaux ne s'active jamais sur
le corpus — la médiane elle-même retombe alors près de zéro (mesuré :
`experts.3` a une médiane de `mean_abs` EXACTEMENT à 0,0, 64,3 % des
canaux sous 1e-5), et `1e-2 × médiane` devient un plancher plus bas que
l'ancien `1e-6`, élargissant l'étendue au lieu de la borner.

**Conséquence** : `Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-repli135`
est un converti produit par un instrument fautif. poste3 l'a mesuré (PPL
1,2634) avant que le défaut du plancher soit identifié — ce chiffre juge
l'instrument, pas la calibration bras A elle-même, et n'entre dans aucun
classement. **RETIRÉ**, renommé sur disque en
`...-repli135-INVALIDE-plancher-median-fautif` (conservé, pas supprimé).

## Geste (d), VERSION CORRIGÉE (ce commit) : plancher sur l'ÉTENDUE, pas une statistique de position

`_magnitude_avec_plancher_relatif` : `plancher = max(1e-6,
max(mean_abs)/4096)`. Une borne sur le MAXIMUM du tenseur, jamais sa
position centrale, tient quelle que soit la fraction de canaux nuls — y
compris à 90 % de canaux nuls (testé). Sur une activation sans motif
creux (Coder/GLM), le maximum est déjà proche des canaux bas et ce
plancher reste sous `1e-6`, sans effet (vérifié à 1e-3).

**Garde n°2** (nouvelle, même commit) : l'ÉTENDUE des échelles par canal
(`scaler.scale.max()/min()`) est le nombre qui relie directement la casse
à la PPL (1,7e6× → PPL 1,4301 ; 6e5× → 1,2634 ; ~630× sain) — le ratio de
norme est un SYMPTÔME en sortie, l'étendue est la cause côté échelle.
Repli à l'identité si étendue > 4096 (même mécanisme que le geste b),
défense en profondeur pour ce qui échapperait quand même au plancher
(`forced_scale`, qui contourne la recherche).

Cinq tests (dont le bogue du plancher par médiane FIGÉ en témoin
cassant, pas seulement décrit) : `tests/test_awq_plancher_relatif.py`,
`tests/test_awq_garde_repli.py` (réécrit avec des métriques forcées par
monkeypatch — le motif organique à un seul canal extrême ne suffit plus
à casser le plancher corrigé, ce qui EST le résultat attendu).

## Suite

Reconversion Nemotron avec le plancher corrigé : en cours (variante A,
AWQ partout borné à l'étendue 4096 ; variante B, `down_proj` des experts
en identité PAR DÉCISION — teste si AWQ a un sens sur une entrée ReLU²).
Scan du parc (garde n°2, étendue) à refaire après — le premier passage
(116 modèles, `verdict-controle-parc-ratio-norme-17-09.md`) n'a mesuré
que le ratio de norme, pas encore l'étendue.
