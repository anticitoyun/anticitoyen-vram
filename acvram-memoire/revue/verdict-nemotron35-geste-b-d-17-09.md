# Verdict — Nemotron-3.5-30B-A3B : geste (b) repli, geste (d) plancher relatif RÉVISÉ (le premier était fautif), converti retiré

Manon, 17-18/09. Suite de `verdict-controle-parc-ratio-norme-17-09.md` :
la garde ratio_norme s'est déclenchée pour de vrai sur Nemotron calibA
(145 tenseurs hors bornes malgré `MIN_ECHANTILLONS_AWQ=8`, sur la
conversion réelle complète). Sage (`sage-awq-relu2-garde-repli-17-09.md`)
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

`1 % de la médiane du tenseur` comme plancher. Prédiction Sage : « 0 hors
bornes Nemotron ». Mesuré : **135/5935** tenseurs toujours hors
`[0,80;1,25]` — prédiction RÉFUTÉE.

Ventilation demandée par Sage sur les 135 : TOUS des
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

6/10 AGGRAVÉS, certains massivement. Cause confirmée (Sage,
`sage-awq-plancher-median-faute-18-09.md`) : la formule `1e-2 × médiane`
suppose MOINS DE LA MOITIÉ des canaux quasi nuls. ReLU² viole cette
hypothèse par nature dès qu'une majorité de canaux ne s'active jamais sur
le corpus — la médiane elle-même retombe alors près de zéro (mesuré :
`experts.3` a une médiane de `mean_abs` EXACTEMENT à 0,0, 64,3 % des
canaux sous 1e-5), et `1e-2 × médiane` devient un plancher plus bas que
l'ancien `1e-6`, élargissant l'étendue au lieu de la borner.

**Conséquence** : `Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-repli135`
est un converti produit par un instrument fautif. Laure l'a mesuré (PPL
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

## Deux conversions avec le plancher corrigé

### Variante A — `-calibA-etendue4096-repli8` : AWQ partout, plancher borné

Prédiction Sage : « 0 hors bornes ET 0 étendue>4096 ». Mesuré :
**8/5935** tenseurs toujours hors `[0,80;1,25]` (0,13 %, contre 145 puis
135 avec les planchers fautifs — nette amélioration), repliés à
l'identité, renommage automatique en `-repli8`. **La garde n°2
(étendue) ne s'est jamais déclenchée** — le plancher borne bien
l'étendue à 4096 par construction, comme prévu. Mais 8 tenseurs restent
hors bornes sur le RATIO DE NORME malgré une étendue correctement
bornée : un résidu distinct, que le plancher sur l'étendue ne couvre
pas entièrement. SNR sortie moyen 23,1 dB, durée 283,5 s. Prédiction
partiellement tenue (étendue : oui : 0 ; ratio de norme : non, 8 pas 0).

```
Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-etendue4096-repli8/
  acvram-00000.safetensors : b456cbd09b7c9096e783cede978d48bec1a5802b4a13ad80e3e4883dadf26654
  acvram-00001.safetensors : fb5a52a7fb1dae2388b17a3229fcee595883ae75c844becfdaf5d7337e44416f
  acvram-00002.safetensors : aa6572d1c743c266947c199a9fb28257b701619ba80aa07fa031820fbd91fd33
  acvram-00003.safetensors : b60435f7a0678db7f1c1ce2cdcebc15cfaaa3b99c2e358715d68ab1a50273257
  acvram-00004.safetensors : 33d3bcddf61dbea9657835a64a2147daae80fc2b05f75783340e1e808ecfb09d
  acvram_manifest.json     : 765d4e288a3575f8205d3b1083e5d8cdffb4ccc831dd34dd6e384e97a6ec42b8
```

### Variante B — `-calibA-sansdown` : down_proj des experts en identité PAR DÉCISION

Les 2943 tenseurs `mlp.experts.*.down_proj.weight` sont exclus du
dictionnaire de statistiques AVANT conversion — pas d'AWQ du tout sur
ces tenseurs (identité explicite, arrondi au plus proche), AWQ normal
sur `up_proj`/`gate_proj`/attention/expert partagé. Aucun repli déclenché
(0 tenseur hors bornes, cohérent : sans stats, pas de recherche AWQ, pas
d'échelle instable possible). SNR sortie moyen 20,8 dB (plus bas que la
variante A, attendu : `down_proj` n'a plus aucune optimisation AWQ), durée
214,5 s.

```
Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-sansdown/
  acvram-00000.safetensors : e1608722a075dd94d0bd3f511058b60565ab0480874b033f0eb1d657c4c1829b
  acvram-00001.safetensors : 521cf8b6f7271d7d75ba878ccd6ae9419586cca02e071a0de4b39fe3f0d19e92
  acvram-00002.safetensors : 5945e33f1360fec7c16facad7d5497638475a5a7ae40b517744d64c1ce524c34
  acvram-00003.safetensors : 18e441be24f9dac2045fbe797c402199d4dea171dfd08811456102bc1e84a0a7
  acvram-00004.safetensors : 32428f96b1f27e11b492567f69b4cd6e9e2cff9dbd688c229dfb7489a15cacde
  acvram_manifest.json     : e5d2b9c8267334439724661b2b74d6eb359bcb2085e7abfe96596c2d0eb09059
```

## Suite

Prêt pour Laure : PPL 3 tranches sur les DEUX variantes (A et B), pour
juger si l'AWQ sur `down_proj` (entrée ReLU²) vaut son coût une fois
l'étendue bornée, ou si l'identité pure (B) fait aussi bien voire mieux.
Scan du parc (garde n°2, étendue) à refaire après — le premier passage
(116 modèles, `verdict-controle-parc-ratio-norme-17-09.md`) n'a mesuré
que le ratio de norme, pas encore l'étendue.
