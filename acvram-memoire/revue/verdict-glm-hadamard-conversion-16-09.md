# Reconversion GLM `--hadamard-experts` faite — NOMINAL/piles_ok, MAIS capture de graphes CUDA impossible en décodage

poste2, 16/09 (date système 17/09 selon l'horloge). Ordre chef (relayant
`ETAT.md` file immédiate #1) : lancer la reconversion GLM avec le
convertisseur Hadamard fusionné (`--hadamard-experts`), carte libre.

## En-tête de mesure

`outils/carte.sh`, une carte (`cuda:0`), en-tête vérifiée en tête de
journal : `acvram.__file__` = `.../anticitoyen-vram/acvram/__init__.py`,
commit `cbea64e` (contient `f4ca4c9`/`7f8f213`/`e667831`, le patch Hadamard
et la fusion avec le noyau de poste4 `7eb44da`). Source `GLM-4.7-Flash-
bf16` (58,2 Gio), sortie `GLM-4.7-Flash-srcbf16-nvfp4-hadamard512`.

## Conversion

```
tenseurs 9751, sortie 18,3 Gio (x3,17), SNR sortie moyen 21,3 dB
335 tenseurs promus int8 (plancher SNR 25), durée 414,5 s
```

Manifeste vérifié : 9024/9024 tenseurs d'experts portent `rotation:
"hadamard-512"` ET `hadamard_block: 512`, `has_act_scale: false` sur les
9024 (aucune échelle AWQ, conforme à la spécification — la rotation
remplace l'échelle, elle ne s'y ajoute pas). `calib_source` tracé
(corpus intégré `DEFAULT_CALIB_TEXT`, sha256 publié).

## Régime — NOMINAL, mais un défaut nouveau en décodage

`acvram serve --regime` (deux passes, reproduit à l'identique) :

```
régime NOMINAL — graphes=on couches_exilées=0/47 experts_exilés=0/2944 piles_ok=True
[acvram] graphes CUDA desactives, decodage en eager — capture impossible :
  RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
  capture unless the CPU tensor is pinned. Please use tensor.pin_memory()...
  graphes CUDA : actifs (decodage), 0 godets capturés d'avance
```

**Le champ `graphes=on` du régime ment sur l'état réel** : c'est un
drapeau de configuration (demandé), pas de résultat (obtenu) —
`graphs.py:390-430` (`GraphRunner._capture`, catch générique, réplique en
eager après avoir journalisé `self.raison`) confirme la capture RÉELLE
échouée (0 godets capturés contre 10 sur `-k48-w4a4`, même modèle, même
protocole, sans `--hadamard-experts`). Ce n'est PAS le warning OOM
transitoire déjà vu et sans conséquence sur `-k48`/`-k48-w4a4` (celui-là
apparaît aussi ici, ligne 2, indépendant) — c'est un second défaut, propre
à cette conversion : le décodage tourne en eager, jamais capturé.

Cause probable (à confirmer par poste4, pas d'intervention de ma part —
hors du périmètre convertisseur) : une copie hôte→carte non épinglée
quelque part sur le chemin de décodage groupé quand `hadamard_block=512`
est actif (le chemin sans rotation, `-k48-w4a4`, capture sans problème sur
le même modèle/même script).

## Conséquence pour la suite

Le scellé « pas ≤ 19,0 ms » (chantier Hadamard, prefill×3 attendu) ne peut
PAS se juger tant que la capture échoue — le décodage réel tournera bien
plus lentement qu'en graphes, un chiffre de temps mesuré maintenant ne
dirait rien du régime NOMINAL visé en production. La PPL (prefill ET
décodage, préfixe compris) reste mesurable telle quelle — le décodage en
eager donne la MÊME sortie qu'en graphes (le régime capturé est un
raccourci d'exécution, pas un chemin de calcul différent), seul le temps
est faux.

Rendu à chef : converti prêt pour la PPL de poste3 (prefill+décodage,
scellé ≤ 1,010, préfixe `[gMASK]<sop>`) ; le temps de décodage attendra le
correctif de poste4 sur la capture.
