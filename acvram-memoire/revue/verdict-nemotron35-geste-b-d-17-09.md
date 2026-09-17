# Verdict — Nemotron-3.5-30B-A3B : geste (b) repli + geste (d) plancher relatif, prédiction (d) réfutée

poste2, 17-18/09. Suite de `verdict-controle-parc-ratio-norme-17-09.md` :
la garde ratio_norme s'est déclenchée pour de vrai sur Nemotron calibA
(145 tenseurs hors bornes malgré `MIN_ECHANTILLONS_AWQ=8`). poste7
(`poste7-awq-relu2-garde-repli-17-09.md`) a identifié la cause dans le
code, pas "peu routés" : l'entrée de `down_proj` chez nemotron_h est
ReLU²(up(x)) (`config.py:563`, `model.py:622`) — creuse par construction,
les canaux jamais activés s'écrasent au plancher `clamp(min=1e-6)`
(`calibrate.py:302`) contre ~1 ailleurs, étendue 1,7e6× mesurée,
indépendante du nombre d'échantillons.

## Geste (b) : repli identité par tenseur (commit `982ad80`)

`convert_checkpoint` refusait toute la conversion dès qu'un seul tenseur
sortait de `[0,80;1,25]`. Nouveau comportement : chaque tenseur fautif est
REPLIÉ à l'identité individuellement (RTN, pas d'AWQ), sous un plafond de
50 % du total de tenseurs vérifiés — au-delà, refus comme avant (plus une
réparation ciblée). Manifeste : `tenseurs_replies_identite` (nombre, part,
liste des noms). `cli.py` renomme le dossier de sortie en `-repliN`
(REGLES §4).

## Geste (d) : plancher relatif (même commit)

`search_channel_scales`/`search_channel_scales_commun` : le plancher sur
`mean_abs` passe de la constante absolue `1e-6` à `1 % de la médiane du
tenseur` (`_magnitude_avec_plancher_relatif`). Borne l'étendue
salience/plancher à ~100× quelle que soit l'échelle absolue du tenseur,
sans toucher les tenseurs sans motif creux (vérifié à 1e-3 par test).

## Résultat réel : prédiction (d) réfutée

Prédiction poste7 : « 0 hors bornes Nemotron ». Mesuré : **135/5935**
tenseurs toujours hors `[0,80;1,25]` après le plancher relatif (contre
145 sur un échantillon de 346 avant — 5935 est le total réel de tenseurs
vérifiés, pas un échantillon). Le plancher relatif RÉDUIT le problème
mais ne l'ÉLIMINE PAS : le motif ReLU² creux résiste encore, pour une
fraction des experts, même avec un plancher à 1 % de la médiane.

Grâce au geste (b), la conversion ABOUTIT quand même : 135/5935 = 2,3 %,
bien sous le plafond de 50 %, repliés à l'identité automatiquement.
Renommage automatique : `Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA` →
`Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-repli135`.

```
tenseurs         6243 ; entree 58,8 Gio ; sortie 18,5 Gio (×3,19)
SNR sortie moyen 23,0 dB
30 expert(s) sans statistique (jamais routes/trop peu, seuil de 8)
135/5935 tenseurs repliés a l'identite (ratio de norme hors bornes,
  motif ReLU2 resistant au plancher relatif)
duree 1667,4 s
```

## sha256 (`.../Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-repli135/`)

```
acvram-00000.safetensors : 58eae476ac7d47a2e16392bf5047e1f6e59b6af76a5d984a8b1796208d99c1f1
acvram-00001.safetensors : de1d09d01fa346df6645aa87b7c9c7789ef8a8903142ed871ba7afaf26c00396
acvram-00002.safetensors : cf4996ec4bd985456c7e70fe37a9f0f2d4ea8532f9aeda7dbd8daa97764856c4
acvram-00003.safetensors : a19ca29bb07097d8833b3aada066e4717283ca904db3e6997cbfdeaf815d8666
acvram-00004.safetensors : 2f95836d2147590bf4b8fa6384332f9224897d3b96838f051b905f98097a67dd
acvram_manifest.json     : c4aeb8f2531cd3f59a6701356b09ff4c69eb95b82d73ed820e8cd97e8b718c98
```

## Suite

Prêt pour poste3 : PPL 3 tranches. Le repli (geste b) est une protection
structurelle qui tient quelle que soit la cause exacte du désaccord
restant — la conversion est saine à publier même si le plancher relatif
n'a pas tout résolu. Pas d'hypothèse supplémentaire prête sur pourquoi
135 tenseurs résistent encore au plancher à 1 % ; en attente d'instruction
(accepter tel quel, resserrer le plancher, ou autre piste diagnostique).
