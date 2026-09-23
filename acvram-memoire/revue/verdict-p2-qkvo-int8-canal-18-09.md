# Verdict — P2 : Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c converti

Manon, 18/09. Sage (relais Jérôme) : q/k/v/o de Qwen3-Coder-30B-A3B en int8
SYMÉTRIQUE PAR CANAL (une échelle par ligne, sans point-zéro variable —
cible torch._int_mm/cuBLASLt), reste identique au converti classé
(`Qwen3-Coder-30B-A3B-nvfp4`, `awq=False`, `group_size=128` ailleurs,
`mixed_precision=auto`, `snr_floor=25.0`, `max_promotions=0.15`). PPL
prédite 1,015-1,018, classé si ≤ 1,020 sinon « P2 fermé ». Laure mesure la
PPL réelle dans sa fenêtre P1.

## Ce qui a été construit

Nouveau mode symétrique dans `_quantize_int8` (`acvram/quant/formats.py:219`) :
zéro FIXE à 128 (au lieu de min/max par groupe), même conteneur `INT8Tensor`
— chargeur et noyaux runtime (`model.py:2446`, `mla.py:345`, qui testent
`getattr(w, "format", "") == "int8"`) n'ont rien à savoir du symétrique.
Transmis via `quantize_with_calibration` (`calibrate.py:591`) jusqu'à
`formats.quantize` (`formats.py:153`). `ConversionOptions.attn_qkvo_int8_canal`
(`--attn-qkvo-int8-canal` CLI) scope le changement aux quatre projections
d'attention SEULEMENT quand elles atteignent l'int8.

## Défaut trouvé en convertissant le vrai modèle (pas en test synthétique)

Le premier code ne gérait que le cas « self_attn déjà routé en int8 avant la
quantification » (comme sous cible `q3n`, `convert.py:358-367`). Sur le
converti réel, q/k/v/o partent en **nvfp4** (SNR 20,5-20,7 dB, sous le
plancher de 25 dB) et n'atteignent l'int8 QUE PAR PROMOTION
(`PROMOTE["nvfp4"] = "int8"`, branche plancher SNR `convert.py:1552-1589`) —
un chemin que je n'avais pas câblé. Premier converti (rejeté, refait) :
`group_size: 128`, pas de clé `symmetrique`, malgré `attn_int8: "canal"` au
sommet du manifeste — un mensonge de bilan. Corrigé : `attn_canal`/
`group_size_tenseur` recalculés sur le format PROMU (`wider`), appliqués aux
variables du manifeste seulement si la promotion est acceptée. Témoin
falsifiable ajouté (`tests/test_int8_symetrique_canal.py::
test_promotion_nvfp4_vers_int8_par_plancher_snr_est_aussi_symetrique_par_canal`) :
reproduit le motif sur un modèle jouet, plancher SNR forcé haut pour
déclencher la promotion.

Bonus, trouvé au même endroit : `cli.py` ne construisait plus AUCUN
sous-parseur sous Python 3.14 (`--speculative` de `acvram serve`, un `%`
littéral lu par `HelpFormatter._expand_help` comme `%d` contre un dict —
`TypeError: %d format: a real number is required, not dict`). Corrigé
(`%%`), commit séparé `3e4ad91`, sans rapport avec le P2 mais bloquant tout
`acvram convert`.

## Converti (deuxième essai, celui qui compte)

```
model.layers.0.self_attn.q_proj.weight  int8 canal  group_size=2048  snr 36,97 dB  (nvfp4 -> int8)
model.layers.0.self_attn.k_proj.weight  int8 canal  group_size=2048  snr 38,48 dB  (nvfp4 -> int8)
model.layers.0.self_attn.v_proj.weight  int8 canal  group_size=2048  snr 37,85 dB  (nvfp4 -> int8)
model.layers.0.self_attn.o_proj.weight  int8 canal  group_size=4096  snr 37,27 dB  (nvfp4 -> int8)
```

193 tenseurs promus au total, tous nvfp4->int8, tous q/k/v/o (48 couches ×
4). SNR post-promotion 37-39,7 dB — plus bas que le premier essai (41-43,9
dB, groupe de 128 affine standard) : le symétrique par canal coûte
mesurablement plus de précision que l'affine par groupe, cohérent avec la
prédiction de Sage (1,015-1,018 contre ~1,010 pour le classé). Aucune fuite
hors self_attn (contrôlé : aucun autre tenseur int8 ne porte `symmetrique`).

- source : `/mnt/4TO_SATACMR_2022/Modeles/models/Qwen3-Coder-30B-A3B-Instruct`
  (61 066 575 656 octets, sha256 dans le manifeste, `source.sha256`)
- convertisseur : commit `3e4ad91` (arbre modifié : fichiers non suivis du
  scratchpad seulement, aucun autre changement de code)
- sortie : `/mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`,
  17,1 Gio (nvfp4 15,2 / int8 1,1 / bf16 0,6), sha256 combiné des fragments
  dans le manifeste (`AWQ non deterministe : un converti est identifie par
  son sha256, jamais par ses options`, `avertissement_determinisme`)
- durée conversion : 686,1 s
- `attn_int8: "canal"` au sommet du manifeste, `symmetrique: true` +
  `group_size` = largeur d'entrée sur les 192 autres tenseurs q/k/v/o

## Tests

6 tests dédiés (`tests/test_int8_symetrique_canal.py`, formats.py isolé +
convert_checkpoint bout en bout, routage direct ET promotion, bras cassants
pour chacun) + suite ciblée sur les fichiers touchés (13 tests,
`test_manifeste_convertisseur.py`, `test_awq_garde_repli.py`,
`test_awq_plancher_relatif.py`) : tout vert. Suite complète NON relancée
(charge machine partagée, règle Sage du 18/09 : suites completes hors
frontiere de bloc interdites, tests cibles seulement).

## Suite

Laure mesure la PPL sur ce converti dans sa fenêtre P1. Seuil scellé par
Sage : ≤ 1,020 classé, sinon P2 fermé (pas de nouvelle tentative).
