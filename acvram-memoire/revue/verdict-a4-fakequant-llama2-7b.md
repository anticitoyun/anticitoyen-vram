# Verdict — fake-quant activations W4A4/W4A8, Llama-2-7B (13/09/2026)

Manon, bead `anticitoyen-vram-brd` étape 1. Mesuré sur la 5090 (carte
prêtée par Océane, rendue à Laurine puis à moi entre les deux régimes),
`python outils/fake-quant-a4-llama2-7b.py --pour-de-vrai --regimes a4,a8`.
Prédiction et seuil scellés **avant** mesure dans
`revue/prediction-a4-fakequant-llama2-7b.md`.

## Résultat

| régime | PPL | écart vs A16 (5,6102) | seuil ≤ 5,6663 | durée |
|---|---|---|---|---|
| A4 (E2M1, 224 projections hookées) | **5,7550** | **+2,581 %** | **DÉPASSÉ** | 141 s |
| A8 (E4M3, 224 projections hookées) | **5,6175** | **+0,130 %** | **RESPECTÉ** | 57 s |

224 projections hookées = 7 genres × 32 couches de Llama-2-7B, confirmé
au journal des deux passes — couverture complète, pas partielle.

## Contre la prédiction scellée

* **A4 : direction confirmée, magnitude largement surestimée.** Prédit
  « échec du seuil, PPL 6,5-12 » (fourchette fondée sur « W4A4 naïf coûte
  largement plus qu'1 % sans rotation », consensus 3/3 duck.ai). Mesuré :
  échec du seuil bien réel (+2,58 % contre +1 % toléré) mais **quatre à
  huit fois plus doux** que prévu — 5,7550, pas 6,5+. La direction de la
  prédiction est juste ; son amplitude ne l'était pas.
* **A8 : confirmée, dans la fourchette prédite.** Prédit « 5,61-5,66,
  marge étroite ». Mesuré 5,6175 — à l'intérieur de la fourchette, plus
  proche du bord bas (confortable) que du bord haut (limite).

## Pourquoi l'écart de magnitude sur A4 (hypothèse, pas encore vérifiée)

La littérature citée (QuaRot, SpinQuant, DuQuant) mesure généralement le
coût d'un W4A4 **complet** — poids ET activations quantifiés de façon
statique/calibrée sur l'ENSEMBLE du modèle, souvent avec une méthode de
calibration différente de celle d'acvram. Ici, seuls les **7 projections
linéaires** sont fake-quantifiées **dynamiquement** (échelle recalculée à
chaque appel, pas de calibration hors ligne) — les normalisations, le
routeur, les plongements restent intacts (`keep_sensitive_16bit`), et
`down_proj` — genre le plus fragile documenté par Bridging Gap
(`piste-hadamard-downproj-refutee.md`) — reçoit ici le MÊME traitement
dynamique que les autres, sans aucun soin particulier. Un fake-quant
DYNAMIQUE peut s'adapter par appel à la distribution réelle du lot,
contrairement à une calibration statique figée à l'avance qui doit
couvrir tous les cas — ceci pourrait expliquer une partie de l'écart,
mais reste une hypothèse non testée ici.

## Ce que ce résultat dit pour le noyau W4A4 (bead brd)

* **W4A4 naïf ne suffit pas** : +2,58 % dépasse le seuil scellé de
  Jérôme. Un noyau `mxf4nvf4` qui fake-quantifie bêtement les activations
  sans rotation/lissage produirait un modèle mesurablement dégradé.
* **W4A8 (mxf8f6f4) est un repli viable dès maintenant** : +0,13 %,
  large marge sous le seuil. Si le débit du GEMM `mxf8f6f4` sur sm_120a
  justifie son coût mémoire (A en 8 bits au lieu de 4), c'est le régime
  à cibler en premier — sans attendre une solution au problème W4A4.
* Le vrai gain de la MMA native (`mxf4nvf4`, ×7,9 selon Laurine) reste
  hors de portée sans une transformation des activations (Hadamard,
  SmoothQuant, ou calibration statique meilleure que le fake-quant
  dynamique testé ici) — la piste 2 réfutée
  (`piste-hadamard-downproj-refutee.md`) montre déjà qu'une rotation
  aveugle ne suffit pas non plus sur les POIDS en NVFP4 ; rien ne dit
  qu'elle suffirait mieux sur les activations.

## Ce qui reste

* Isoler la contribution de `down_proj` seul (fake-quant sur ce genre
  uniquement, les 6 autres en A16) pour vérifier l'hypothèse de fragilité
  différenciée par genre.
* Tester une calibration STATIQUE (comme les poids) pour A4, plutôt que
  le fake-quant dynamique mesuré ici — pourrait creuser ou combler
  l'écart avec la littérature.
* Détail complet du relevé A4 perdu par un bug d'écrasement du script
  (deux invocations successives sur le même `--sortie` s'écrasaient
  l'une l'autre au lieu de fusionner — corrigé dans le même commit que
  cette note) ; seuls PPL/écart/seuil du régime A4 ont pu être reconstitués
  depuis le journal, pas le détail par tranche de contexte.
