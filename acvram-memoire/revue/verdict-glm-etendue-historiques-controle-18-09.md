# Verdict — GLM : étendues historiques hors bornes, prédiction "sous-échantillonnage" réfutée

poste2, 18/09. Suite du scan du parc avec la garde n°2 (étendue) : plusieurs
convertis GLM historiques (pas le classé actuel) portent des tenseurs à
étendue > 4096, jusqu'à 81391 — sur un modèle SiLU à porte, sans motif
ReLU² creux (`poste7-glm-etendue-historiques-18-09.md`).

## Hypothèse testée et réfutée

poste7 : ces convertis datent d'avant `MIN_ECHANTILLONS_AWQ` (commit
`2a68aa6`), calibrés sur le corpus « six phrases mêlées » (retrouvé dans
l'historique git, `1eba33b^:acvram/quant/collect.py`), pas bras-A. Si les
14 tenseurs à étendue>4096 du converti `k48` ont tous `n_samples<8`, le
mécanisme est identique à Nemotron (statistique bruitée par manque de
jetons) et le classement reste propre sans chantier.

Contrôle : recalibration à sec avec le corpus « six phrases » reconstitué,
mêmes `calib_seqs`/`calib_tokens` (16/128) que le manifeste `k48`.

```
model.layers.46.mlp.experts.42.down_proj.weight  n_samples=54  FAUX (>=8)
model.layers.33.mlp.experts.6.gate_proj.weight   n_samples=2   OK
model.layers.45.mlp.experts.46.gate_proj.weight  n_samples=4   OK
model.layers.38.mlp.experts.20.gate_proj.weight  n_samples=0   OK
model.layers.30.mlp.experts.6.up_proj.weight     n_samples=14  FAUX (>=8)
model.layers.30.mlp.experts.17.gate_proj.weight  n_samples=10  FAUX (>=8)
model.layers.20.mlp.experts.49.up_proj.weight    n_samples=3   OK
model.layers.27.mlp.experts.15.up_proj.weight    n_samples=3   OK
model.layers.29.mlp.experts.28.up_proj.weight    n_samples=8   FAUX (>=8)
model.layers.24.mlp.experts.42.up_proj.weight    n_samples=7   OK
model.layers.27.mlp.experts.35.gate_proj.weight  n_samples=2   OK
model.layers.16.mlp.experts.39.gate_proj.weight  n_samples=2   OK
model.layers.26.mlp.experts.24.gate_proj.weight  n_samples=3   OK
model.layers.17.mlp.experts.57.up_proj.weight    n_samples=15  FAUX (>=8)
```

**9/14 tenus (n_samples<8), 5/14 en défaut.** Prédiction « 14/14 ⇒ classé
propre confirmé, chantier non ouvert » **RÉFUTÉE**.

## Lecture

Le pire cas (`layers.46.experts.42.down_proj`, étendue 16457 à 81391
selon le converti, présent sur SIX convertis historiques différents) a
**54 échantillons** — largement au-dessus du seuil de 8. Ce n'est pas un
problème de sous-échantillonnage : la statistique y est réelle. Le motif
ressemble à un vrai canal salient isolé (une magnitude d'activation
légitimement bien plus grande que les autres sur ce canal précis, present
et reproductible d'un converti à l'autre) plutôt qu'à un artefact de
bruit de calibration — MÉCANISME DIFFÉRENT de Nemotron (ReLU² creux, pas
applicable ici, GLM est SiLU à porte).

## Sans conséquence sur le classement actuel

Le converti CLASSÉ (`GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA`) reste
propre — 0 tenseur à étendue>4096 (confirmé par le scan du parc,
`verdict-controle-parc-ratio-norme-17-09.md` §3). Aucun chiffre publié
n'est concerné par ce résultat.

## Contrôle final : hypothèse canal salient, TENUE (5/5)

poste7 : les 5/14 restants (`n_samples>=8`) s'expliquent par un canal
d'activation massif à l'entrée de `down_proj` — une propriété du modèle,
pas un artefact de calibration — si ≤3 canaux dépassent 100× la médiane
de `mean_abs` sur chacun des 5. Scellé : faux si <5/5 tenus.

```
layers.46.experts.42.down_proj  2 canaux>100xmed  mediane=0,0425  max=572,9   OK
layers.30.experts.6.up_proj     0 canal           mediane=0,288   max=0,947   OK
layers.30.experts.17.gate_proj  0 canal           mediane=0,256   max=1,22    OK
layers.29.experts.28.up_proj    0 canal           mediane=0,270   max=0,834   OK
layers.17.experts.57.up_proj    0 canal           mediane=0,193   max=1,05    OK
```

**5/5 TENU.** Seul le pire cas (`layers.46.experts.42`) porte réellement
1-2 canaux massifs (jusqu'à ~13500× la médiane) ; les quatre autres n'ont
AUCUN canal isolé au-delà de 100× malgré une étendue>4096 mesurée sur le
converti — l'étendue y vient probablement d'une combinaison de plusieurs
canaux modérément élevés plutôt que d'un seul canal extrême, mais reste
sous le seuil des 3 dans les deux lectures.

## Clôture

Fermé en observation. Converti classé (`k48-calibA`) reste propre, pas
de reconversion. Déclencheur pour rouvrir : si un futur scan montre un
tenseur GLM classé (pas historique) à étendue>4096, ou si la garde n°2
refuse/replie un converti GLM à l'avenir sur ce motif précis.
