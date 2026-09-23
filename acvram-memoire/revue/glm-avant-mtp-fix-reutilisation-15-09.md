# -avant-mtp-fix : NON réutilisable tel quel sous le nouvel invariant

Manon, 15/09 soir. Question de Jérôme (relais Sage a6a7436) : le converti
`GLM-4.7-Flash-srcbf16-nvfp4-avant-mtp-fix` (AWQ experts, PPL ~0,998,
antérieur à tout correctif MTP/homogénéité de ce soir) est-il directement
utilisable avec le futur loader de Laurine, ou faut-il reconvertir ?

## Constat sur le manifeste existant

`acvram_manifest.json` de ce dossier, couche 1, `gate_proj` : 57 experts
`has_act_scale: True`, 7 `has_act_scale: False` (exemples : experts 2, 16).
Ces 7 n'ont NI clé de valeur d'échelle NI `experts_sans_stats` au niveau du
manifeste (champ absent — ce dossier précède son introduction). **L'absence
d'échelle y est représentée par une absence ambiguë** (`has_act_scale:
False` + rien d'autre), pas par l'identité EXPLICITE (des 1 écrits) que
Sage/Jérôme viennent de trancher comme invariant (commit `2205709`,
ce soir).

## Verdict

**Non directement réutilisable sous l'invariant retenu ce soir.** Le bloc
MTP mal routé (cause 1, corrigée par `740c91c`) est sans conséquence — le
moteur ne charge jamais `layers.47` (vérifié dans `loader.py`, boucle sur
`plan.layers`). Mais la représentation de l'échelle absente, elle,
diffère structurellement de ce que produit `convert.py` depuis `2205709` :
une absence ambiguë, pas une identité écrite. Si le loader de Laurine
distingue les deux cas (probable, puisque c'est exactement ce que Sage a
demandé de garantir), ce dossier donnera un résultat différent d'une
reconversion fraîche.

## Suite

Reconvertir `GLM-4.7-Flash-srcbf16-nvfp4` avec le `convert.py` actuel
(`2205709`) dès que le loader de Laurine est disponible, plutôt que de
réutiliser `-avant-mtp-fix`. Pas de mesure PPL/régime lancée ce soir sur ce
point précis (question de compatibilité manifeste, pas de qualité) — le
juge (PPL 0,998 ± 0,002 + doctor nominal + contrôle MTP) s'appliquera sur
le converti frais une fois le moteur poussé.
