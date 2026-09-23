# Protocole — nsys sur le rejeu b=12 Coder-30B : trous de lancement et dépendances

Laure, 15/09/2026, avant mesure. Ordre : Sage § 5
[`sage-reprise-15-09-b.md`](sage-reprise-15-09-b.md) (main 6521562), 30 min de
carte. Moteur : main du jour (v0.6.5 : MMA au godet 12, route+pack), commit
relevé au journal, constantes relues (`[PREUVE]`).

## Montage

`nsys profile --cuda-graph-trace=node -t cuda,nvtx` sur
`scratchpad/nsys-rejeu-b12-15-09.py` : Coder-30B, b=12 constant (12 invites
de 256, ctx 2048), graphes capturés, 30 pas de chauffe, puis **50 pas de
décodage purs**, chacun dans une plage NVTX `pas` (avec `synchronize` par
pas : la durée de plage = pas GPU + retour hôte), le tout dans `mesure`.
Analyse (`nsys-trous-analyse-15-09.py`) sur le .sqlite : par pas, **union**
des intervalles de noyaux (les recouvrements multi-flux ne comptent pas
deux fois) contre durée de la plage → **trou = pas − union** ; somme brute
publiée aussi ; tableau par poste (MoE GEMM, projections, lm_head,
attention, normes, route+pack, quant, glue MoE, elementwise, autre) par
regex sur les noms démanglés — les noms réels sont publiés pour que le
classement soit vérifiable. Médiane sur 50 pas.

## Seuils de Sage (scellés)

trous ≥ 1,5 ms → P8 PDL devient le chantier ; < 0,8 ms → fusion
normes/routage/quant.

## Ma prédiction (scellée)

Sous graphes, un nœud coûte ≈ 0,61 µs de creux (`un-noeud-de-graphe-coute-
0-61-us`, 10/09) ; route+pack a ramené le pas à ≈ 1 500 lancements
(Laurine : 3 677 → 1 517) → **creux de lancement ≈ 0,9 ms**, plus les
dépendances entre petits noyaux sous-occupés (normes, quant, glue : ~200
noyaux de 5-15 µs qui ne remplissent pas la carte — mais ce n'est pas un
« trou », c'est du noyau court). **Trou total prédit : 0,9-1,3 ms sur un
pas de ≈ 14,5-15,5 ms** — entre les deux seuils de Sage : ni PDL seul, ni
fusion seule ne tranche ; le poste dominant restera MoE GEMM (≈ 45-55 %),
puis attention (≈ 15-20 %). Réfuté si trou ≥ 1,5 (PDL) ou < 0,8 (fusion).
Alarme : le `synchronize` par pas ajoute ~20-40 µs hôte à chaque plage —
compté dans le trou, publié à part (pas sous nsys sans sync : 50 pas
chronométrés aussi).
