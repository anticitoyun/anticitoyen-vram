# Verdict — générique graphes/eager, Coder-30B b=4 W4A16 (16/09)

Océane, ordre Sage §12 (`revue/sage-duel-verdict-16-09.md`, main 9be9629),
rappelée sur scellé réfuté, carte 10 min (`outils/carte.sh`, relâchée en
fin de script, `nvidia-smi` libre après).

## Protocole

`Qwen3-Coder-30B-A3B-nvfp4`, `ACVRAM_HYBRID_SLOTS=4` (b=4),
`ACVRAM_MOE_DECODE_MMA=0` (W4A16), corpus réel (`wiki-gptq.txt`, 220
jetons pris, prompt 128 + 84 points). Référence = arbitre prefill (§9) :
préfixe croissant, `b=1`, sans graphes, une capture par position. Deux
bras de décodage teacher-forcé (jeton réel du corpus forcé à chaque pas,
`_sample_only` détourné, même montage que `outils/ppl-decode-mma-coder30b.py`) :
EAGER (`enable_cuda_graphs=False`) et GRAPHES (`=True`). Point compté
« hors quasi-égalité » si `top1_test ≠ top1_ref` ET
`ref[top1_ref] − ref[top1_test] > 1,0` logit (mesuré sur la référence).

Script `outils/arbitre-prefill-coder30b-16-09.py`, résultat complet
`/tmp/arbitre-prefill-coder30b-resultat.json`.

## Mesuré

**EAGER : 3/84 hors quasi-égalité — k = 10, 29, 81.**
**GRAPHES : 3/84 hors quasi-égalité — k = 10, 29, 81.**

Mêmes positions, mêmes jetons choisis, **mêmes deltas au dernier bit** :

| k | top1_ref | top1_test (eager ET graphes) | delta/ref |
|---|---|---|---|
| 10 | 6798 | 13255 | 1,201146125793457 |
| 29 | 36412 | 92301 | 1,687897682189941 |
| 81 | 28450 | 4312 | 1,474864959716797 |

EAGER et GRAPHES rendent des logits **identiques au flottant près** sur
ces trois points (et implicitement sur les 81 autres, non listés car
sous le seuil).

## VERDICT : aucun défaut générique graphes-spécifique — CLOS

Ni « eager≈0 et graphes≥3 » (les deux valent 3, pas 0) ni littéralement
« graphes≤1 et eager≤1 » (3>1) — mais le fait qui tranche n'est PAS le
compte, c'est l'**identité bit à bit** entre les deux bras : le rejeu
sous graphes ne diverge JAMAIS de l'eager sur ce montage (b=4, Coder-30B).
S'il y avait un défaut propre au `GraphRunner` (rejeu sur un tampon
périmé, table `ptrs` capturée, etc. — la famille de causes trouvée sur
GLM `_mla_lot`, §10-11), EAGER et GRAPHES ne pourraient pas coïncider au
bit : un chemin sans capture ne peut pas reproduire fortuitement l'erreur
d'un chemin qui rejoue une capture. Les 3/84 sont un écart **décodage
contre prefill**, présent identiquement dans les deux bras — exactement
le témoin déjà connu et cité par Sage (7,9 % de divergence top-1
décodage/prefill sur quasi-égalités, PPL décodage meilleure que le
prefill, `verdict-ppl-narrow-b12`). Rien de nouveau, rien de spécifique
à Coder, rien à propager à GLM.

## Conséquence

Le générique est clos, plus fermement que le scellé initial ne l'exigeait
(pas seulement « bruit sous le seuil », mais « aucune trace de rejeu
défaillant, prouvée par coïncidence bit à bit »). Pas de comparaison de
`x` d'entrée du pas 2 nécessaire (elle ne se justifie qu'en cas de
divergence graphes-spécifique). GLM `_mla_lot` (§11) reste un défaut
propre à GLM, sans lien de cause avec un mécanisme générique de rejeu —
Laurine continue seule sur ce chantier. Retour en veille.
