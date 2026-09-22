# Pourquoi `Qwen3-Coder-30B-A3B-nvfp4-qkv-22-09` charge en `experts_layout=naturel` : les échelles AWQ de gate et up diffèrent, `experts_repli` n y est pour rien (22/09, poste1, à sec)

Symptôme (poste2 61708420, ccb39f2c, `verdict-familles-qkv-nvfp4-22-09`) : nouvel alias, manifeste experts nvfp4 identique à l officiel, mais 144 lancements experts, chemin `mma-a4` atteint, pas 8,62 ms contre 6,7.

## 1. La condition, fichier:ligne
`engine/moe.py:336-337` (avant la scission b6047cc9 : `engine/model.py:346-348`) :
```python
if g is not None and u is not None and torch.equal(g, u):
    awq["up_proj"] = g                       # fusion : le MÊME objet
awq["up_distinct"] = not (awq.get("up_proj") is g)
```
puis `engine/moe.py:446` (`model.py:457` avant) :
```python
elif awq.get("up_distinct") and _MARLIN_DISTINCT != "1":
    raison = "gate/up à entrées distinctes (tables AWQ séparées) : GEMV Marlin fusionné inapplicable…"
```
La table est **[E, K]** et `torch.equal` est **global** : un seul expert dont l échelle de gate diffère de celle d up fait tomber la disposition Marlin pour TOUTE la couche — et avec elle le préfill Marlin et le GEMV (b) du décodage, puisque la disposition est unique depuis le 18/09. Le refus lui-même est **juste** : le noyau fusionné ne sait pas lire deux entrées.

Deuxième condition, celle qui explique que d autres convertis passent — `engine/moe.py:257-258` : une table entièrement à 1 est écartée (`awq[nom] = None`), donc `up_distinct` est faux.

## 2. Les trois alias, mesurés à sec (`outils/diag-disposition-experts.py`, sans carte, sans charger le modèle)
| alias | `has_act_scale` | experts sans stats | gate ≠ up (couche 24) | disposition |
|---|---|---|---|---|
| `…-nvfp4` (13/09, officiel) | **non** | — | 0/128 (aucune échelle) | marlin |
| `…-nvfp4-qkvo-i8c` (18/09) | oui | **18 432** (tous) | 0/128 (**toutes à l identité**) | marlin |
| `…-nvfp4-qkv-22-09` | oui | 5 235 | **51/128**, écart relatif médian **0,185** (max 0,98) | **naturel** |
Couches 0 / 47 : 45/128 (écart 0,075) et 23/128 (0,061).

Donc la cause n est **ni** `experts_repli=identite`, **ni** la collecte obs-min en tant que telle : c est que, pour la première fois, les experts portent des échelles AWQ **réelles**, et que celles de `gate` et d `up` sont **cherchées séparément**. Le 18/09 passait parce que TOUS ses experts étaient sans statistique (échelles à l identité) ; l officiel du 13/09 parce qu il n en a aucune. La correction de la collecte (5 235 sans stats au lieu de 18 432) est ce qui a réveillé la condition — elle n est pas le défaut.

## 3. Cause en amont, fichier:ligne
`quant/convert.py:530-544` — la pré-passe d **alpha AWQ commun gate/up** (A7, `verdict-a7-alpha-commun-gateup-15-09` : neutre en qualité, ±0,05 % sur Llama-2-7B) exclut explicitement les experts MoE :
```python
if ".mlp.experts." in name or tensor.dim() != 2:
    continue        # « hors experts MoE — leur pile groupée obeit a une autre regle »
```
La justification citée renvoie à A6 (`verdict-a6-int8-snrfloor0-14-09`), qui portait sur `snr_floor`, pas sur l alpha commun. Or la pile groupée n obéit pas à « une autre règle » : elle **exige** l entrée unique, puisque gate et up lisent la même sortie de norme et que le GEMV Marlin les fusionne.

## 4. Correctif borné proposé (conversion, pas chargement)
Étendre l alpha commun gate/up **aux experts MoE, par expert** (`_precalculer_alpha_commun_gate_up`, une option `--alpha-commun-experts` d abord, défaut ensuite si la mesure tient) : les deux projections d un expert partagent alors leur table, `torch.equal` passe, la fusion a lieu, Marlin revient — sans toucher une ligne du moteur, et le repack continue de ne dépendre que du format des experts.
**Prédiction écrite avant** : pas 8,62 → **6,7-6,9 ms** (le régime de l officiel) ; PPL **+0 à +0,3 %** contre l alias actuel (A7 mesure ±0,05 % sur dense, et l écart gate/up médian est de 0,06-0,19 — l alpha commun tombe entre les deux). **Réfuté si** : la disposition reste `naturel` après reconversion (alors la cause nommée ici est fausse — à reprendre par `diag-disposition-experts`), ou PPL > +0,5 %, ou KL contre bf16 supérieure à celle de l officiel.
Ce qu il ne faut **pas** faire : forcer `ACVRAM_MARLIN_DISTINCT=1` — le GEMV fusionné lirait une seule entrée pour deux échelles, sorties fausses au décodage (c est le chantier C10, GEMV Marlin à une projection).

## 5. Contrôles à sec livrés
* `outils/diag-disposition-experts.py ALIAS…` : prédit la disposition et sa raison depuis le manifeste et les `act_scale`, sans carte (tableau ci-dessus).
* `tests/test_disposition_experts_awq.py` (3 verts, processeur) : gate == up → `up_distinct` faux et refus par la seule absence de carte ; **une seule** échelle d up changée sur un expert sur quatre → `up_distinct` vrai et refus « entrées distinctes » ; toutes les tables à l unité → écartées, Marlin gardé. Ce test casse si l on rend la fusion tolérante sans traiter le GEMV fusionné.
