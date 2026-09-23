# Verdict — AWQ int8 sur q/kv/o (attention), GLM (16/09)

Océane, à sec (aucun GPU), ordre Sage §5
(`revue/sage-glm-pile-correctif-16-09.md`, main f127a2e).

**Note sur le seuil transmis** : le message relayé (Jérôme) donnait
« err(sans)/err(avec) ≥ 0,9 ». La note originale de Sage (§5.3) écrit
« err(avec)/err(sans) ≥ 0,9 » — sens inverse. Mesuré ci-dessous avec le
sens de la note originale ; dans ce cas précis les deux lectures donnent
le même verdict (voir plus bas), mais à signaler pour la prochaine fois.

## Protocole

Même montage que le discriminateur MoE : poids bf16 réels de
`GLM-4.7-Flash-bf16` (source), table AWQ réelle de
`GLM-4.7-Flash-srcbf16-nvfp4-avant-noawq-experts` (int8, group_size=128),
`x` réel (forward CPU d'un mini-répertoire 7 couches, bf16 pur, aucun AWQ,
même prompt synthétique que les contrôles précédents), erreur relative de
`x·Wᵀ` avec et sans échelle AWQ (`x/s`, `W·s`).

**q** = `q_a_proj`, **kv** = `kv_a_proj_with_mqa` — couche **6**, seule
couche parmi les dix premières où les deux ont une table AWQ réelle
non-identité simultanément (scan des `has_act_scale` sur les 47 couches :
44/47 pour q_a_proj, 33/47 pour kv_a_proj_with_mqa, intersection incluant
la couche 6). **o** = `o_proj` : **aucune des 47 couches n'a de table AWQ
réelle** (`has_act_scale=False` partout) — rien à mesurer, rien à retirer,
contribution nulle aux +2,06 ms par construction.

`DecoderLayerGDN` (nom historique) sert d'enveloppe générique MLA dans ce
dépôt — le module d'attention vit sous `.linear_attn` même quand ce n'est
pas une récurrence linéaire ; capturé là, pas `.self_attn` (qui n'existe
pas sur cette classe).

## Mesuré

| projection | err(sans) | err(avec) | err(avec)/err(sans) |
|---|---|---|---|
| q_a_proj | 0,005363 | 0,005267 | **0,9820** |
| kv_a_proj_with_mqa | 0,001507 | 0,001482 | **0,9833** |
| o_proj | — | — | (identité partout, 0/47) |

Erreurs déjà de l'ordre de 0,15-0,5 % sans échelle (confirme le point 1 de
Sage : « erreur déjà ~0,3 % » — int8 par groupe est intrinsèquement
précis, l'AWQ n'a presque rien à corriger). Sous l'interprétation inverse
(err(sans)/err(avec)) : 1,0183 et 1,0170 — également ≥ 0,9, même verdict
dans ce cas.

## VERDICT : ≥ 0,9 → **retrait SANS RISQUE**

Les deux ratios mesurés (0,9820 et 0,9833) dépassent nettement le seuil.
L'AWQ int8 sur q/kv (et trivialement o, qui n'en a pas) vaut moins de
2 % d'une erreur déjà minuscule : la recette 2 de Sage s'applique — AWQ
sur les tenseurs NVFP4 seulement, aucune table sur les projections int8
pour la reconversion de Manon. Le scellé du pas reconverti reste à une
seule variable (correctif W4A4 down_proj).

## Conséquence

Manon : reconversion avec `--no-awq` restreint aux projections int8
(experts et dense en nvfp4 gardent l'AWQ). Scellé du pas (Sage §5.4) :
≤ 1,012× contre `-sansawq`, réfuté > 1,03×.
