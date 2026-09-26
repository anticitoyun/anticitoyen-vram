# Verdict — saturation haute (gate/up) + entrée down_proj, GLM (16/09)

poste1, à sec (aucun GPU), ordre poste7 §3 (main da2290f,
`revue/poste7-glm-pile-correctif-16-09.md`), suite au plancher bas écarté
sur gate/up (`revue/verdict-glm-blocs-zero-16-09.md`).

Script `outils/controle-saturation-16-09.py`, résultat complet
`/tmp/glm-discriminateur-mma0/resultat-saturation.json`.

## (2) Le haut — saturation à 448 sur gate/up

`nvfp4_quant_act_kernel` borne l'échelle décodée à 448 (E4M3 max,
`acvram_kernels.cu:2039`) : un bloc dont `amax > 6×448 = 2688` reçoit une
échelle trop petite et sature (`satfinite`) à ±6 en E2M1 —
`quantize_nvfp4` (poids) n'a pas cette borne.

Sur `x / s_gate[e]` et `x / s_up[e]` (échelles réelles, 8 experts,
16384 blocs poolés, même x réel qu'avant) :

| | max(amax) | p99,9(amax) | fraction amax>2688 |
|---|---|---|---|
| x / s_gate | 2,7027 | 1,3902 | 0,000000 % |
| x / s_up | 2,7027 | 1,3902 | 0,000000 % |

(x_sur_s_gate et x_sur_s_up identiques : `s_gate == s_up`, contrainte A7
déjà vérifiée par `model.py:820`.)

**VERDICT (2) : cause ÉCARTÉE.** Max mesuré 2,70, très loin du seuil 2688
(marge ×1000) — aucun bloc n'approche la saturation à ce point d'entrée,
sur cet échantillon.

## (3) Troisième entrée : down_proj

`act = silu(g)·u`, quatre bras (W4A16/W4A4 × sans/avec échelle `s_down`
réelle) sur `down_proj`, plus les deux fractions sur `act` et
`act / s_down` :

| bras | err |
|---|---|
| (i) W4A16 sans échelle | 0,093816 |
| (ii) W4A16 avec échelle | 0,107267 |
| (iii) W4A4 sans échelle | 0,129701 |
| (iv) W4A4 avec échelle | 0,152358 |

err(iv)/err(iii) = **1,1747** (< 1,5, ne confirme pas seul)

| | max(amax) | p99,9(amax) | fraction amax<2⁻¹/16 (bas) | fraction amax>2688 (haut) |
|---|---|---|---|---|
| act (brut) | 0,2312 | 0,0959 | **30,92 %** | 0,000000 % |
| act / s_down | 0,2886 | 0,1555 | **23,13 %** | 0,000000 % |

**VERDICT (3) : cause CONFIRMÉE**, par la fraction (critère OU) : **23-31 %
des blocs de 16** entrant dans `nvfp4_quant_act` côté `down_proj` ont un
`amax` sous le plancher dénormal E4M3 — ils sont **écrits entièrement à
zéro** par le noyau réel, alors que `quantize_nvfp4` (utilisé par le
discriminateur initial) ne les zéroterait jamais de cette façon (échelle
globale float32 non bornée par bloc). C'est un ordre de grandeur au-dessus
du seuil scellé (≥1 %) — pas une hétérogénéité marginale entre experts
(30,9 %/23,1 % sont des moyennes poolées sur 12288 blocs, pas un artefact
d'un expert isolé).

Cause : `act = silu(g)·u` a une magnitude structurellement petite
(post-non-linéarité, silu borné, produit de deux quantités du même ordre)
comparée à l'entrée brute `x` de gate/up (contrôle précédent : 0 % de
blocs sous le plancher côté gate/up). Le plancher bas — écarté sur gate/up
— **n'est pas écarté sur down_proj** : c'est le point d'entrée où le
mécanisme du plancher dénormal se manifeste réellement.

## Conséquence

Les trois contrôles (discriminateur initial, blocs-à-zéro gate/up,
saturation haute + down_proj) convergent : **(1a) application**, cause
localisée à `down_proj` — le plancher dénormal E4M3 sans échelle globale
zéro-te silencieusement 23-31 % des blocs d'activation à ce point précis.
poste4 : corriger `nvfp4_quant_act_kernel` (ou son usage pour `down_proj`
spécifiquement, `model.py:1028`) — soit une échelle globale par appel
(comme `quantize_nvfp4`), soit un plancher qui préserve l'information au
lieu de zéroter (ex. clamp au plus petit pas non nul plutôt qu'à zéro).
Équivalence pile/boucle **avec la table réelle du converti**, re-PPL
scellé ≤ 1,010 dans le même commit (ordre poste7, `poste7-glm-mma0-verdict`
§3). poste2 attend.
