# Devstral : scaling « llama 4 » porté sur q (poste4, 18/09, à sec, ~2 h)

Ordre poste7 (relais chef) : appliquer 1 + β·ln(1 + ⌊pos/8192⌋) sur q dans
l'attention llama sous rope yarn, β > 0 ; équivalence contre les logits
transformers ministral3 au-delà de 8 192 ; bras cassant β = 0 → rouge ; le
refus de 67aa280 (poste2) tombe quand le scaling est porté.

## Fait
- `model.Attention.__init__` lit `spec.rope_scaling` (yarn, `llama_4_scaling_beta`,
  `original_max_position_embeddings`) → `self.llama4 = (β, plafond)` ou None ;
  `_echelle_llama4(q, positions)` : s = 1 + β·log1p(⌊pos/plafond⌋) en fp32, `.to(q.dtype)`,
  la même arithmétique que `get_llama_4_attn_scale` (modeling_ministral3.py:105-107),
  appliquée APRÈS le RoPE comme chez HF (:145-150), sur les trois chemins : `forward`
  (préfill et décodage général), `decode_fixed` fusion rope_kv et chemin RoPE torch.
  Sous le plafond s = 1 exactement (aucun changement de sortie).
- `runner.Engine` : le refus « llama_4_scaling_beta non servi » ne tombe plus que si
  AUCUNE Attention ne porte le scaling (spec modifié après construction) ;
  `regime_ligne` : `llama4_scaling_beta=0.1(servi|non_servi)`.
- Branche de poste2 (67aa280 : table Mistral3 → llama, rope_parameters → rope_scaling pour
  ministral3, import fp8) fusionnée dans poste4 pour bâtir dessus.

## Preuve (à sec, CPU, fp32, hors pytest : le verrou carte de poste3 refuse la suite)
Point de contrôle jouet de forme llama (4 couches, H 256) sous config Mistral3/ministral3
(text_config, rope_parameters yarn facteur 48, plafond 8 192, β 0,1, mscale 1/1 → facteur
d'attention 1 comme Devstral), converti en bf16 plat (aucune quantification), contre
`Ministral3ForCausalLM` eager fp32 sur les mêmes poids ; logits du dernier jeton de 48
jetons placés aux positions :

| positions | écart relatif max, avec scaling | sans (bras cassant β = 0) |
|---|---|---|
| 0-47 (sous le plafond) | 8,5e-7 | 8,5e-7 (identique au bit à « avec ») |
| 9 000-9 047 (échelle 1 + 0,1·ln 2) | 1,2e-6 | **3,6e-3** |
| 16 384-16 431 (échelle 1 + 0,1·ln 3) | 8,8e-7 | **4,9e-3** |

Seuils scellés dans le test : < 1e-4 tenu, > 1e-3 pour le bras cassant. Le yarn d'acvram
(layers.py `_build_inv_freq`) coïncide déjà avec celui de HF (truncate, mscale = 1 sur
Devstral : `attention_factor` = 1, vérifié sur le config.json publié). Engine à
max_model_len 16 384 se construit, régime « (servi) » ; les 4 tests de refus de poste2
restent verts (spec modifié après coup = non servi = refus).

## Hors périmètre
Sur carte : le chemin `paged_attention` + `rope_kv` fusionné est couvert par le même
`_echelle_llama4` (un produit par élément sur q, b × 4 096 éléments par couche au
décodage) mais pas mesuré ; la vraie conversion Devstral (poste2) et sa PPL (poste3)
suivent l'ordre poste7.
