# Chantier Gated DeltaNet (qwen35 / qwen35moe / kimi-linear)

Objectif : exécuter les hybrides à récurrence linéaire — les vrais modèles
« kimi » du parc. Refusés proprement aujourd'hui ; ce document fixe ce qui a
été établi pour l'implémentation.

## Oracle
`transformers 5.16` (déjà dans le venv) contient l'implémentation de
référence : `Qwen3NextGatedDeltaNet`, `torch_chunk_gated_delta_rule`
(prefill), `torch_recurrent_gated_delta_rule` (décodage). Stratégie : nos
QuantLinear font les projections, la règle delta vient de ces fonctions de
référence (pures, sans poids) — comme exllamav3 pour l'EXL3. Second oracle de
bout en bout : llama.cpp sert le même GGUF (jetons greedy comparables).

## Mapping GGUF (vérifié sur Agents-A1-4B-kimi, h=2560)
| GGUF | référence transformers | forme |
|---|---|---|
| blk.N.attn_qkv.weight | in_proj_qkvz **sans z** (2·key_dim+value_dim) | [8192, 2560] |
| blk.N.attn_gate.weight | z (gate), séparé | [4096, 2560] |
| blk.N.ssm_alpha.weight / ssm_beta.weight | in_proj_ba scindé (a, b) | [32, 2560] ×2 |
| blk.N.ssm_conv1d.weight | conv1d dépthwise sur qkv | [8192, 4] |
| blk.N.ssm_dt.bias / ssm_a | dt_bias / A_log | [32] |
| blk.N.ssm_norm.weight | RMSNormGated par tête v | [128] |
| blk.N.ssm_out.weight | out_proj | [2560, 4096] |
| couches d'attention (1/4) : attn_q [8192,2560] | q (+gate fusionné ? à trancher à l'oracle) | q_norm/k_norm [256] |

Hyperparamètres (métadonnées GGUF) : `ssm.inner_size`=4096 (value_dim),
`ssm.group_count`=16 (num_k_heads), `ssm.time_step_rank`=32 (num_v_heads),
`ssm.state_size`=128 (head_k_dim=head_v_dim), `ssm.conv_kernel`=4,
`full_attention_interval`=4 (3 SSM : 1 attention),
`rope.dimension_sections`=[11,11,10,0] (mrope partiel des couches attention).

## À faire, dans l'ordre
1. gguf.py : mapping ci-dessus + hf_config type qwen3_next.
2. ModelSpec : layer_types (ssm|attention), paramètres linear_*.
3. engine/gdn.py : GatedDeltaNet (projections QuantLinear + règle delta de
   référence) ; conv1d causale avec état (fenêtre 4) par séquence.
4. loader : couches mixtes ; runner : états récurrents par séquence
   ({seq_id: (conv_state, S)} par couche) — prefix cache, spéculatif et
   graphes CUDA désactivés pour ces modèles en v1.
5. Attention des couches pleines : gate de sortie + mrope par sections.
6. Vérification : logits couche à couche contre transformers (tiny synthétique
   au format qwen3_next), puis jetons greedy contre llama.cpp sur le 4B réel.
