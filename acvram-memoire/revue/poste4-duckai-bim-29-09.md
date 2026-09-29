# Q bim (29/09) — vLLM sm_120 shared memory 102400/101376

Noyau fautif : `_fwd_grouped_kernel_stage1` (attention MLA, `triton_decode_attention.py`), pas fused_moe.
BLOCK_DMODEL=512, num_stages=2 → 102 400 o, 1 024 o au-dessus de la limite sm_120 (101 376 o). Source :
issue vLLM (garde `BLOCK_DMODEL >= 1024 or is_hip_` n'inclut pas MLA à 512) + dev.to (conatusai, cas FP8 KV
identique côté MLA). Fix direct : forcer `num_stages=1` sur ce chemin (patch source, pas de flag public) ;
vider `~/.triton/cache` après patch. `--enforce-eager` NE corrige PAS (issue le confirme, Triton compile
quand même). `VLLM_ATTENTION_BACKEND=FLASHINFER` contourne (évite le chemin Triton MLA) si supporté par le
modèle/KV-cache. Pas de version stable confirmée intégrant le correctif à ce jour (issue encore ouverte).

## Croisement duck.ai (29/09)

Gemma 4 31B et gpt-oss 120B accusent `fused_moe` — **non sourcé**, aucun fichier/ligne cité, écarté.
gpt-oss confabule en plus une variable `VLLM_FUSED_MOE_JSON` inexistante et un numéro de version vLLM
"0.4.5" incohérent avec le calendrier du projet — écarté.
GPT-5.6 Luna seul cite le fichier et le kernel exacts avec lien GitHub, cohérent avec la source primaire
(dev.to) — retenu.
