# bim — vLLM du parc morts au démarrage : verdict (poste5, 29/09)

* instrument : `parc/bin/vllm-serveur` (lanceur de la branche) + une requête de 32 jetons, `scratchpad/poste5-bim-29-09/prise.sh` ; à sec `smem_mla.py` (AOT sm_120)
* commit : poste5-bim f02221ff3 (ATTENDU vérifié par prise.sh) ; vLLM 0.29.0, triton 3.7.1, flashinfer 0.6.18, torch 2.13.0 cu130
* régime : 4 096 jetons, 5090 seule (CUDA_VISIBLE_DEVICES=0), gpu_memory_utilization 0,90 ; llama-server 5 606 Mio seul hors verrou, avant = après
* scellé : `scratchpad/poste5-bim-29-09/scelle.md` (poussé avant la prise)
* mesuré : Devstral-AWQ 200 (TRITON_ATTN, KV fp8, 144 s) ; Qwen3-VL-30B-AWQ 200 (TRITON_ATTN, KV fp8, 192 s) ; GLM-4.7-Flash-NVFP4 200 (TRITON_MLA, KV bf16, 167 s) ; Qwen3-Coder-30B-FP4 200 (TRITON_ATTN, KV fp8, 147 s) ; 0 OutOfResources, 0 « FlashInfer absent »
* verdict : VRAI (4/4, toutes les issues prédites)
* durée : prévue 15 min / tenue 651 s (carte.sh 09:26:38-09:37:29) ; drapeau edz posé 09:20, retiré à la fin (≈ 17 min, 6 min d'attente du dernier alias d'poste1) ; 1re tentative à 09:04 avortée en 3 s (verrou imbriqué)

## Deux causes, pas une (serveur.log : 22 démarrages sur 22 morts depuis le 21/09)
1. **FlashInfer sans nvcc (18 démarrages, tous les non-MLA).** `parc/bin/vllm-serveur` laissait vLLM choisir depuis
   le 21/09 → FLASHINFER → décodage dédié sm_120 `flashinfer_xqa_batch_decode_with_kv_cache`
   (vllm/v1/attention/backends/flashinfer.py:2065/2403) → `vllm/utils/flashinfer.py:140 _missing`, parce que
   `has_flashinfer()` (:68-83) est faux : pas de nvcc dans le PATH du service, pas de flashinfer-cubin, aucun xqa en
   cache JIT. Le choix du moteur ne consulte pas ce test : vLLM choisit un moteur qu'il sait ensuite indisponible.
2. **TRITON_MLA avec KV fp8 (4 démarrages, GLM-4.7-Flash ×2).** `triton_decode_attention.py:534`
   `_fwd_grouped_kernel_stage1`, BLOCK_N 32, BLOCK_H 16, num_stages 2 codés en dur (:496-520). AOT sm_120 à sec, avec
   alignement 16 : KV fp8 **102 400 o** (= journal, au bit), fp8 num_stages 1 83 968, KV bf16 **63 488** ; limite 101 376.
   `--enforce-eager` n'y change rien (noyau de décodage, pas de graphe).

## Correctif (lanceur seul, vLLM intact)
`choisir_moteur` (une logique pour le service et `--afficher`) : TRITON_ATTN par défaut ; MLA → moteur laissé à vLLM
(TRITON_MLA, seul moteur MLA sur sm_120) et KV bf16 ; FA2 non forcé → TRITON_ATTN ; choix de l'appelant respecté
(KVDT, VLLM_ATTENTION_BACKEND). Tests `tests/test_vllm_serveur_bim.py` : 5 verts, 4 rouges sans le correctif.

## Restes
* FlashInfer resterait possible (souvent plus rapide) avec nvcc 13 dans le PATH du service : le xqa sm_120 se compile
  alors au 1er démarrage, sous le verrou. C'est une pièce à part, avec mesure de débit contre TRITON_ATTN.
* Les 5 autres alias vLLM ne sont pas rejoués ici (même famille AWQ, même lanceur) ; edz les rejouera.
* Le script de prise lit mal la ligne « Using AttentionBackendEnum.TRITON_ATTN backend » (moteur=? au TSV) ; relu à la main dans les journaux.
