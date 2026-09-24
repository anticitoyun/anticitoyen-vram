# Pièce 126 — profil nsys Qwen3.8-27B-nvfp4, b=8, décodage établi (24/09, ordre chef)

Contexte : 102 de poste2, NInfer +56 % débit / −35 % énergie vs acvram à b=8 sur ce modèle (égalité à b=1).

## Instrument

- `outils/carte.sh` (ACVRAM_NOM=poste3-piece126-nsys-qwen38, ACVRAM_DUREE_MAX=300), prise 66 s d'attente
  + profil, rc 0, cpu-safe (load 2,06 avant prise).
- Modèle : `/mnt/AI_GENERATOR/models_acvram/Qwen3.8-27B-nvfp4` (config : `Qwen3_5ForConditionalGeneration`,
  64 couches, 48 `linear_attention` (GDN, `full_attention_interval=4`) + 16 `full_attention`).
- `outils/gpu/mesure/frontiere-pas.py`, B=8, 20 pas, sous nsys (`-t cuda --cuda-graph-trace=node`).
- Commit worktree : 8f9210b2.

## Familles

`_dense_etroit_kernel` (gemm_dense_etroit nvfp4) est le chemin **PARTAGÉ** attention (q/k/v/o,
`attention.py:_multi_projection`) + GDN (`loader.py` : les `lin()` de KimiDeltaAttention/GDN passent par
le même `QuantLinear`) + MLP (gate/up/down, même mécanisme) + tête (`model.py:296-302`, DENSE_NVFP4_MIN_M
≤ b ≤ 32) — **pas séparable par nom de noyau** sans instrumenter les sites d'appel (lu au bit dans
`attention.py`, `model.py`, `loader.py`). Rapporté comme un seul poste plutôt qu'un éclatement inventé.
Les GEMM cutlass_80_tensorop bf16 vus dans la trace brute (26 k+ lancements) tombent TOUS hors des
fenêtres de décodage établi (prefill/chauffe uniquement) : 0 dans le régime mesuré, confirmant que le
chemin triton nvfp4 est bien le défaut actif à b=8 (`gemv_marlin_temoin` aussi à 0, cohérent avec
`regime.py:66`).

n=66 fenêtres jugées (sur 68, tête/queue exclues), mur (borne GDN à borne GDN) 26 184,5 µs/pas.

| famille | µs/pas | lancements/pas | % plafond |
|---|---|---|---|
| gemm_dense_etroit_nvfp4 (attn+GDN-proj+MLP+tête, partagé) | 21 072,3 | 401 | 80,5 % |
| auxiliaires (elementwise/reduce/copy torch) | 1 618,7 | 1 374 | 6,2 % |
| gdn_recurrence (fused_recurrent + chunk_* + conv1d + l2norm) | 1 155,6 | 96 | 4,4 % |
| normes (rmsnorm) | 492,1 | 129 | 1,9 % |
| copies (memcpy/memset CUDA) | 444,4 | 61 | 1,7 % |
| attention_pleine (paged_attn + rope + kv_write, 16 couches) | 397,7 | 64 | 1,5 % |
| mlp_activation (swiglu) | 70,6 | 64 | 0,3 % |
| gemv_marlin_temoin | 0,0 | 0 | 0,0 % |
| gemm_bf16_cutlass | 0,0 | 0 | 0,0 % |
| noyaux classés | 25 251,4 | — | 96,4 % |
| trou (hors noyaux) | 933,1 | — | 3,6 % |

## Verdict

Le poste dominant est de très loin le GEMM dense nvfp4 partagé (80,5 % du pas, 401 lancements) — c'est
lui qui porte l'écart avec NInfer à b=8, pas GDN (4,4 %) ni l'attention pleine (1,5 %). Reste non
distingué : quelle part de ces 401 lancements/401 relit les poids par godet vs par pas (comme signalé
`gemm_dense_etroit.py:3`, nvfp4_gemv relit par séquence à b>1 — vérifier si `_dense_etroit_kernel` a le
même défaut à b=8, ou s'il tient déjà la promesse « poids lus une fois par pas »). Piste suivante si
demandée : `--detail gemm_dense_etroit_nvfp4` par forme (Grd) pour situer combien de ces 401 sont
QKVO/tête vs MLP par TAILLE de sortie (dims connues : hidden 5120, intermediate 17408 → MLP largement
plus gros, distinguable par shape même sans instrumentation).
