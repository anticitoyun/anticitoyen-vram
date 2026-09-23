# Témoins arXiv — 2605.00519 et 2601.09527 (16/09)

Océane, à sec, ordre Sage (main 6c7cfd9). Les deux IDs cités par Jérôme ne
sont dans AUCUNE section de `references-arxiv-et-editeurs.md` (ni A, ni B,
ni C) — à ajouter, pas encore fait. Titres confirmés : **2605.00519** =
« Silicon Showdown » (Javat & Kazakov, TensorRT-LLM/MLX/GGUF, RTX 5090 vs
Apple Silicon) ; **2601.09527** = « Private LLM Inference on Consumer
Blackwell GPUs » (Knoop & Holtmann, vLLM 0.12, RTX 5060 Ti/5070 Ti/5090).

**Attention confirmée** : le ×1,6 de 2605.00519/2601.09527 = NVFP4 contre
BF16 **au sein du même moteur** (TRT-LLM ou vLLM). Le ×1,46 de Laure =
TRT-LLM contre vLLM, moteurs différents, même format. Aucun rapport entre
les deux, ne pas les composer.

## Table

| modèle | quant | lot | ctx | moteur+version | carte | t/s | J (Wh/MTok) |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | NVFP4 | c8 | 8k | TensorRT-LLM 1.1.0 (PyTorch backend) | RTX 5090 | 151,4 | — |
| Qwen3-8B | NVFP4 | c8 | 8k | TensorRT-LLM 1.1.0 (Legacy C++ backend) | RTX 5090 | 93,1 | — |
| Qwen3-8B | BF16 | c8 | 8k | llama.cpp (GGUF, optimisé) | RTX 5090 | 92,2 | — |
| Qwen3-8B | BF16 | c8 | 8k | vLLM 0.12 | RTX 5090 ×1 | 260 | 403 |
| Qwen3-8B | W4A16 | c8 | 8k | vLLM 0.12 | RTX 5090 ×1 | 314 | 325 |
| Qwen3-8B | NVFP4 | c8 | 8k | vLLM 0.12 | RTX 5090 ×1 | 411 | 239 |
| Qwen3-8B | NVFP4 | c8 | 8k | vLLM 0.12 | RTX 5090 ×2 | 530 | — |
| Qwen3-8B | NVFP4 | c8 | 8k | vLLM 0.12 | RTX 5070 Ti ×1 | 211 | — |
| Qwen3-8B | NVFP4 | c8 | 8k | vLLM 0.12 | RTX 5060 Ti ×1 | 115 | — |
| GPT-OSS-20B | MXFP4 | api (court) | — | vLLM 0.12 | RTX 5060 Ti ×1 | 488 | — |

(38 lignes au total avec l'en-tête ; le reste des 79 configurations du
second papier n'est pas repris — RAG-16k, multi-LoRA, API-peak — hors
périmètre de nos cinq moteurs.)

## Témoins externes qui recoupent nos bras

- **Qwen3-8B NVFP4 vs BF16, vLLM 0.12, RTX 5090** : 1,58× (411/260) —
  même moteur, même format que notre propre bras vLLM. Comparable en
  nature (pas en valeur : leur Qwen3-8B ≠ notre Coder-30B).
- **TRT-LLM Backend Dichotomy** (2605.00519) : un même format NVFP4 rend
  151 ou 93 t/s selon le backend choisi — piège méthodologique direct pour
  notre propre comparatif TRT-LLM, à vérifier quel backend le duel
  compare.

## Signalé pour Laure (modèles qu'on a)

- **Qwen3-8B** : table complète ci-dessus, vLLM 0.12, RTX 5090, ctx 8k,
  c8 — reproductible tel quel (même moteur, même carte). Scellé ± 15 %.
- **GPT-OSS-20B** MXFP4 : 488 t/s, mais sur RTX 5060 Ti (nous n'avons que
  des 5090) — pas directement reproductible à carte égale, à signaler
  sans scellé.
- **Gemma3-27B** : W4A16 contre NVFP4 comparés dans le papier (BF16 exclu,
  hors VRAM même chez eux), mais aucun chiffre t/s exact trouvé dans le
  texte — tableau détaillé (Table 1-3 ou annexe) non extrait, à relire
  si Laure en a besoin.
