# Fiche d'alias — Qwen3-VL-2B-Instruct-bf16-vision (20/09, 21 h 15, Jérôme ; ordre Sage `sage-p3-1-clause4-int8-20-09`)

Alias servi par 0.6.34 (candidat, arbre 645f87df) : texte bf16, tour de vision bf16, deepstack [5,11,17], M-RoPE [24,20,20] entrelacé, masque image **causal** (famille Qwen3VL, porte `masque_images=causal`), KV int8.

| mesure | valeur | source |
|---|---|---|
| scellé (b″) contre transformers 2B fp32 (TF32 coupé), 20 images, kv=bf16 | dispersion 8,21 % ≤ 13,92 % (2 × témoin bf16 7,0 %) ; premier jeton 19/20 ; marges fp32 vraies sous seuil ; godets {1,2,8,16} avec image servis ; texte au bit (Coder i8c) | `verdict-p3-1-rejeu-masque-20-09` (avant correctif du masque : 22,2 %, 17/20 — `verdict-p3-1-qwen3vl-2b-20-09`) |
| **kv=int8 vs bf16 (vision, 20 images)** | **+2,02 % ± 1,5 % (SE), non résolu, sous réserve** — int8 conservé ; réfutation n = 60 après le feu vert (géo − 2 SE > +1 % → `kv=bf16(vision)` en 0.6.35) | Sage 21 h 14 |
| régime | `graphes=on(hybrides≤4) kv=int8 vision=bf16(eager,transformers=5.17.0) masque_images=causal mrope=[24,20,20](interleaved) deepstack=3 ctx_tenu=…` | ligne du moteur |

Défaut nommé et corrigé le 20/09 : le bloc bidirectionnel de Gemma était appliqué à tout modèle porteur d'images (model.py:593) ; Qwen3-VL est causal chez HF (`create_causal_mask`) — porte de famille par `architectures` du manifeste (`oceane-masque-famille` 8ce208eb), refus nommé `MasqueImageInconnu` hors Gemma3/4 et Qwen2/2.5/3-VL.
