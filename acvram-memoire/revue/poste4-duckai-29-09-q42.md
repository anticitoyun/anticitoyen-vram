# Q42 (poste5/poste6, vidéo) — Wan 2.2 14B / LTX-2.3 dans ComfyUI, RTX 5090 32 Go

**duck.ai INACCESSIBLE cette séance** : extension Chrome non connectée (`tabs_context_mcp` →
« Browser extension is not connected »). Pas de croisement 3 modèles ici — sources primaires
seules ci-dessous. À refaire quand l'extension répond.

## fp8 vs GGUF
- GGUF Q8_0 ≈ indiscernable du FP8 scaled à 480-720p pour l'usage courant. La chute de qualité
  est entre Q5_K_M et Q4_K_M, pas entre Q8_0 et Q5_K_M.
  [Wan 2.2 GGUF Guide](https://wan27.org/blog/wan-2-2-gguf-guide)

## LoRA de distillation (4-8 pas)
- Modèles distillés Wan2.2 : 4 pas au lieu de 40-50, jusqu'à ~20× sur mouvement complexe seul ;
  combiné LightX2V + step/cfg-distill + FP8 : **jusqu'à ~42×** revendiqué.
  [ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning) ·
  [Wan2.2 Distilled Models (HF)](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) ·
  [Step Distillation docs](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html)
- FP8/INT8 distillés : ~50 % de taille en moins, tournent sur RTX 4060 (a fortiori 5090).
- CausVid LoRA v2 (Wan 2.1) : 8 pas, gain qualité couleurs/saturation revendiqué — pas de
  chiffre de temps trouvé, à vérifier séparément pour Wan 2.2.
  [dev.to CausVid v2](https://dev.to/furkangozukara/causvid-lora-v2-of-wan-21-brings-massive-quality-improvements-better-colors-and-saturation-12g8)
- Un avis communautaire (2025, à prendre avec réserve, pas de banc) : LightX2V donnerait de
  meilleurs résultats sur Wan2.1 que sur Wan2.2.

## LTX-2.3 sur RTX 5090
- Pipeline distillé 2 étages : clip 97 images (~4 s) + audio synchro en **~40 s à 768×512**,
  **~50 s à 1280×704**, une seule RTX 5090.
  [LTX-2.3 vs Wan2.2 5090 benchmark](https://zenn.dev/toki_mwc/articles/ltx23-vs-wan22-i2v-benchmark-rtx5090?locale=en) ·
  [rtx-5090-benchmarks/ltx-2.3.md (HF dataset)](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ltx-2.3.md)
- **SageAttention3 (Blackwell sm120) : AUCUN gain mesuré sur LTX-2.3** — temps identique avant/
  après migration Python 3.13 + sageattn3, contrairement à Wan 2.2 qui en profite. Pas d'explication
  sourcée, hypothèse : implémentation d'attention de LTX-2.3 déjà optimisée différemment.
- torch.compile / TeaCache : pas de chiffre combiné trouvé pour LTX-2.3+5090 dans cette recherche ;
  seule mention générique de FBCache (inspiré TeaCache) utilisable avec torch.compile côté ComfyUI
  (nœud Comfy-WaveSpeed), sans banc chiffré associé.

## RESTE
- Chiffres torch.compile/TeaCache combinés sur LTX-2.3+5090 : non trouvés, à chercher côté
  ComfyUI-WaveSpeed / discussions Comfy-Org si le groupe en a besoin.
- Croisement 3 modèles duck.ai à refaire (extension Chrome à reconnecter).
