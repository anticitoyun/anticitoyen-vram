# 296 — quatre workflows vidéo ComfyUI 0.37.4 pour la Pipe OWUI (poste6, 28/09, branche poste6-296)

Ordre chef (utilisateur, avec poste5) : I2V, T2V, VACE, LTX-2.3 au format API, encodeurs et VAE présents, un rendu court
réussi chacun sous carte.sh, JSON dans le dépôt. Fichiers : `parc/share/openwebui/videos/<nom>.api.json` + `<nom>.noeuds.json`
(format de poste5), test à sec `tests/test_videos_owui.py` (15), rendu `outils/videos-owui-rendu.sh` (sous carte.sh, reprenable).

## Rendu court (2 s), prise carte.sh 20:13 → 20:34, HEAD a7973746d, RTX 5090 éco 2700 — `poste6-piece296-rendu-28-09.tsv`

| graphe | durée (chargement compris) | sortie | sha256 (16) |
|---|---|---|---|
| wan22-14b-i2v | 252 s | 832×480, 33 img, 16 i/s, 360 013 o | 5475bfc7f659f42e |
| wan22-14b-t2v | 306 s | 832×480, 33 img, 302 999 o | 635c39ae2964fde1 |
| wan-vace | 406 s | 832×480, 33 img, 261 314 o | 66d1022e46a99e06 |
| ltx23 (i2v) | 274 s | 768×512, 49 img, 24 i/s + audio, 326 449 o | a4b1f184e56880db |
| ltx23-t2v | 6 s (modèles déjà chargés) | 768×512, 49 img + audio, 310 893 o | 1c1e9c82fdc2eab7 |

Les cinq vidéos sont non vides et distinctes (YAVG/SATAVG par ffmpeg signalstats sur les images 0 et 16 : i2v 170/63,
t2v 64/13 — sombre —, vace 173/53, ltx i2v 175/59, ltx t2v 104/10 — terne). Aucun jugement de qualité : la durée du
rendu est dominée par le chargement depuis /mnt/16TO (2 × 15 Go par Wan, 19,5 + 13 Go pour LTX) ; LTX à modèles chauds = 6 s.

## Choix
* Wan 2.2 A14B I2V/T2V : cœur ComfyUI, experts HIGH/LOW fp8 KJ, 20 pas partagés 10/10, cfg 3,5, shift 8, euler/simple
  (template Comfy-Org sans la LoRA lightning : `steps` réglable). **T2V HIGH fp8 KJ était déjà présent** (15 001 361 458 o,
  même taille que LOW) : rien téléchargé.
* Wan VACE : les modules `Wan2_2_Fun_VACE_module_A14B_{HIGH,LOW}_bf16` sont des ajouts aux experts T2V que le cœur ne charge
  pas → wrapper Kijai (`WanVideoModelLoader` + `WanVideoVACEModelSelect`, deux `WanVideoSampler` enchaînés 0-10 / 10-fin,
  `attention_mode=sdpa`). Référence = **vidéo de contrôle (VHS_LoadVideo.video) ET image de référence (LoadImage.image)**,
  toutes deux obligatoires dans ce graphe (le rendu a pris le mp4 de l'I2V en contrôle, example.png en référence).
* LTX-2.3 : `ltx23FP4_ltx2322BDevNVFP4.safetensors` est un **checkpoint complet** (DiT NVFP4 + VAE vidéo + VAE audio +
  projection texte, métadonnées `_quantization_metadata` ComfyUI) → `CheckpointLoaderSimple`, `LTXVAudioVAELoader` et
  `LTXAVTextEncoderLoader` lisent le même fichier ; LoRA distillée `ltx23_ltx2322bDistilled` (1,0), 8 sigmas distillés en une
  passe (pas de second étage ×2 : l'upscaler 2.3 est absent), cfg 1, audio joint (VHS écrit `<nom>-audio.mp4`).
  Seul Gemma 3 12B complet du parc : `ltx219BAllYouNeedIs_ltx2GEMMATXTENCODER` (13,2 Go ; les « gemma_3_12B_it*.safetensors »
  de Loras/ sont des LoRA de 0,2-0,5 Go).
* Deux liens symboliques HORS dépôt, à reposer sur une autre machine : `/mnt/16TO_LORAS_2025/checkpoints/LTXV2/ltx-2.3/
  ltx23FP4_ltx2322BDevNVFP4.safetensors` → `diffusion_models/LTXV 2.3/…` ; `ComfyUI_PARTAGE/models/text_encoders/ltx/
  gemma_3_12B_it_ltx2.safetensors` → `diffusion_models/LTXV/ltx219BAllYouNeedIs_ltx2GEMMATXTENCODER.safetensors`.
* Validation à sec avant la carte : `execution.validate_prompt` de ComfyUI avec `PromptServer(loop, default_asset_manager())`
  et CUDA masqué (sinon les nœuds VHS/wrapper n'importent pas) : 4/5 VALIDE, wan-vace refusé seulement pour son mp4 absent.

## Défauts pour la Pipe (poste5)
| nom | i/s | longueur | résolution × durée par défaut | entrées |
|---|---|---|---|---|
| wan22-14b-i2v | 16 | 4k+1 | 832×480 × 81 (5 s), 20 pas | prompt, negatif, width, height, length, fps, seed, steps, image |
| wan22-14b-t2v | 16 | 4k+1 | 832×480 × 81, 20 pas | idem sans image |
| wan-vace | 16 | 4k+1 | 832×480 × 81, 20 pas | + video (contrôle) ET image (référence) ; length et fps sur 2 nœuds |
| ltx23 / ltx23-t2v | 24 | 8k+1 | 1024×576 × 121 (5 s) ; pas fixes (sigmas) | prompt, negatif, width, height, length (2 nœuds), fps (3 nœuds), seed, image |

Restes : rendu par défaut (5 s, 1024×576 LTX) non chronométré ; qualité non jugée (t2v sombre, ltx t2v terne au premier
essai, une graine) ; VACE sans masque ; lightning 4 pas possible plus tard (LoRA LOW T2V absente).
