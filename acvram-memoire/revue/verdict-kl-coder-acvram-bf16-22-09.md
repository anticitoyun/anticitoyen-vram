# KL acvram/bf16 Coder (bras acvram), decode-pas texte — TENU 5/5 — 22/09 (Manon)

* instrument : `scratchpad/mm-diag-20-09/decode-pas-texte.py` (Sage, vision à l'origine), 4 correctifs appliqués pour le texte seul : CONTENU en chaîne (pas liste, Jinja du gabarit Coder le refusait), `AutoModelForCausalLM` au lieu de `AutoModelForImageTextToText` (Coder = `Qwen3MoeConfig`, pas une config vision), `force(**kwargs)` (pipeline.py passe `depuis_graphe=`, même bug que `mesure-c.py:32` déjà corrigé ailleurs), mode batch (`DECODE_PAS_TEXTES_LISTE`) : un seul chargement HF bf16 offload pour les 5 invites au lieu d'un par invite (694 s/invite × 5 ≈ 58 min → charge unique 595 s + 5 forwards ≈ 22 s chacun)
* commit : main/manon à jour (68e0e638)
* régime : alias acvram `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`, référence `Qwen3-Coder-30B-A3B-Instruct-bf16-hub` (offload MAXMEM=26GiB,80GiB), 5 invites `scratchpad/trtllm-cellules-22-09/invites-kl-texte.txt`, 8 pas gloutons chacune
* scellé (groupe, 22/09) : KL max ≤ 1,0 par invite — **distinct** du champ `verdict` du script (BRUIT/ÉGAL/DÉFAUT, critère Sage `|Δ logit|≤2` sur les 8 pas, plus strict et non demandé ici — invite0 seule tombe déjà en DÉFAUT sur ce critère malgré KL négligeable, à ignorer pour ce test)
* mesuré :

| invite | kl_max | pas égaux (top-1 acvram==HF) |
|---|---|---|
| invite0 | 0,00807 | 8/8 |
| invite1 | 0,08894 | 8/8 |
| invite2 | 0,51934 | 7/8 |
| invite3 | 0,12542 | 7/8 |
| invite4 | 0,23262 | 7/8 |

* verdict : **TENU sur 5/5** — kl_max max sur les 5 invites = 0,519 (invite2), très en dessous du seuil 1,0. invite4 rejouée seule après le correctif `repetition_penalty=1.0` : plus de crash, pas 0 divergent (marge top-2 HF serrée, 0,875 nats — near-tie, pas un défaut), 7/8 pas égaux.
* durée : ~22 min (4 invites en lot) + ~12 min (invite4 seule, rejeu après correctif)

## Suite
Dumps HF bf16 (format `{ids, cibles, logits}`, 8×vocab chacun) dans `scratchpad/kl-coder-texte-22-09/dumps/lot-hf/decode-pas-hf-invite{0,1,2,3}.txt.pt` + `dumps/decode-pas-hf-invite4.txt.pt` (chemin single, dans le worktree `anticitoyen-vram`, pas committé — dossier de sortie, pas du code) — chemins transmis à Laure pour KL(bf16‖trtllm) sans refaire ce bras. `decode-pas-texte.py` (5 correctifs) committé avec ce verdict, tracké par git.

**invite4 : diagnostiquée à sec puis rejouée avec succès.** Cause : `generation_config.json` du modèle porte `repetition_penalty: 1,05` par défaut ; `hf.generate()` l'appliquait (non surchargé), le forward manuel en teacher-forcing calculait des logits bruts sans pénalité — un jeton répété dans les 8 pas suffisait à faire diverger `cibles` (généré, pénalisé) de `argmax(logits bruts)`. Corrigé (`repetition_penalty=1.0` explicite), rejouée : kl_max=0,23262, TENU.
