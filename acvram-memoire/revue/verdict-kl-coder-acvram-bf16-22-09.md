# KL acvram/bf16 Coder (bras acvram), decode-pas texte — TENU 4/5, invite4 ÉCHEC instrument — 22/09 (poste2)

* instrument : `scratchpad/mm-diag-20-09/decode-pas-texte.py` (poste7, vision à l'origine), 4 correctifs appliqués pour le texte seul : CONTENU en chaîne (pas liste, Jinja du gabarit Coder le refusait), `AutoModelForCausalLM` au lieu de `AutoModelForImageTextToText` (Coder = `Qwen3MoeConfig`, pas une config vision), `force(**kwargs)` (pipeline.py passe `depuis_graphe=`, même bug que `mesure-c.py:32` déjà corrigé ailleurs), mode batch (`DECODE_PAS_TEXTES_LISTE`) : un seul chargement HF bf16 offload pour les 5 invites au lieu d'un par invite (694 s/invite × 5 ≈ 58 min → charge unique 595 s + 5 forwards ≈ 22 s chacun)
* commit : main/poste2 à jour (68e0e638)
* régime : alias acvram `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`, référence `Qwen3-Coder-30B-A3B-Instruct-bf16-hub` (offload MAXMEM=26GiB,80GiB), 5 invites `scratchpad/trtllm-cellules-22-09/invites-kl-texte.txt`, 8 pas gloutons chacune
* scellé (groupe, 22/09) : KL max ≤ 1,0 par invite — **distinct** du champ `verdict` du script (BRUIT/ÉGAL/DÉFAUT, critère poste7 `|Δ logit|≤2` sur les 8 pas, plus strict et non demandé ici — invite0 seule tombe déjà en DÉFAUT sur ce critère malgré KL négligeable, à ignorer pour ce test)
* mesuré :

| invite | kl_max | pas égaux (top-1 acvram==HF) |
|---|---|---|
| invite0 | 0,00807 | 8/8 |
| invite1 | 0,08894 | 8/8 |
| invite2 | 0,51934 | 7/8 |
| invite3 | 0,12542 | 7/8 |
| invite4 | **ÉCHEC instrument** | — |

* verdict : **TENU sur 4/5** — kl_max max sur les 4 invites valides = 0,519 (invite2), très en dessous du seuil 1,0. **invite4 en ÉCHEC nommé** : `hf_pas_batch` plante sur `assert lg.argmax(-1).tolist() == cibles` (teacher forcing HF ≠ generate) — un désaccord réel entre la génération gloutonne de HF et le forward en teacher-forcing pour cette invite précise, cause non diagnostiquée (hors budget de cette pièce ; les 4 dumps précédents avaient réussi le même test, donc pas un défaut systémique du harnais). Dump invite4 absent (le crash survient avant l'écriture).
* durée : ~22 min (chargement HF unique 595 s + 4×22 s forwards + acvram ~1 min pour les 4 invites, sous carte.sh)

## Suite
Dumps HF bf16 (format `{ids, cibles, logits}`, 8×vocab chacun) dans `scratchpad/kl-coder-texte-22-09/dumps/lot-hf/decode-pas-hf-invite{0,1,2,3}.txt.pt` (dans le worktree `anticitoyen-vram`, pas committé — dossier de sortie, pas du code) — chemin transmis à poste3 pour KL(bf16‖trtllm) sans refaire ce bras. invite4 à reprendre séparément si le groupe le juge utile (cause de l'assertion à diagnostiquer d'abord). `decode-pas-texte.py` (4 correctifs) committé avec ce verdict, tracké par git.
