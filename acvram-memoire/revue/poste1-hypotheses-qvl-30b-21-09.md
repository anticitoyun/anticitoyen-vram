# Qwen3-VL-30B « nvfp4-vision » : TTFT 4,28 s, J ×4,4, P3 (3) 21,8 % — trois hypothèses à sec, une établie (poste1, 21/09)

instrument : `jq` sur `acvram_manifest.json` de l alias + journal `serveur-duel-30b.log` (arbre 665eeacc, worktree poste2-w-21-09), 0 min de carte
commit : 665eeacc (verdicts `verdict-p3-3-30b-665eeacc-21-09`, `verdict-p3-4-30b-665eeacc-21-09`)

## H1 — ÉTABLIE à sec : l alias n est pas nvfp4, c est un int4_awq planifié pour le processeur
* manifeste `Qwen3-VL-30B-A3B-abl-nvfp4-vision/acvram_manifest.json` : `tensors.*.format` = **int4_awq × 18 625** (bpw 4,156, dont `lm_head.weight`), bf16 × 593 (tour, normes, routeur) ; **aucun tenseur nvfp4** ; `plan.tiers` = `[{name: cpu, kind: host, weight_format: int4_awq}]` — pas de palier GPU ; `options.format_impose` null, `group_size` 128, calibration `calibration-anglais.txt` 32 × 16 384.
* cause : `memory/tiering.py:360-366` `build_tiers` n émet un palier que par GPU visible ; sans carte à la conversion (`CUDA_VISIBLE_DEVICES=""`), `head_fmt`/format retombent sur `"int4_awq"` (tiering.py:397, 514, 542) — la conversion « à sec » du 30B (20/09) a produit le format hôte, sous un nom qui dit nvfp4.
* conséquences lues dans le journal : ligne 6 « disposition Marlin refusée : piles non NVFP4 — pile naturelle gardée, prefill « groupe », décodage d avant » (`engine/model.py:1169`) → experts int4_awq sur le chemin générique (pas de Marlin, pas de mma-a4 malgré `chemin_moe=mma-a4` sur la ligne, qui décrit la capacité et non ce que ces piles ont pris) → **TTFT 4,28 s et J ×4,4** ; **P3 (3) 21,8 %** = double quantification AWQ g32 → bf16 → int4_awq g128 avec une calibration anglaise étrangère, contre une référence qui est l AWQ déquantifié lui-même.
* mesure qui tranche (poste2, ≈ 15 min de carte) : reconvertir le 30B **carte visible** (`outils/carte.sh env CUDA_VISIBLE_DEVICES=0 acvram convert <AWQ déquantifié ou AWQ> …`, `--echelle=4sur6` si Q1 est voulu ici), vérifier `jq '.plan.tiers[].weight_format'` = nvfp4 et `tensors` nvfp4 AVANT toute prise, puis rejouer P3 (4) et P3 (3). Prédictions : journal sans « piles non NVFP4 », `chemin_moe=mma-a4` effectif ; TTFT ≤ 0,3 s (tour 784 jetons + prefill 800 jetons nvfp4) — le scellé 0,094 s peut rester réfuté par la tour eager transformers, à dire ; J ≤ 60 ; P3 (3) haut_2se ≤ 3 % — issue qui me gênerait : > 3 % parce que nvfp4-de-l AWQ reste une seconde quantification (le plafond est alors la source AWQ, à nommer sur la fiche), réfuté si > 3 %.
* garde à écrire (une fonction, un test, après ce verdict) : `acvram convert` refuse (rc 2) un nom de sortie qui contient `nvfp4` quand aucun palier GPU n existe dans le plan — le nom ne peut plus mentir sur le format.

## H2 — tour de vision (appareil, dtype) : peu probable, à exclure après H1
* `engine/runner.py:506` : la tour est matérialisée sur `self.model.embed_tokens.device` ; `couches_exilées=0/48` → cuda:0 ; `vision.py:185-236` : sous-modules `visual` seuls, bf16, `get_image_features` ; journal : 21 lignes « tour : [13,797) (784, 2048) sha=c0371e35 Σ=3.153e+05 » — même Σ à chaque requête, tour déterministe ; tous les tenseurs `model.visual.*` sont bf16 (351).
* mesure (≤ 2 min) : après reconversion, P3 (4) rejoué ; si TTFT reste > 1 s, chronométrer la tour seule (`TourVision.depuis_dossier` + `traits_niveaux` sur une image, événements CUDA) : attendu ≤ 60 ms.

## H3 — placement des jetons image / M-RoPE / deepstack : même code que le 2B, à contrôler sur le 2B
* `runner.py:903-907` positions M-RoPE avant la tour, `model.py:3189-3205` `disperser_images` (intersection par morceau), `model.py:3221-3250` `ajouter_deepstack` sur (x + delta) ; masque causal par famille (`masque_images=causal`).
* mesure (≤ 5 min, carte) : `SEC=0` mais alias 2B — `chaine-p3-3.sh` sur `Qwen3-VL-2B-Instruct-bf16-vision` contre `ref-2b-fp32.pt` (déjà scellée) : attendu haut_2se ≤ 3 % (le 2B bf16 servi contre sa référence fp32) ; s il est réfuté, H3 est en cause indépendamment du format.

durée : 0 min de carte (lecture du manifeste et du journal) · verdict : H1 établie sur pièces, H2/H3 à exclure après la reconversion
