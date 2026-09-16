# Comparatif à cinq moteurs — Coder-30B et GLM-4.7-Flash (17/09)

Assemblé par chef, critère 6 points `poste7-plan-completion-comparatif-17-09`. Corpus privé scellé `5909d27` (jamais publié en clair) ; corpus public = échantillon documenté dans chaque verdict. Préfixe `[gMASK]<sop>` obligatoire sur toute mesure GLM. Tout acvram mesuré sur main ≥ `0971c90` (défaut `ACVRAM_PREFILL=bf16`), régime porté par `regime_ligne()` dans chaque instrument.

## Table Coder-30B-A3B (`Qwen3-Coder-30B-A3B`)

| moteur / régime | PPL privé | PPL public | b=1 t/s | b=1 J | b=12 t/s | b=12 J | prefill j/s | classé |
|---|---|---|---|---|---|---|---|---|
| acvram W4A16 (nvfp4, bf16 prefill) | 1,0144 | 1,0099 | 233,9 | 1,365 | 730,4 | 0,546 | 8 633 | oui |
| EXL3 4,25 bpw (exllamav3, TabbyAPI) | 1,0006 | 1,0004 | 168,0 | 1,546 | 855,7 | 0,362 | 10 656 | oui |
| llama.cpp Q4_K_M | 1,0103 | 1,0146 | 341,4 | 1,108 | 709,2 | 0,523 | 15 717 | oui |
| vLLM W4A16 Marlin (ModelOpt communautaire) | 1,1227 | 1,0876 | 302,9 | — | 2 031,4 | 0,197 | 20 988 | non |
| vLLM W4A4 CUTLASS (ModelOpt communautaire) | 1,1554 | 1,1160 | 200,9 | — | 1 621,7 | 0,227 | 35 241 | non |
| TRT-LLM W4A4 (même ModelOpt, KV fp8) | 1,2313 | 1,1376 | 234,9 | 1,481 | 2 104,7 | 0,177 | 55 419 | non |

**Statut vLLM/TRT-LLM Coder : ModelOpt propre = refus accepté**, deux blocages fichier:ligne indépendants et documentés (`unified_export_hf.py:419-422` Qwen3MoeExperts non supporté transformers 5 ; `nvfp4_tensor.py:84` déport CPU/GPU casse le calcul NVFP4 sous transformers 4, modèle > 32 Gio VRAM). Cellule remplie par le checkpoint communautaire (1,16-1,30), non classée — scellé du protocole réfuté : le checkpoint porte ~12 % de la perte, les activations 4 bits ~3 %, Marlin bat CUTLASS en décodage.

**Revendication Coder** : acvram est le seul moteur classé **et** compétitif en b=12 sur ce jeu (730 t/s, 0,546 J) ; EXL3 gagne la qualité pure (1,0006) et l'énergie b=12 (0,362 J) mais perd b=1 (168 t/s, le plus lent des classés) ; llama.cpp reste le plus rapide en b=1 (341 t/s) parmi les classés. vLLM/TRT-LLM ne sont pas comparables sur le privé (checkpoint non classé) — seule leur vitesse brute (2 000-2 100 t/s b=12) est acquise, hors classement qualité.

## Table GLM-4.7-Flash (poids GadflyII, sauf mention)

| moteur / régime | PPL privé | PPL public | b=12 t/s | b=12 J | prefill j/s | classé |
|---|---|---|---|---|---|---|
| acvram W4A16 bf16 prefill (`-vllm-direct`) | 1,0096 | — | — | — | — | oui |
| acvram W4A16 bf16 prefill (`-k48-calibA`) | 1,0143 | 1,0281 | — | — | 4 464 | oui |
| vLLM Marlin W4A16 | 1,0164 | 1,0133 | 858 | 0,397 | 18 117 | oui |
| llama.cpp Q4_K_M (unsloth, imatrix) | 1,0248 | 1,0397 | 702 | 0,495 | 11 293 | non (> 1,02) |
| vLLM W4A4 (défaut) | 1,0717 | 1,0751 | 796 | 0,445 | 26 732 | non |
| acvram W4A16 a8 (ancien défaut, historique) | 1,0280 | 1,0213 | 514* | 0,774* | ~3 681-4 453 | non |

*ligne `-k48` sous ancien défaut `a8`, conservée pour mémoire du chantier clos, pas pour classement.
b=12 GLM `-k48-calibA` bf16 (converti retenu) : **539,5 t/s, 0,732 J** (verdict-profil-coder-pas-17-09, poste3) — écart avec le 514 t/s du 17/09 matin dû au converti calibA (+3,5 %), pas au régime `ACVRAM_PREFILL` (n'affecte pas le décodage).

b=1 acquis séparément : vLLM Marlin W4A16 **183,5 t/s** (1,705 J brut / 1,329 net) ; vLLM W4A4 153,7 t/s. b=1 acvram GLM non mesuré isolément dans cette campagne (non bloquant : b=12 et prefill suffisent au classement qualité/vitesse demandé).

**Statuts non servis** : TabbyAPI EXL3 GLM (MLA non supporté par exllamav3, blocage structurel) ; TRT-LLM GLM (refus rc15, `model_config.py:438-482`).

**Revendication GLM** : sous le défaut `bf16` désormais livré (poste4, `0971c90`), acvram est le seul moteur classé qui bat vLLM Marlin sur la qualité (1,0096-1,0143 contre 1,0164) ; llama.cpp Q4_K_M dépasse le seuil de classement (1,0248 > 1,02) sur ce corpus privé. vLLM Marlin reste le plus rapide et le plus économe parmi les classés (858 t/s, 0,397 J, 18 117 j/s prefill) ; acvram n'a pas de mesure de vitesse b=12 comparable sous son régime `bf16` actuel dans cette campagne — vitesse acvram GLM non revendiquée au-delà du prefill `-k48-calibA` (4 464 j/s).

## Chantier clos associé

« 1 % W4A16 GLM » clos et confirmé : cause = expert partagé en W8A8 caché sous l'ancien défaut `a8` (`fp4_gemm.py:257`, `nvfp4_mm_w4a8`). Témoin dense `Qwen2.5-Coder-14B-nvfp4` (337 projections) : `bf16` gagne qualité (−1,4 % PPL) ET vitesse (+56 % prefill j/s) contre `w8a8` — chantier GEMM fusionnée sans objet (aucune perte de j/s à compenser). Nouveau défaut `ACVRAM_PREFILL=bf16` livré, 48/55 convertis affectés (réétiquetage INDEX, pas remesure).

## Sources

`verdict-coder-acvram-w4a16-17-09.md`, `verdict-coder-exl3-vitesses-17-09.md`, `verdict-coder-vllm-17-09.md`, `verdict-coder-trtllm-17-09.md`, `verdict-modelopt-coder-17-09.md`, `verdict-modelopt-coder-tf4-17-09.md`, `verdict-temoin-dense-w8a8-17-09.md`, `verdict-prefill-bf16-17-09.md`, `verdict-bissection-w4a16-17-09.md`, `verdict-bissection-w4a16-suite-17-09.md`, `poste7-plan-completion-comparatif-17-09.md`.
