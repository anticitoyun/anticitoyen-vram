# Comparatif à cinq moteurs — Coder-30B et GLM-4.7-Flash (17/09)

**Note datée 17/09 soir (`poste7-poste-d-verdict-17-09`)** : budget KV dimensionné pour 8 séquences (`loader.py:1459`) tronquait 4/12 séquences dans toute cellule acvram b>8 mesurée avant ce correctif. Coder et GLM édités avec les valeurs correctes (bogue lui-même : chantier séparé, non corrigé sur main à cette date — anciennes valeurs 730/514-540 t/s, étiquetées « 4/12 tronquées », conservées en historique dans les verdicts nommés).

**Note datée 17/09 soir (`poste7-b0-et-cause-lm4-17-09`)** : prefill GEMM groupée passée défaut `ACVRAM_PREFILL_GROUPED=groupe` (0.6.6, Triton persistant, B0) — Coder 8 633→9 913 j/s (+14,8%), GLM 4 462→4 661 j/s (+4,5%, sous la prédiction 5 000 mais aucune régression, PPL±0,002 tenue). Ancien défaut `grouped_mm` gardé comme témoin. B1 (lecture NVFP4 dans la tuile) scellé Coder≥15 000, GLM≥6 000.

**Note datée 17/09 soir, étiquette 0.6.8-E+C (`poste7-e-c-verdict-17-09`, `verdict-coder-c-mixte-17-09`)** : décodage édité — `ACVRAM_PAGED_ATTN=triton` (E) ET `ACVRAM_NARROW_KERNEL=mixte` (C, dès b≥2 ; le plantage initial sous capture était un débordement mémoire réel sur les tranches K, corrigé) passés défaut. Coder b=1 233,9→257,8 t/s (+10,2%, J −11%), b=12 743,4→997,8 t/s (+34%, J −26%) — table éditée avec les valeurs 0.6.8, anciennes valeurs (233,9/743,4) conservées ici en historique, pas remesurées ailleurs (GLM, autres moteurs).

Assemblé par chef, critère 6 points `poste7-plan-completion-comparatif-17-09`. Corpus privé scellé `5909d27` (jamais publié en clair) ; corpus public = échantillon documenté dans chaque verdict. Préfixe `[gMASK]<sop>` obligatoire sur toute mesure GLM. Tout acvram mesuré sur main ≥ `0971c90` (défaut `ACVRAM_PREFILL=bf16`), régime porté par `regime_ligne()` dans chaque instrument.

## Table Coder-30B-A3B (`Qwen3-Coder-30B-A3B`)

| moteur / régime | PPL privé | PPL public | b=1 t/s | b=1 J | b=12 t/s | b=12 J | prefill j/s | classé |
|---|---|---|---|---|---|---|---|---|
| acvram W4A16 (nvfp4, bf16 prefill) | 1,0148 géo (302025e) | 1,0099 | **366,4** | **0,907** | **1 198** | **0,334** | **9 913** | oui |
| EXL3 4,25 bpw (exllamav3, TabbyAPI) | 1,0006 | 1,0004 | 168,0 | 1,546 | 855,7 | 0,362 | 10 656 | oui |
| llama.cpp Q4_K_M | 1,0103 | 1,0146 | 344,0 | 1,151 | 971–1 020 | 0,300–0,306 | 15 717 | oui |
| vLLM W4A16 Marlin (ModelOpt communautaire) | 1,1227 | 1,0876 | 302,9 | — | 2 031,4 | 0,197 | 20 988 | non |
| vLLM W4A4 CUTLASS (ModelOpt communautaire) | 1,1554 | 1,1160 | 200,9 | — | 1 621,7 | 0,227 | 35 241 | non |
| TRT-LLM W4A4 (même ModelOpt, KV fp8) | 1,2313 | 1,1376 | 234,9 | 1,481 | 2 104,7 | 0,177 | 55 419 | non |

**Statut vLLM/TRT-LLM Coder : ModelOpt propre = refus accepté**, deux blocages fichier:ligne indépendants et documentés (`unified_export_hf.py:419-422` Qwen3MoeExperts non supporté transformers 5 ; `nvfp4_tensor.py:84` déport CPU/GPU casse le calcul NVFP4 sous transformers 4, modèle > 32 Gio VRAM). Cellule remplie par le checkpoint communautaire (1,16-1,30), non classée — scellé du protocole réfuté : le checkpoint porte ~12 % de la perte, les activations 4 bits ~3 %, Marlin bat CUTLASS en décodage.

**Revendication Coder (mise à jour 18/09 harnais égal, `poste7-harnais-egal-verdict-18-09`)** : acvram devant à b=1 ET b=12 au harnais égal (CUDA_VISIBLE_DEVICES=0, régime identique instrument/contexte). b=1 : 366,4 t/s · 0,907 J contre llama.cpp 344,0 · 1,151 J (+6,5% t/s, −21% J). b=12 : 1 198 t/s · 0,334 J contre llama.cpp 971–1 020 · 0,300–0,306 (+5,7% t/s, +25% J brut/+37% net). Ancienne revendication « llama.cpp +19% à b=1 » comparait régimes inégaux (instrument, contexte, cartes) — retirée. Qualité identique (1,0148 géo classé, 302025e). llama.cpp reste classé sur cet écart de b=12 énergie (+37% net) ; poste ouvert pour optimisation acvram b=12. vLLM/TRT-LLM ne sont pas comparables sur le privé (checkpoint non classé).

## Table GLM-4.7-Flash (poids GadflyII, sauf mention)

| moteur / régime | PPL privé | PPL public | b=12 t/s | b=12 J | prefill j/s | classé |
|---|---|---|---|---|---|---|
| acvram W4A16 bf16 prefill (`-vllm-direct`) | 1,0096 | — | — | — | — | oui |
| acvram W4A16 bf16 prefill (`-k48-calibA`) | 1,0143 | 1,0281 | — | — | **4 661** | oui |
| vLLM Marlin W4A16 | 1,0164 | 1,0133 | 858 | 0,397 | 18 117 | oui |
| llama.cpp Q4_K_M (unsloth, imatrix) | 1,0248 | 1,0397 | 702 | 0,495 | 11 293 | non (> 1,02) |
| vLLM W4A4 (défaut) | 1,0717 | 1,0751 | 796 | 0,445 | 26 732 | non |
| acvram W4A16 a8 (ancien défaut, historique) | 1,0280 | 1,0213 | 514* | 0,774* | ~3 681-4 453 | non |

*ligne `-k48` sous ancien défaut `a8`, conservée pour mémoire du chantier clos, pas pour classement.
b=12 GLM `-k48-calibA` bf16 (converti retenu, budget KV correct CERT_PLAN_LEN=3072) : **568,1 t/s, 0,700 J** (verdict-glm-b12-plan-17-09, poste3, 6d2c5fb, aucune troncature, 21 444 jetons). Fourchette 548-560 réfutée par le haut : la traîne de contexte coûte +5,3 % à GLM contre +1,8 % à Coder.

b=1 acquis séparément : vLLM Marlin W4A16 **183,5 t/s** (1,705 J brut / 1,329 net) ; vLLM W4A4 153,7 t/s. b=1 acvram GLM non mesuré isolément dans cette campagne (non bloquant : b=12 et prefill suffisent au classement qualité/vitesse demandé).

**Statuts non servis** : TabbyAPI EXL3 GLM (MLA non supporté par exllamav3, blocage structurel) ; TRT-LLM GLM (refus rc15, `model_config.py:438-482`).

**Revendication GLM** : sous le défaut `bf16` désormais livré (poste4, `0971c90`), acvram est le seul moteur classé qui bat vLLM Marlin sur la qualité (1,0096-1,0143 contre 1,0164) ; llama.cpp Q4_K_M dépasse le seuil de classement (1,0248 > 1,02) sur ce corpus privé. vLLM Marlin reste le plus rapide et le plus économe parmi les classés (858 t/s, 0,397 J, 18 117 j/s prefill) ; acvram n'a pas de mesure de vitesse b=12 comparable sous son régime `bf16` actuel dans cette campagne — vitesse acvram GLM non revendiquée au-delà du prefill `-k48-calibA` (4 464 j/s).

## Chantier clos associé

« 1 % W4A16 GLM » clos et confirmé : cause = expert partagé en W8A8 caché sous l'ancien défaut `a8` (`fp4_gemm.py:257`, `nvfp4_mm_w4a8`). Témoin dense `Qwen2.5-Coder-14B-nvfp4` (337 projections) : `bf16` gagne qualité (−1,4 % PPL) ET vitesse (+56 % prefill j/s) contre `w8a8` — chantier GEMM fusionnée sans objet (aucune perte de j/s à compenser). Nouveau défaut `ACVRAM_PREFILL=bf16` livré, 48/55 convertis affectés (réétiquetage INDEX, pas remesure).

## Sources

`verdict-coder-acvram-w4a16-17-09.md`, `verdict-coder-exl3-vitesses-17-09.md`, `verdict-coder-vllm-17-09.md`, `verdict-coder-trtllm-17-09.md`, `verdict-modelopt-coder-17-09.md`, `verdict-modelopt-coder-tf4-17-09.md`, `verdict-temoin-dense-w8a8-17-09.md`, `verdict-prefill-bf16-17-09.md`, `verdict-bissection-w4a16-17-09.md`, `verdict-bissection-w4a16-suite-17-09.md`, `poste7-plan-completion-comparatif-17-09.md`.
