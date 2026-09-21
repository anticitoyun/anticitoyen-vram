# C9 — Mistral-Small-4-119B-2603 : conception à sec (21/09, poste1) — pièce 19, OUI utilisateur 17 h

* sources : en-têtes safetensors du NVFP4 officiel (`/mnt/2TO_2023_980PRO/Modeles/originaux/Mistral-Small-4-119B-2603-NVFP4`, 13 shards, 70,8 Go lus en-tête par en-tête ce soir), `params.json` ; `verdict-c9-m0-19-09` (PCIe H2D épinglé **22,6 Go/s**, 18,7 = paginable) ; `verdict-c9-llamacpp-119b-19-09` (barre **24,2 j/s** b=1, experts en RAM) ; `releve-3080ti-services-13-09` (DDR5 **71 Go/s**, expert exilé 0,30 ms/couche Coder = calcul hôte à ≈ 70 Go/s) ; chantiers `chantier-c9-m1`, `chantier-c9-m3` (branche `poste1-c9-m3-cache-experts` 009bc039 : `memory/cache_experts.py` 414 l + 282 l de tests, créneaux + tables d adresses, une couche) ; Vibe R9 lu en diagonale (rien de chiffré au-delà de l arithmétique de poste7).
* modèle (lu) : 36 couches MLA (q_lora 1024, kv_lora 256, nope 64 + rope 64, 32 têtes, d 4096), MoE 128 experts top-4 + 1 expert partagé, `expert_hidden_dim` 2048, vocab 131 072, ctx 1 M (`llama_4_scaling` β 0,1, orig 8 192 — **inconnu à vérifier** contre mistral-inference avant tout jeton), tour Pixtral (alias texte seul d abord).
* volumes (octets des en-têtes, pas d une note) : **un expert = 14,16 Mo** (3 × 4,19 Mo packed + 1,57 Mo d échelles E4M3 g16) · 4 608 experts = **65,2 Go** · résidents hors experts **4,66 Go** (attention bf16 2,02 — wo 1,21, wq_a/wq_b 0,30 + 0,30, wkv 0,21 ; expert partagé 0,45 ; routeurs 0,04 ; embeddings 1,07 + tête 1,07) · KV MLA 23 Ko/jeton (0,75 Go à 32 k).
* par jeton décodé : 4 × 36 = **144 experts = 2,04 Go** de poids à lire, où qu ils soient.

## 1. Paliers de placement
| palier | où | quoi | octets |
|---|---|---|---|
| P0 résident | VRAM | attention, normes, routeurs, expert partagé, embeddings, tête, KV, graphes | 4,66 + 0,75 + ~2 (contexte, activations, graphes) ≈ **7,4 Go** |
| P1 chaud | VRAM | experts en cache : **26,8 Go → 1 895 experts = 41 % = C ≈ 52 par couche** ; politique `pin` (fréquence apprise sur trace, `taux_de_succes_pin`) ou LRU (`cache_experts.py` créneaux) | 26,8 Go |
| P2 froid | RAM épinglée (128 Gio : tout tient, 65 Go) | soit **transféré** par PCIe au moment du routage (0,63 ms l expert à 22,6 Go/s), soit **calculé sur l hôte** (`nvfp4_matmul_cpu`, 0,20 ms l expert à 71 Go/s ; activations 8 Ko aller-retour) | 65,2 Go |

## 2. Bornes de débit b=1 (ms/jeton = 144 × 14,16 Mo × (1 − h) ÷ débit + part carte)
Part carte hors experts froids ≈ **3 ms** (tête bf16 1,07 Go 0,6 ms + attention 2 Go 1,1 ms + partagé 0,25 + experts chauds ≤ 1 Go 0,6) ; chemin hôte : + **0,1 ms de synchronisation par couche** (3,6 ms) ; **sans graphes CUDA (obligatoire au premier run, le chemin hôte casse la capture) : + 8-10 ms de lancements eager**, non compté ci-dessous, nommé § 5.
| stratégie | h | ms/jeton | j/s | contre la barre 24,2 |
|---|---|---|---|---|
| S0 tout PCIe (aucun cache) | 0 | 90 + 3 | **10,7** | perd |
| S1 cache PCIe, C = 52/couche, routage uniforme | 0,41 | 53 + 3 | **17,8** | perd |
| S1 cache PCIe, routage concentré | 0,70 | 27 + 3 | 33 | gagne seulement si h ≥ 0,55 (parité) — le h de poste7 |
| S2 tout hôte (llama.cpp-like, carte = attention) | 0 | 28,7 + 3 + 3,6 | **28** | ≈ parité (llama.cpp lit à 44 Go/s, nous 71 mesurés : +16 %) |
| **S3 mixte : chauds VRAM, froids calculés hôte** | 0,41 | 16,9 + 3 + 3,6 | **43** | gagne ×1,8 |
| S3 mixte, routage concentré | 0,70 | 8,6 + 3 + 3,6 | 66 | gagne ×2,7 |
Le terme décisif : un expert froid coûte **0,63 ms transféré contre 0,20 ms calculé** — le calcul hôte bat PCIe × 8 d un facteur 3,2 tant que h < 0,9. **C9 tel que conçu (cache PCIe) ne vaut la peine que si h ≥ 0,55 ; S3 vaut la peine dès h = 0 si le noyau hôte tient 71 Go/s sur ce modèle.** h supposé nommé : **0,45 ± 0,10** (C = 41 %, routage réel un peu concentré ; Coder synthétique donnait 0,89 à 80 % de concentration, aucun chiffre réel).

## 3. Mesures qui invalident les bornes, avant tout code (poste2)
1. **M-HÔTE (≤ 5 min, carte tenue mais quasi vide)** : `nvfp4_matmul_cpu` sur 4 experts réels du 119B (w1/w3 [2048, 2048] packed, w2 [4096, 1024], lus des shards, RAM épinglée), x [1, 4096] bf16, 200 répétitions, 8 P + 8 HT fils comme llama.cpp : **prédit 0,80 ± 0,15 ms les 4** (57 Mo à 71 Go/s). Réfute S2/S3 si > 1,2 ms (< 47 Go/s : l hôte ne bat plus llama.cpp) ; tout le § 2 se recale à la valeur.
2. **M-TRACE proxy (≤ 10 min carte)** : `outils/trace-taux-succes-experts.py` sur Coder (top-8 de 128, 48 couches, graphes off, 5 000 jetons b=1, 20 invites réelles) puis `rapport-m1-cache-experts.py --experts 128` à C = 52 (41 %) : **prédit h_pin 0,55 ± 0,10, h_lru 0,50 ± 0,10**. Proxy assumé (Coder ≠ 119B) ; la trace 119B vraie n existe qu après le chargement (§ 4, commit 3), et la première chose qu on relève alors, c est elle.
3. **M-SYNC** : coût d un aller-retour x → hôte → y par couche dans le chemin `mlp_exec=cpu` existant (Coder, 48 couches, b=1) : prédit ≤ 0,1 ms/couche ; > 0,25 ms → S3 perd 5 ms/jeton, recalculé.

## 4. Ce qui manque au chantier C9 pour un premier run (commits nommés, dans l ordre)
1. `c9-spec-mistral4` : `ModelSpec` depuis `params.json` (pas de `config.json`) — dim, MLA (q/kv_lora, nope/rope), MoE (128/4/partagé 2048), vocab tekken (`tokenizer.json` HF présent), `llama_4_scaling` porté ou refusé nommément ; test à sec sur `params.json` réel.
2. `c9-noms-consolidated` : table de noms `layers.N.attention.wq_a/wq_b/wkv_a_with_mqa/wkv_b/wo`, `experts.N.w1/w3/w2`, `shared_experts.*`, `gate`, `attention_norm`/`ffn_norm`, `tok_embeddings`/`output`/`norm` → noms acvram (mêmes que GLM-MLA) ; import **direct** du nvfp4 compressed-tensors (`weight_packed` U8 + `weight_scale` E4M3 g16 + `global_scale`) sans requantifier — test : déquantification acvram == déquantification compressed-tensors au bit sur un expert réel (ordre des nibbles, échelle globale), comme `dequantiser-awq-bf16` l a fait pour l AWQ.
3. `c9-charge` : chargement texte seul, attention bf16 telle quelle, **tous les experts en RAM épinglée, mlp_exec=cpu** (chemin existant, loader.py:1411) — c est S2, le premier jeton, et la **trace M1 du 119B** (`ACVRAM_TRACE_ROUTAGE`). Cellule S2 b=1 contre la barre : prédit 25-30 j/s eager.
4. `c9-plan-experts` : planificateur par expert (capacité C par couche depuis le budget VRAM, pas depuis une note : `Σ` des octets réels du manifeste), pin depuis la trace.
5. `c9-moe-mixte` : `MoEBlock` chemin mixte b=1 — routage sur carte, experts chauds par le GEMV groupé existant, froids par `nvfp4_matmul_cpu` en parallèle (fil hôte), somme sur carte ; **ids au bit** contre le tout-carte sur Coder à C forcé (le même expert calculé hôte ou carte = deux arithmétiques : équivalence à ± 1 ulp bf16 par expert, PPL en plus avec SE).
6. `c9-cache-creneaux` : politique dynamique (LRU/fréquence) par `cache_experts.py` (M3) si la trace montre une dérive du chaud par invite ; sinon pin statique suffit (moins de code, aucune éviction en service).
7. Ensuite seulement : graphes CUDA avec le chemin hôte (nœud de synchronisation hors graphe, ou graphe par couche), prefill (b=2 047 : tout hôte, borne 57 Mo × 2 047 / 71 Go/s = 1,6 s → 1 300 j/s prefill théorique, à vérifier), vision.

## 5. Prédiction et seuil « ça ne vaut pas la peine »
* **Prédit** (b=1, éco 2 700, ctx 2 048, invite 256, S3 pin C = 52, h = 0,45, eager) : **35 ± 8 j/s** (S3 § 2 à h = 0,45 = 46 j/s moins 8-10 ms d eager) ; S2 seul (commit 3) : **25-30 j/s**. J/jeton non prédit avant M-HÔTE (processeur à 8-16 fils : + 60-100 W, à mesurer au compteur mural comme llama.cpp, pas à la carte seule).
* **Ça ne vaut pas la peine** — arrêt de C9, écrit avant : (a) M-HÔTE > 1,2 ms les 4 experts ; (b) S2 mesuré (commit 3) < **22 j/s** (llama.cpp − 10 %) : notre chemin hôte est moins bon que le sien, on ne rattrape pas 18 j/s de marge par le cache ; (c) h_pin 119B réel < 0,30 à C = 52 ET M-HÔTE > 1,0 ms : S3 ≤ 30 j/s pour six commits.
* **Vaut la peine** : S3 b=1 ≥ **30 j/s** (barre + 25 %) à ids au bit ; **alarme** : > 60 j/s eager mesure autre chose (lot, longueur, cache page) — je le dis avant de le publier.
* Coût : commits 1-3 ≈ une session poste1 à sec + 2 prises poste2 ≤ 30 min ; 4-6 ≈ deux sessions ; graphes/prefill/vision hors devis.
