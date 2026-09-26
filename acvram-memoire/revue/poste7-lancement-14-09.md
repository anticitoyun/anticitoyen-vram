# poste7 — lancement du travail de groupe après relecture TensorRT-LLM / ggrun (14/09, ordre de l'utilisateur)

## 1. Ce que la relecture a rendu : rien de neuf chez eux, un trou chez nous

**TensorRT-LLM** `43cd45f9f` : 2 commits depuis `9cd401a` (ma veille du 14/09), tous deux infra et tests (`#18126` NGC PyTorch 26.08, `12c5de87f` borne de lot dans un test llmapi). **ggrun** `f777280` : 0 commit. `poste7-veille-trtllm-ggrun-14-09.md` reste la référence ; P1-P8 et leurs seuils sont inchangés, rien n'est resserré.

**Ce qui contredit un acquis (REGLES §8) : le chantier 6 (conversion GLM-4.7-Flash ce soir) ne peut pas tourner sur le code actuel.** Lu dans le code, pas supposé :

| fait | où | conséquence |
|---|---|---|
| `config.json` : `model_type` = **`glm4_moe_lite`**, `Glm4MoeLiteForCausalLM` | `/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16/config.json` | — |
| notre branche MLA s'ouvre sur `mt in ("deepseek_v2", "deepseek_v3", "glm4_moe")` | `acvram/engine/config.py:575` (mla_rope, layer_types, expert partagé), `acvram/quant/convert.py:513` (scission `kv_b_proj` → `k_b`/`v_b`), `acvram/engine/loader.py:704` | **`glm4_moe_lite` n'y est pas** : la conversion passerait en silence par le chemin non-MLA — `kv_b_proj` non scindé, pas de `mla_rope` — et rendrait un dossier faux et plausible, découvert à la PPL de demain matin |
| dans HF transformers du venv vLLM, `glm4_moe` (GLM-4.5/4.6) est une **GQA classique** ; seul `glm4_moe_lite` porte `q_lora_rank`, `kv_lora_rank`, `expand_kv`, `yarn_apply_mscale` | `transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:245-306` | le nom de la liste dit « famille », la dimension réelle est « **a un `kv_lora_rank`** » — un mécanisme posé sur une liste de noms au lieu de la dimension (MECANISMES) |
| dimensions inhabituelles : `q_lora_rank` 768, `qk_nope` 192, **`v_head_dim` 256 ≠ nope**, rope 64, 20 têtes, `kv_lora_rank` 512, 47 couches, `first_k_dense_replace` 1, 64 experts top-4 + 1 partagé, I_moe 1536, sigmoid + `e_score_correction_bias`, `routed_scaling_factor` 1,8, vocab 154 880, **`torch_dtype` absent** | config.json | routeur et couche dense 0 : gérés (`loader.py:451-459`, `config.py:172`). `q_lora_rank` : géré (`convert.py:1554`). **`v_head_dim ≠ qk_nope_head_dim` : aucun site ne l'affirme ni ne le nie** (grep `v_head_dim` sans assertion) — DeepSeek et Kimi ont v = nope, c'est la dimension jamais posée. `torch_dtype` : `Optional` (`config.py:96`), le dtype vient des en-têtes safetensors — correct |
| index : 9 703 tenseurs, 48 shards ; 59 Go sur disque | `model.safetensors.index.json` | poste2 vérifie les 48 fichiers présents et leurs tailles (l'index ne dit pas ce qui est sur disque) |

## 2. Prérequis 6-pré (poste1, à sec, ce matin, avant 14 h) — scellé

**Poser la dimension une fois** : `ModelSpec.est_mla` = `kv_lora_rank > 0`, employée aux trois sites (`config.py:575`, `convert.py:513`, `loader.py:704`) à la place de la liste de noms ; les modèles GQA à `kv_lora_rank` absent ne changent pas de chemin. **Tests dans le même commit** : (i) un `config.json` `glm4_moe_lite` minimal → `mla_rope` vrai, `layer_types` posés, `shared_expert_intermediate_size` = 1536 ; (ii) le test **casse si l'on remet la liste** ; (iii) `v_head_dim ≠ qk_nope_head_dim` : un test de forme sur la scission `kv_b_proj` avec nope 192 / v 256 (formes `[nh, rank, 192]` et `[nh, 256, rank]`), et le point du moteur qui consomme `v_head_dim` au décodage MLA (sortie de `mla_decode`, projection `o_proj` d'entrée `nh × v`) nommé fichier:ligne — ou « aucun site ne le lit » écrit noir sur blanc.

**Équivalence avant conversion (poste2, à sec, après le commit d'poste1)** : forward CPU bf16 des couches 0-1 (dense + première MoE, ≈1,5 Go de tenseurs), acvram contre HF transformers du venv vLLM, même invite de 16 jetons. Scellé : **max |Δlogit| ≤ 5·10⁻² et cosinus ≥ 0,999** sur les logits des 16 positions. Réfuté → la conversion de ce soir **ne se lance pas** et chef l'apprend à 14 h, pas à 22 h. Sans ce test, l'issue « dossier faux et plausible » reste ouverte : un HF `expand_kv` avec v = 256 et un `k_b`/`v_b` scindés au mauvais offset donnent une PPL de 30, pas un plantage.

## 3. P1 — commande exacte (poste4, après (5), carte, 20 s)

```bash
# sous carte.sh, CUDA_VISIBLE_DEVICES=0, -pl 400 relevé avant/après, nvidia-smi --query-compute-apps début et fin
cd /opt/ia/flashinfer/src && /opt/ia/flashinfer/.venv/bin/python benchmarks/bench_b12x_mxfp4_moe.py \
  --tokens 1 8 12 16 --hidden-size 2048 --intermediate-size 768  --num-experts 128 --top-k 8 --warmup-ms 2000 --repeat-ms 20000   # forme Coder-30B
cd /opt/ia/flashinfer/src && /opt/ia/flashinfer/.venv/bin/python benchmarks/bench_b12x_mxfp4_moe.py \
  --tokens 1 8 12 16 --hidden-size 2048 --intermediate-size 1536 --num-experts 64  --top-k 4 --warmup-ms 2000 --repeat-ms 20000   # forme GLM-4.7-Flash
```

Défauts du banc (7168/2048/256/8, `:122-125`) = DeepSeek, jamais notre forme. En-tête de mesure obligatoire ; W au compteur NVML pendant les 20 s (`energie.py`, pas la médiane).

Scellé (Coder-30B, inchangé) : b12x statique **80-100 µs/couche à 12 jetons**, direct micro 30-40 µs à 1 jeton ; réfuté > 110 (la fusion ne vaut rien) ou < 70 (ma borne d'octets est fausse, à publier). **Scellé GLM-4.7-Flash, nouveau** : à b=12 top-4 sur 64 experts, experts touchés ≈ 64·(1 − (60/64)^12) ≈ 34 ; 3 matrices × 2048 × 1536 × 0,5 o ≈ 4,7 Mo/expert → ≈ 160 Mo/couche → 152 µs à 1 050 Go/s ; **prédit 170-210 µs/couche, 7,8-9,7 ms pour 46 couches MoE** ; b=1 : 4 experts + partagé ≈ 24 Mo → 25-40 µs. Réfuté > 240 (structure, pas octets) ou < 140.

## 4. File du jour — ce qui change par rapport à `poste7-ordre-15-09` + `poste7-reprise-14-09`

Carte, dans l'ordre : **0 poste4** llama.cpp b=1 sous ncu → **2 poste4** bt=32 → **4 poste3** modes + **P5** (20 min, garde de décodage sous prefill 512/2 048/8 192) → **5 poste4** pas complet MoE MMA sous graphes → **P1 poste4** (2 h) → **6 poste2** conversion GLM (soir), **seulement si 6-pré et l'équivalence sont verts**.

Sans carte, en parallèle :

| session | ordre | seuil / ce qui rend « faux » |
|---|---|---|
| **poste1** | (1) **6-pré** avant 14 h ; (2) gardes `energie.py` (`poste7-ordre` (b)) ; (3) **P3** `hit_rate` + `cached_prompt_tokens` dans `/metrics` et `regime_ligne` ; (4) **P4** porte de spéculation (moyenne glissante 64 pas, coupe < 1,15, `auto` = n-gram si lot ≤ 8) | (1) §2 ; (3) un banc « tour d'agent » de poste3 plus tard doit lire ≥ 85 % de réutilisation, sinon bogue ; (4) taux 1,05 simulé → coupée en ≤ 64 pas |
| **poste2** | (1) 48 shards présents, tailles vs index, `load_model_spec` sur le config : chaque champ MLA/MoE **lu de la source** (liste des ABSENT publiée) ; (2) équivalence 2 couches (§2) ; (3) corpus et VRAM de conversion prédits avant ; (4) A7 alpha commun si le temps reste ; (5) conversion le soir | (2) réfuté → pas de conversion ; (5) PPL NVFP4 ≤ étalon × 1,01, réfuté > 1,02 |
| **poste4** | entre deux tours de carte : **P7** tampons tournants > 96 Mo dans les bancs de noyaux (tuile, NV, bt) | le classement bt=32/64 change entre L2 chaud et froid → le banc chaud mentait |
| **poste8** | audit « appareil » (N/M, fichier:ligne), envoyé, vérifié par le dépôt, **jamais attendu** | `campagne-20s-vllm-14-09.py:45` doit figurer dans la liste |
| **chef** | commite ANNUAIRE + cette note ; ouvre le bead 6-pré ; correction datée B2 dans `references-moteurs-2026-09-09.md` (veille §1) ; distribue ; relevé `nvidia-smi` de début (services 8081-8083 éteints à noter) | — |

Après le duel, inchangé : P2 (MLA batching + découpe du KV, 1-2 j), P6, P8.

## 5. Ce que je ne demande pas

Installer TensorRT-LLM (son XQA MLA sm120 est de forme DeepSeek 128/576, GLM a 20 têtes : il ne sert pas le duel) ; relire ggrun tant qu'il ne bouge pas ; toucher aux seuils de la veille.
