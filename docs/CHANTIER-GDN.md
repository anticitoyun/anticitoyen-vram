# Chantier Gated DeltaNet (qwen35 / qwen35moe / kimi-linear)

Objectif : exécuter les hybrides à récurrence linéaire — les vrais modèles
« kimi » du parc. Refusés proprement aujourd'hui ; ce document fixe ce qui a
été établi pour l'implémentation.

## Oracle
`transformers 5.16` (déjà dans le venv) contient l'implémentation de
référence : `Qwen3NextGatedDeltaNet`, `torch_chunk_gated_delta_rule`
(prefill), `torch_recurrent_gated_delta_rule` (décodage). Stratégie : nos
QuantLinear font les projections, la règle delta vient de ces fonctions de
référence (pures, sans poids) — comme exllamav3 pour l'EXL3. Second oracle de
bout en bout : llama.cpp sert le même GGUF (jetons greedy comparables).

## Mapping GGUF (vérifié sur Agents-A1-4B-kimi, h=2560)
| GGUF | référence transformers | forme |
|---|---|---|
| blk.N.attn_qkv.weight | in_proj_qkvz **sans z** (2·key_dim+value_dim) | [8192, 2560] |
| blk.N.attn_gate.weight | z (gate), séparé | [4096, 2560] |
| blk.N.ssm_alpha.weight / ssm_beta.weight | in_proj_ba scindé (a, b) | [32, 2560] ×2 |
| blk.N.ssm_conv1d.weight | conv1d dépthwise sur qkv | [8192, 4] |
| blk.N.ssm_dt.bias / ssm_a | dt_bias / A_log | [32] |
| blk.N.ssm_norm.weight | RMSNormGated par tête v | [128] |
| blk.N.ssm_out.weight | out_proj | [2560, 4096] |
| couches d'attention (1/4) : attn_q [8192,2560] | q (+gate fusionné ? à trancher à l'oracle) | q_norm/k_norm [256] |

Hyperparamètres (métadonnées GGUF) : `ssm.inner_size`=4096 (value_dim),
`ssm.group_count`=16 (num_k_heads), `ssm.time_step_rank`=32 (num_v_heads),
`ssm.state_size`=128 (head_k_dim=head_v_dim), `ssm.conv_kernel`=4,
`full_attention_interval`=4 (3 SSM : 1 attention),
`rope.dimension_sections`=[11,11,10,0] (mrope partiel des couches attention).

## À faire, dans l'ordre
1. gguf.py : mapping ci-dessus + hf_config type qwen3_next.
2. ModelSpec : layer_types (ssm|attention), paramètres linear_*.
3. engine/gdn.py : GatedDeltaNet (projections QuantLinear + règle delta de
   référence) ; conv1d causale avec état (fenêtre 4) par séquence.
4. loader : couches mixtes ; runner : états récurrents par séquence
   ({seq_id: (conv_state, S)} par couche) — prefix cache, spéculatif et
   graphes CUDA désactivés pour ces modèles en v1.
5. Attention des couches pleines : gate de sortie + mrope par sections.
6. Vérification : logits couche à couche contre transformers (tiny synthétique
   au format qwen3_next), puis jetons greedy contre llama.cpp sur le 4B réel.


## RÉSOLU — 2 septembre 2026

Le 4B « kimi » (qwen35/Gated DeltaNet) génère **jeton pour jeton la même
sortie que llama.cpp** sur les mêmes ids, et sert un chat cohérent en NVFP4
(34 jetons/s). Trois conventions du convertisseur llama.cpp faisaient toute
la différence — aucune n'est documentée ailleurs que dans son code :

1. `ssm_a` stocke **−exp(A_log)**, pas A_log : inversé au chargement
   (`log(−ssm_a)`).
2. Les têtes V sont réordonnées « **tiled** » pour le broadcast ggml — dans la
   partie V de `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`,
   `ssm_dt.bias`, `ssm_a`, la partie V des canaux de `ssm_conv1d`, et les
   colonnes de `ssm_out`. Dé-tiling appliqué à la conversion (gguf.py),
   aller-retour testé.
3. Les poids de RMSNorm zéro-centrés portent déjà le **+1** dans le GGUF
   (sauf `ssm_norm`) : notre RMSNorm ×w est la bonne, telle quelle.

Leçon de méthode : l'oracle transformers reconstruit depuis le GGUF portait
les mêmes hypothèses fausses que le moteur — il validait nos erreurs. Seul
llama.cpp, exécuteur indépendant du même fichier, a permis de trancher, et la
bissection couche à couche a localisé chaque écart.

## qwen35moe (Ornith-1.0-35B-A3B) — 1er septembre 2026

Étendu et validé. Trois ajouts suffisaient :

1. `ffn_gate_inp_shexp` → `mlp.shared_expert_gate` : porte sigmoïde de
   l'expert partagé (vecteur GGUF [d] remis en Linear(d,1)), appliquée
   `y_partagé × sigmoid(x·g)` dans MoEBlock. Jamais quantifiée.
2. Couches MTP (`nextn_predict_layers`) : stockées en **fin de pile**
   (blk.40 porte à la fois attention pleine, experts et les tenseurs
   `nextn.*`) — la couche entière est ignorée à la conversion et retranchée
   de `num_hidden_layers`.
3. Les couches GDN peuvent porter un MoE : construction MLP/MoE factorisée
   dans le loader (`faire_mlp`).

Vérification : 7 premiers jetons greedy identiques à llama.cpp
(`' Paris. Paris is a city of'`), puis divergence sur un quasi-ex-æquo —
attendue : llama.cpp exécute l'IQ4_XS, acvram le NVFP4 reconverti, et le
routage à 256 experts amplifie les écarts d'arrondi. Les deux suites sont
grammaticales et factuelles. Alias : `acvram-ornith-35b-kimi`.

## kimi-linear (Kimi-Linear-REAP-35B-A3B) — 1er septembre 2026

KDA + MLA + routeur DeepSeek implémentés (`engine/kda.py`, `engine/mla.py`),
**24/24 puis 47/64 jetons greedy identiques à llama.cpp** avant un
quasi-ex-æquo (Q4_K vs NVFP4). Ce qu'il a fallu établir :

1. **KDA** (trad. llama.cpp kimi-linear.cpp + delta-net-base.cpp) : conv
   causales séparées q/k/v (+SiLU), q/k L2-normalisés par tête, q/√d,
   ``g1 = ssm_a · softplus(f_b(f_a(x)) + dt_bias)`` par canal,
   ``β = sigmoid(beta(x))`` par tête, sortie RMSNorm×sigmoid(g_b(g_a(x))).
   **La décroissance exp(g1) porte l'axe CLÉ de S** (convention fla/vLLM) —
   ma lecture du broadcast ggml disait l'axe de sortie : l'autre axe donne
   7 jetons justes puis des boucles. Tranché par bissection.
2. **MLA absorbée, sans RoPE** : cache latent [t, 576] par séquence (rank 512
   + rope 64), q_nope absorbé par k_b, valeurs relues dans l'espace latent
   puis v_b. Les couches MLA passent par le même magasin d'états que les
   couches récurrentes — aucun cache paginé dans ce modèle.
3. **Routeur DeepSeek** : sigmoïde + biais de sélection (exp_probs_b, hors
   poids) + renormalisation + ×2,446 ; expert partagé sans porte ; couche 0
   dense (leading_dense_block_count).
4. **EOG du gabarit** : le GGUF ne déclare qu'un eos (163585) mais le chat
   Kimi clôt par ``<|im_end|>`` (163586) — heuristique d'export ajoutée.
5. **Bug de gabarit générique découvert** : `_render_jinja` passait
   ``bos_token`` deux fois (explicite + **config) → TypeError → repli ChatML
   silencieux. Invisible sur les modèles ChatML (qwen), fatal pour le format
   ``<|im_user|>…<|im_middle|>`` de Kimi. Corrigé.

Qualité : le modèle lui-même (REAP-élagué, abliterated) est fragile en
factuel — llama.cpp répond « Brisbane » à la capitale de l'Australie sur le
même GGUF. Fiche ★★, 39 t/s, alias `acvram-kimi-linear-35b`.

Le flag ACVRAM_GDN=1 couvre désormais qwen35, qwen35moe et kimi-linear.

## Accélération des hybrides — 1er septembre 2026

Point de départ : kimi-linear à 39 t/s contre ~210 pour llama.cpp. Profil
d'un pas de décodage : **9 800 lancements CUDA**, 14 ms de GPU pour 25 ms de
mur — le pas était noyé dans les lancements, pas dans le calcul.

1. **Graphes CUDA pour les couches à états** (`graphs.py`, `model.py`) :
   chaque couche hybride reçoit des tampons fixes (états KDA/GDN, cache
   latent MLA borné + longueur sur l'appareil) ; une séquence par graphe,
   clé `(b=1, q_len=1, blocs, palier MLA de 1024)`. Au changement de
   séquence, l'état du propriétaire est exporté vers le magasin
   (sentinelle `_STATIC`) et celui de la nouvelle séquence chargé ; le chemin
   eager sait relire un état résidant dans les tampons.
   Piège : l'échauffement de capture (2 passes + rejeu) *avance* une
   récurrence trois fois pour un jeton — les états sont photographiés avant
   et restaurés avant le premier vrai rejeu.
   Piège : eager et graphe divergeaient d'un ulp bf16 dès la couche 8
   (conv manuelle vs cuDNN, `addcmul_`, GEMM d'attention sur 1024 colonnes
   contre N) — même noyaux partout et godet `MLA_BUCKET` partagé par le
   forward t=1 : les deux chemins sont désormais **bit-identiques**.
   → kimi-linear 39 → 105 t/s, 4B kimi 49 → 105, Ornith 25 → 91.
2. **Noyaux fusionnés** (`acvram_kernels.cu`) : `kda_decode_kernel` (un bloc
   par tête : convs, normes, portes, récurrence en registres, norme de
   sortie — 1 lancement au lieu de ~35), `mla_scores/reduce` (2 au lieu de
   ~20). GPU 14 → 8,7 ms/pas, 2 388 lancements. → 113 t/s.
3. Reste dans le pas (8,7 ms) : GEMV int8 des 271 projections hors experts
   (2,45 ms — promotions SNR du convertisseur ; à comparer avec une
   conversion tout NVFP4), MoE groupé 2,0 ms (40 % de la bande passante sur
   K=1024/2304), ~580 copies de conversion (1,2 ms).

`ACVRAM_GRAPHS_EAGER=1` exécute le chemin fixe sans capture ;
`ACVRAM_HYBRID_KERNELS=0` rétablit le chemin torch des couches hybrides.

### Bilan — 1er septembre 2026, 13 h 45

| modèle | avant | moteur (graphe) | serveur (flux HTTP) |
|---|---|---|---|
| kimi-linear 35B (KDA+MLA) | 39 t/s | **135** | 54 |
| Agents 4B kimi (GDN) | 49 | **125** | 84 |
| Ornith 35B (GDN+MoE) | 25 | **108** | 68 |
| Qwen3-14B (dense, non hybride) | 36 | — | 44 |
| Qwen3-Coder-30B-A3B (MoE) | 49 | — | 72 |

Les modèles classiques profitent aussi des GEMV bf16 et de la RMSNorm
fusionnée (+20-45 %). L'écart moteur/serveur est le coût du prefill du
gabarit de chat, du flux SSE et du tokenizer par jeton — chantier suivant.
Restent côté pas de décodage (7,4 ms) : le MoE groupé (2 ms, 40 % de la bande
passante sur K=1024/2304) et l'empilement des projections KDA (9 → 5 GEMV).

### Volet serveur — 1er septembre 2026, après-midi

L'écart moteur/serveur (kimi-linear 135 contre 54 t/s) n'était pas un
surcoût par jeton : la médiane des écarts inter-jetons valait le pas moteur
(7,8 ms), mais p90 55 ms et max 190 ms. Trois causes, trois correctifs :

1. **Collectes GC de génération 2** (~100 ms, tout le tas parcouru :
   manifeste de 14 000 tenseurs, tokenizer, modules) → `gc.collect()` +
   `gc.freeze()` après le chargement, seuils espacés (`ACVRAM_GC_FREEZE=0`
   pour comparer).
2. **Spéculation n-gram par défaut** (`--speculative ngram`, k=4) : sur les
   hybrides la vérification q_len = 5 est inéligible au graphe → passe eager
   de 26 ms à chaque proposition (plateau sur 10-20 % des pas ; 4B kimi à
   68 t/s). Désactivée pour les hybrides tant que ce chemin manque.
3. **Captures de godets en pleine réponse** (40-130 ms aux passages 128, 256,
   512, 1024 jetons) → `warm_graphs` au démarrage (`ACVRAM_WARM_GRAPHS`,
   2048 par défaut).

Diagnostic : `ACVRAM_TRACE_STEPS=1` journalise médiane/p90/max des pas, les
pauses GC (`gc.callbacks`) et le détail des pas lents ; côté client, la
distribution des écarts (médiane contre p90/max) distingue un surcoût
constant de décrochages. Le lanceur `acvram-serveur` active le venv du
projet, pas celui du .deb.

### Tableau final — 1er septembre 2026, 14 h 45

| modèle | départ | moteur (graphe) | serveur (flux HTTP) |
|---|---|---|---|
| kimi-linear 35B (KDA+MLA) | 39 t/s | **139** | **105** |
| Agents 4B kimi (GDN) | 49 | 125 | **111** |
| Ornith 35B (GDN+MoE 256) | 25 | 108 | 56-89 (instable) |
| Qwen3-14B (dense) | 36 | — | 46 |
| Qwen3-Coder-30B-A3B (MoE) | 49 | — | 78 |

Ornith reste instable d'une passe à l'autre (48 à 89 t/s) : une dizaine de
rejeux de graphe à 70-150 ms par réponse de 300 jetons, sans GC ni capture
en cause, avec 21,6 Go occupés sur 32 (les piles d'experts doublent
transitoirement la mémoire à la construction, l'allocateur est près de sa
limite). Piste : construire les piles avant le placement du cache KV, ou
libérer explicitement les tenseurs d'origine.

4. **Plongement CPU en mmap** (tous les modèles : le plan place la table sur
   CPU) : `reader.get(...).to(bf16)` ne copie rien, la table reste un mmap du
   safetensors sur le disque SATA — chaque jeton dont la ligne n'est pas en
   cache de pages coûtait 90-150 ms de lecture disque dans `_fill`. Copie
   contiguë en RAM épinglée au chargement : Ornith 48-89 → **100 t/s**
   stable (p99 10,7 ms). C'était la cause de l'« instabilité » ci-dessus.

### Tableau final après le plongement résident — 1er septembre 2026, 19 h

| modèle | départ (31/08) | serveur maintenant |
|---|---|---|
| kimi-linear 35B (KDA+MLA) | 39 t/s | **132** |
| Agents 4B kimi (GDN) | 49 | **111** |
| Ornith 35B (GDN+MoE 256) | 25 | **100** |
| Qwen3-Coder-30B-A3B (MoE) | 49 | **85** |
| Nemo-12B (dense) | 28 | **58** |
| Qwen3-14B (dense) | 36 | **50** |
| Cydonia-24B (dense) | 35 | **44** |

Chiffres en flux HTTP, mesurés par le menu (`acvram-serveur`), prompt de
200 jetons. L'écart moteur/serveur est désormais de 5 % sur kimi-linear.

### Soir du 1er septembre — MoE et GEMV int8

- GEMV groupé MoE réécrit « un warp par ligne » + noyau **gate+up+SiLU
  fusionné** (44 → 32 µs par couche) : kimi-linear 139 → **156 t/s**, Ornith
  108 → 118, Qwen3-Coder-30B 49 → 104 (moteur). Mesuré et écarté : déroulage
  ×2 (neutre), plusieurs lignes par warp (pire — le parallélisme l'emporte).
- Banc int8 : un noyau warp-par-ligne fait jeu égal avec l'existant **hors
  L2 et sous graphe** (~700-800 Go/s) ; les micro-bancs naïfs mentent deux
  fois (matrice résidente dans les 128 Mo de L2 ; lanceur Python ~30 µs par
  appel). Une GEMV de 9 Mo ne rampe pas jusqu'à la bande passante crête :
  seule la fusion de lancements paie encore. Noyau retiré.
- Empilements INT8 (même entrée) : q/k/v et f_a/g_a de KDA, gate/up des MLP
  (experts partagés, denses), q/kv_a de la MLA — 158 t/s.
- Serveur : kimi-linear **152**, Ornith **114**, 4B **112**, Coder-30B **90**.
- Routage MoE en un noyau (`moe_route` : scores, biais, top-k par argmax
  itéré, renormalisation, échelle — 7 lancements en moins par couche) :
  kimi-linear **164 t/s** (6,1 ms/pas), Ornith 121, mêmes experts et poids
  qu'en torch à 1e-7.

Postes restants du pas (6,1 ms) : GEMV int8 2,2 ms (plancher de rampe DRAM
sur 216 projections), MoE 1,5 ms, KDA 0,5, MLA 0,3, normes 0,3, reliquat
élémentaire 0,3 — chaque fusion supplémentaire vaut moins de 3 %. Le gain
suivant est structurel : vérification spéculative à formes fixes pour les
hybrides (états récurrents à photographier par position pour le retour
arrière) — rentable sur les sorties répétitives (code, agents).

### Prefill des hybrides — 1er septembre, nuit

Mesure de départ : kimi-linear 176-411 j/s, 4B 259-561 (les denses : 3 400).
Trois causes : la récurrence KDA en boucle Python par jeton, les scores MLA
matérialisés en fp32 (2 Go à 4k, OOM à 8k), et le MoE au prefill (boucle
par expert : ~3 s fixes ; GEMV groupée : relit les poids par paire).

| modèle | avant | après |
|---|---|---|
| kimi-linear 4096 / 8192 | 5,3 s / OOM | **1,5 s / 2,65 s** (3 088 j/s) |
| Agents 4B kimi 4096 | 7,3 s | **0,45 s** (9 191 j/s) |
| Ornith 4096 | — | 1,30 s (3 160 j/s) |
| Qwen3-Coder-30B 4096 | — | 1,14 s (3 589 j/s) |

- KDA : `fla.ops.kda.chunk_kda` (Triton) pour t > 1 — même mathématique que
  la boucle (7e-4), accord llama.cpp 47/64.
- MLA : attention par tranches de 256 requêtes.
- MoE : jetons triés par expert, pile déquantifiée en bf16 par projection et
  par couche, `torch._grouped_mm` ; contre une référence fp32 exacte, ce
  chemin est à 0,5 %, la GEMV groupée à 0,3 % — et l'ancienne boucle W4A8
  par expert à **8,5 %** (fp8 par ligne sur de petites matrices) : le
  prefill MoE gagne en précision en plus de la vitesse.
- Le premier prefill d'un processus paie l'autotune Triton (3-5 s), absorbé
  par le warm-up du serveur.
