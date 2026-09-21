# Références — matériel, outils, moteurs

Carnet d'adresses du projet. Chaque entrée porte son **statut chez nous** :
`utilisé` (dans acvram ou dans le poste de travail aujourd'hui), `rival`
(mesuré dans le comparatif des moteurs), `à évaluer` (piste ouverte, non
tranchée), `hors sujet` (noté pour mémoire, ne s'applique pas à ce poste).

Relevé du 3 septembre 2026. État local à cette date : pilote **595.84**,
torch **2.13.0+cu130**, `nvcc` **13.0.88** livré par pip
(`.venv/.../nvidia/cu13/bin/nvcc`) — le `nvcc` système est en 12.0 et ne sait
pas viser `sm_120`, c'est bien celui du venv qui compile nos noyaux.

## Matériel et pile CUDA

| ressource | lien | statut |
|---|---|---|
| RTX 5090 — 32 Go GDDR7 | <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/> | utilisé (carte principale, `sm_120`) |
| NVIDIA Local AI | <https://developer.nvidia.com/topics/ai/local-ai> | à évaluer (portail) |
| CUDA Toolkit | <https://developer.nvidia.com/cuda-toolkit> | utilisé |
| CUDA Downloads | <https://developer.nvidia.com/cuda-downloads> | utilisé |
| Guide de compatibilité Blackwell | <https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-compatibility-guide/> | **utilisé** — la référence pour `sm_120`, l'écart PTX/SASS et le repli de compatibilité |
| NVIDIA Container Toolkit | <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/> | à évaluer |
| NGC | <https://catalog.ngc.nvidia.com/> | à évaluer |

## Cockpit tout-en-un

| ressource | lien | statut |
|---|---|---|
| LM Studio | <https://lmstudio.ai/> · [doc](https://lmstudio.ai/docs/app) | à évaluer — s'appuie sur llama.cpp, GGUF, CUDA, MCP, API OpenAI |
| NVIDIA × LM Studio (Blackwell) | <https://blogs.nvidia.com/blog/rtx-ai-garage-lmstudio-llamacpp-blackwell/> | à évaluer |

acvram s'utilise par sa ligne de commande et son serveur, sans cockpit ;
LM Studio reste intéressant comme cible de comparaison, et parce que
[PAIR](VEILLE-EXTERIEURE.md) ne pilote que lui et Ollama.

## Moteurs d'inférence

| moteur | lien | statut |
|---|---|---|
| llama.cpp | <https://github.com/ggml-org/llama.cpp> · [docs](https://github.com/ggml-org/llama.cpp/tree/master/docs) | **rival de référence** — port 8080, et l'appoint « rapide » 8081 sur la 3080 Ti. GGUF, CUDA, Flash Attention, graphes CUDA, quantifications 1,5 à 8 bits, décodage spéculatif, multi-GPU |
| ggml / GGUF | <https://github.com/ggml-org/ggml> | rival (format lu par nos convertisseurs) |
| vLLM | <https://github.com/vllm-project/vllm> · [docs](https://docs.vllm.ai/) | **rival** — port 8000. Continuous batching, PagedAttention, FP8, AWQ |
| TensorRT-LLM | <https://github.com/NVIDIA/TensorRT-LLM> · [docs](https://nvidia.github.io/TensorRT-LLM/) · [page](https://developer.nvidia.com/tensorrt-llm) | **à évaluer, priorité haute** — noyaux NVIDIA, FP8, **NVFP4**, EAGLE-3, prédiction multi-jetons, optimisations Blackwell. C'est le seul moteur qui vise le même format que nous sur la même carte |
| SGLang | <https://github.com/sgl-project/sglang> · [docs](https://docs.sglang.ai/) | à évaluer — agents, long contexte, cache KV, MoE |
| ggrun | <https://github.com/raketenkater/ggrun> · [théorie](https://github.com/raketenkater/ggrun/blob/main/docs/optimizer-theory.md) | **lu le 12/09** — lanceur Go pour llama.cpp : placement MoE multi-GPU/RAM par fonction de coût et inventaire mesuré (VRAM, DRAM, PCIe) ; voir `acvram-memoire/revue/ggrun-lanceur-placement-moe.md` |
| colibrì | <https://github.com/JustVugg/colibri> · [cuda](https://github.com/JustVugg/colibri/blob/main/docs/cuda.md) · [tuning](https://github.com/JustVugg/colibri/blob/main/docs/tuning.md) | **lu le 13/09, à exploiter** — MoE 744B+ en C pur, experts streamés VRAM/RAM/NVMe, cache d'experts apprenant, prefetch double banque, O_DIRECT, KV MLA persistant ; clone `externes/colibri` ; voir `acvram-memoire/revue/colibri-hierarchie-experts-disque.md` |
| Ollama | <https://ollama.com/> · [github](https://github.com/ollama/ollama) · [API](https://github.com/ollama/ollama/blob/main/docs/api.md) | à évaluer — surtout pour son **dialecte d'API**, que nos clients pourraient vouloir |

TabbyAPI (port 5000, EXL3) a été **retiré du duel le 15/09** (Sage : hors
objectif B, ×2,5-8 derrière, aucun chemin MLA/MoE sm_120 ; ses chiffres du
14/09 restent historiques dans `revue/audit-a2`, son banc dans
`outils/archives/banc_tabbyapi.py`, l'installation `/opt/ia/TabbyAPI` reste
à l'utilisateur) ; acvram sert sur **8090**.

## Modèles et formats

| ressource | lien | statut |
|---|---|---|
| Hugging Face | <https://huggingface.co/> · [modèles GGUF](https://huggingface.co/models?library=gguf) | utilisé (source du parc) |
| Transformers | <https://github.com/huggingface/transformers> | utilisé (config, tokenizers) |
| Safetensors | <https://github.com/huggingface/safetensors> | **utilisé** — format de sortie de nos conversions |

## Quantification

| ressource | lien | statut |
|---|---|---|
| GPTQModel | <https://github.com/ModelCloud/GPTQModel> | à évaluer |
| llm-awq (MIT Han Lab) | <https://github.com/mit-han-lab/llm-awq> | **utilisé** — nous lisons l'INT4 AWQ |
| TensorRT Model Optimizer | <https://github.com/NVIDIA/TensorRT-Model-Optimizer> · [docs](https://nvidia.github.io/TensorRT-Model-Optimizer/) | **utilisé** — c'est la définition de référence du NVFP4 que nos noyaux implémentent (e2m1 + échelle e4m3 par bloc de 16 + échelle globale) |

Cible sur 5090 : Q4/Q5/Q6 côté GGUF, FP8 et **NVFP4** côté acvram, INT8 par
groupes de 128 quand le plancher de SNR l'exige.

## Accélération CUDA

| ressource | lien | statut |
|---|---|---|
| cuDNN | <https://developer.nvidia.com/cudnn> | hors sujet (pas de convolutions ici) |
| NCCL | <https://developer.nvidia.com/nccl> | hors sujet (deux cartes, un seul processus) |
| CUTLASS | <https://github.com/NVIDIA/cutlass> | **à évaluer, priorité haute** — nos GEMM groupées NVFP4 sont écrites à la main en `wmma` ; CUTLASS 3.x/4.x a des collectifs Blackwell pour les formats à blocs |
| FlashInfer | <https://github.com/flashinfer-ai/flashinfer> | à évaluer — attention paginée et cache KV, terrain de notre `paged_attn_*` |
| Triton | <https://github.com/triton-lang/triton> | à évaluer (prototypage rapide de noyaux avant portage en CUDA) |

## Long contexte et cache KV

PagedAttention (vLLM), cache KV de llama.cpp, FlashInfer — mêmes liens que
ci-dessus. Objectifs partagés avec nous : cache KV compact (nous sommes en
**INT8 par bloc**, le FP8 a été essayé et écarté), mise en cache de préfixe,
réduction de VRAM.

## RAG et bases vectorielles

| ressource | lien | statut |
|---|---|---|
| Qdrant | <https://github.com/qdrant/qdrant> · [docs](https://qdrant.tech/documentation/) | hors sujet pour le moteur |
| FAISS | <https://github.com/facebookresearch/faiss> | hors sujet pour le moteur |
| Chroma | <https://github.com/chroma-core/chroma> | hors sujet pour le moteur |
| Sentence Transformers | <https://github.com/UKPLab/sentence-transformers> | hors sujet pour le moteur |

Utile au poste de travail (leann est déjà en place), pas au moteur.

## Interface web, agents, MCP

| ressource | lien | statut |
|---|---|---|
| Open WebUI | <https://github.com/open-webui/open-webui> · [docs](https://docs.openwebui.com/) | utilisé occasionnellement — se branche sur notre API OpenAI (8090) |
| Model Context Protocol | <https://github.com/modelcontextprotocol> · [serveurs](https://github.com/modelcontextprotocol/servers) | utilisé (côté client) |
| LM Studio MCP | <https://lmstudio.ai/docs/app/mcp> | à évaluer |

## Codage local

Continue <https://github.com/continuedev/continue> · Aider
<https://github.com/Aider-AI/aider> · Cline <https://github.com/cline/cline> ·
OpenHands <https://github.com/All-Hands-AI/OpenHands> · Tabby
<https://github.com/TabbyML/tabby> — tous clients possibles de notre API ;
aucun n'est en service aujourd'hui.

## Compression de prompt

LLMLingua / LLMLingua-2 <https://github.com/microsoft/LLMLingua> — à évaluer,
côté client et non côté moteur : réduire le nombre de jetons envoyés vaut
mieux que les décoder plus vite.

## Conteneurs

Podman <https://podman.io/> · [github](https://github.com/containers/podman) ·
Docker <https://www.docker.com/> · NVIDIA Container Toolkit (lien plus haut).
Pour ce poste : **Podman + NVIDIA Container Toolkit**. Non utilisé — acvram
s'installe en `.deb` et tourne dans un venv.

## Surveillance GPU

| ressource | lien | statut |
|---|---|---|
| nvtop | <https://github.com/Syllo/nvtop> | utilisé |
| nvidia-smi | <https://docs.nvidia.com/deploy/nvidia-smi/> | **utilisé** — la source de vérité pour VRAM, horloges, puissance, température |
| LACT | <https://github.com/ilya-zlobintsev/LACT> | à évaluer (courbes et limites) |

Deux points relevés au reboot du 3 septembre, à connaître avant toute mesure :

- le **mode persistance est désactivé** sur les deux cartes ; les horloges
  partent de 225 MHz et il faut deux à trois tours de banc avant le plateau —
  d'où les 7 tours de `banc-direct.py`, meilleur retenu ;
- les deux cartes sont bridées en puissance, mais **le réglage dérive dans le
  temps** (pas persistant au redémarrage) — posé à 400 W/275 W le 8/09/2026,
  constaté à **500 W/375 W le 10/09/2026** sans qu'on sache qui ou quoi l'a
  changé. **Toujours relever la limite en vigueur au moment de mesurer**
  (`nvidia-smi --query-gpu=power.limit --format=csv`), jamais la supposer
  d'après cette page. Nos jetons/kJ sont calculés depuis la puissance
  **tirée mesurée**, pas depuis la limite posée — voir `MATERIEL.md`, qui
  établit aussi que la limite ne mordait pas dans la plage testée le 3/09,
  donc la dérive ne rend pas les anciens chiffres faux, seulement l'idée
  qu'un bridage documenté reste valable sans revérification.

## Profilage

| ressource | lien | statut |
|---|---|---|
| Nsight Systems | <https://developer.nvidia.com/nsight-systems> | **à évaluer, priorité haute** — notre profilage se fait aujourd'hui au chronomètre en Python ; la chronologie des lancements est exactement ce que Nsight Systems montre, et le décodage est justement limité par le **nombre** de lancements |
| Nsight Compute | <https://developer.nvidia.com/nsight-compute> | **à évaluer** — occupation, pression de registres, conflits de banques : les trois causes qui ont décidé de nos dernières optimisations, mesurées jusqu'ici indirectement |
| Panorama des outils | <https://developer.nvidia.com/tools-overview> | référence |

## Retours de terrain sur 5090 (lus le 10/09/2026)

| ressource | lien | statut |
|---|---|---|
| « Best Local LLM for RTX 5090 » | <https://openclawdc.com/blog/best-local-llm-rtx-5090/> | **ecarte** — page commerciale, chiffres annonces comme *« expected speed »* et non mesures, tout passe par Ollama, **aucune mesure d'energie**, aucune repetition ni dispersion. Explique les debits par « 1792 Go/s de bande passante » alors que nous avons etabli le 10/09 par deux montages independants que le decodage n'est **pas** limite par la bande passante. Seul usage : situer l'attente du public, 45 a 90 j/s selon les modeles. |
| Retour d'experience 5090, r/LocalLLM | <https://www.reddit.com/r/LocalLLM/comments/1ubkczr/> | **utile pour trois points, pas pour le fil lui-meme** — l'auteur ecrit explicitement « no actual benchmarks ». Ce qui vaut est dans les commentaires, voir ci-dessous. |

**Ce que le fil r/LocalLLM apporte reellement :**

1. **Un concurrent chiffre** : vLLM en conteneur sur une 5090, Qwen3.6 27B MTP en
   NVFP4, cache KV en FP8 → **160 a 200 j/s a contexte plein** (commentaire
   `DataGOGO`). Non verifie par nous, mais c'est le seul chiffre de la page qui
   nomme son moteur, son format et son regime. A confronter a nos mesures.
2. **Le format du cache KV sur Blackwell.** L'auteur du commentaire soutient que
   sm_120 accelere materiellement **FP8 et NVFP4**, et conseille BF16 ou **FP8**
   pour le cache KV en evitant Q8. Notre cache est en **int8 + une echelle fp16
   par vecteur de 128** (`acvram_kernels.cu:749`). La question est ouverte et
   n'a jamais ete posee : un cache FP8 supprimerait-il la dequantification ?
3. **La calibration des NVFP4 publics** : beaucoup de quantifications NVFP4
   diffusees forcent des couches en FP4 qui ne devraient pas l'etre, ou omettent
   la calibration post-quantification. Cela rejoint notre travail sur le SNR par
   tenseur et le quota de promotions — et cela vaut comme mise en garde avant de
   se comparer a un NVFP4 telecharge.

Deux observations d'usage qui recoupent nos propres mesures du 10/09 : *« Estimated
Memory Usage isn't always accurate »* (ecart entre annonce et pic, mesure a
~1,8 Gio chez nous, dont 0,70 de contexte CUDA) et *« Max Concurrent Predictions
a 1 »* qui libere de la VRAM reservee sans usage (meme famille que le budget KV
dimensionne pour 4096 jetons quel que soit `max_model_len`).

## OmniRoute — abandonné (13/09)

Passerelle API locale essayée le 13/09 (`/mnt/AI_GENERATOR/OmniRoute`, port
20128). Verdict de l'utilisateur : ne remplace pas duck.ai (modèles gratuits
paramétriques sans recherche web, REGLES §1), 19 Go de RSS en mode dev, build de
production cassée en amont → inutile, arrêté. Ne pas relancer.

## kimi-k3-in-c — Kimi K3 (2,78 T) en C99, CPU seul, 8 Go (15/09)

Source : https://github.com/FareedKhan-dev/kimi-k3-in-c (lecture, clone local
`externes/kimi-k3-in-c`, ac1584a, Apache-2.0). Ce qui vaut pour acvram :
* **Modèle hors mémoire par construction** : tronc dense épinglé à la profondeur
  choisie + anneau de lecture séquentiel (préfetch parfait, une `pread` par
  couche, « un balayage cyclique bat le LRU → préfixe épinglé ») ; 1,45 To
  d'experts routés jamais résidents, multipliés depuis le MXFP4 empaqueté ;
  sortie **bit-identique de 8 Go à 224 Go** — seule l'horloge change.
* **Deux caches, un seul qui répond** (mesuré, 12 budgets sous cgroup) : le
  cache LRU d'experts reste à 0 % de rétention jusqu'à ~36 Go puis plafonne à
  30-44 % (routeur à équilibrage par quantiles : 16/896 experts, pas de sous-
  ensemble chaud) ; le tronc épinglé rend hit ≈ épinglé/93. À budget FIXE, tout
  donner au tronc bat tout donner aux experts : ×1,69 (28,4 → 16,8 s/jeton à
  128 Go) — **l'allocation bat la capacité**. À rapprocher de notre exil ×9-23
  et de « Où acvram peut gagner » : mesurer notre courbe hit(experts) avant de
  promettre un cache d'experts (le profil AUTOPIN dit si un sous-ensemble chaud
  existe chez Coder/GLM — chez K3 il n'existe pas).
* **Instrument** : plancher de bruit publié (3 passes identiques : 33 %),
  toute table transcrite d'un fichier de données, cgroup dur pour borner la
  RAM (notre superviseur MemFree, en mieux), simulateur de cache hors ligne
  sur une trace de 100 096 requêtes d'experts (`tools/sim_cache.py`) — à copier
  pour notre `expert_usage.py` : rejouer une trace, pas relancer la carte.
* Architecture : 93 couches, 69 KDA (attention linéaire, état porté) + 24 MLA
  gated, la dernière toujours MLA ; KV MLA 2,37 Mo/position ; nibble bas =
  élément pair (fixture qui casse si l'ordre s'inverse — notre leçon du 13/09).
* Débits : 32,7 s/jeton à 8 Go → 19,2 à 224 Go (EPYC 7763, NVMe 3,2 Go/s) :
  c'est un instrument d'exécution hors mémoire, pas un rival de débit.
