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

Ce poste utilise ses propres menus (`kimi-modeles`, `claude-modeles`) plutôt
qu'un cockpit ; LM Studio reste intéressant comme cible de comparaison et
parce que [PAIR](VEILLE-EXTERIEURE.md) ne pilote que lui et Ollama.

## Moteurs d'inférence

| moteur | lien | statut |
|---|---|---|
| llama.cpp | <https://github.com/ggml-org/llama.cpp> · [docs](https://github.com/ggml-org/llama.cpp/tree/master/docs) | **rival de référence** — port 8080, et l'appoint « rapide » 8081 sur la 3080 Ti. GGUF, CUDA, Flash Attention, graphes CUDA, quantifications 1,5 à 8 bits, décodage spéculatif, multi-GPU |
| ggml / GGUF | <https://github.com/ggml-org/ggml> | rival (format lu par nos convertisseurs) |
| vLLM | <https://github.com/vllm-project/vllm> · [docs](https://docs.vllm.ai/) | **rival** — port 8000. Continuous batching, PagedAttention, FP8, AWQ |
| TensorRT-LLM | <https://github.com/NVIDIA/TensorRT-LLM> · [docs](https://nvidia.github.io/TensorRT-LLM/) · [page](https://developer.nvidia.com/tensorrt-llm) | **à évaluer, priorité haute** — noyaux NVIDIA, FP8, **NVFP4**, EAGLE-3, prédiction multi-jetons, optimisations Blackwell. C'est le seul moteur qui vise le même format que nous sur la même carte |
| SGLang | <https://github.com/sgl-project/sglang> · [docs](https://docs.sglang.ai/) | à évaluer — agents, long contexte, cache KV, MoE |
| Ollama | <https://ollama.com/> · [github](https://github.com/ollama/ollama) · [API](https://github.com/ollama/ollama/blob/main/docs/api.md) | à évaluer — surtout pour son **dialecte d'API**, que nos clients pourraient vouloir |

TabbyAPI (port 5000, EXL3) complète ce tableau côté maison ; acvram sert sur
**8090**.

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

Point relevé au reboot du 3 septembre : le **mode persistance est désactivé**
sur les deux cartes ; les horloges partent de 225 MHz et il faut deux à trois
tours de banc avant le plateau. À prendre en compte dans tout protocole de
mesure — d'où les 7 tours de `banc-direct.py`, meilleur retenu.

## Profilage

| ressource | lien | statut |
|---|---|---|
| Nsight Systems | <https://developer.nvidia.com/nsight-systems> | **à évaluer, priorité haute** — notre profilage se fait aujourd'hui au chronomètre en Python ; la chronologie des lancements est exactement ce que Nsight Systems montre, et le décodage est justement limité par le **nombre** de lancements |
| Nsight Compute | <https://developer.nvidia.com/nsight-compute> | **à évaluer** — occupation, pression de registres, conflits de banques : les trois causes qui ont décidé de nos dernières optimisations, mesurées jusqu'ici indirectement |
| Panorama des outils | <https://developer.nvidia.com/tools-overview> | référence |
