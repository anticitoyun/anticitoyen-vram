# duck.ai 02/10-b — NVMe MoE offload + KAT-Coder-V2.5-Dev (ordre chef)

Sources primaires (README colibrì GitHub, discussion llama.cpp #23324, model card HF KAT-Coder-V2.5-Dev, papiers PowerInfer-2/FlashMoE) ; pas de passage duck.ai (3 modèles de raisonnement) sur ce lot — WebSearch/WebFetch seul, chiffres croisés entre au moins 2 sources quand possible.

## Q1 — MoE depuis NVMe : débits, O_DIRECT vs page cache, cache hit

**colibrì** (README GitHub JustVugg/colibri, cité tel quel) :
- GLM-5.2 744B : 6×RTX 5090 pleine résidence 5,8-6,8 tok/s ; desktop CPU-only 128 Go ~1,8 tok/s chaud ; laptop RTX 5070 Ti 1,07 tok/s ; box dev 25 Go 0,05-0,1 tok/s à froid.
- Qwen3.6 CUDA, deux cartes 8 Go : 1,44 → 10,05 tok/s (×7,0) avec sortie bit-identique au CPU.
- O_DIRECT : « bypasses the page cache, souvent un gain net sur disques à cache DRAM » (+34 % décodage mesuré, PIPE=1, Blackwell/Windows, 4,25→9,69 Go/s) — **mais explicitement drive-dependent** : QLC/sans DRAM/virtualisé peut être neutre à négatif.
- Deux NVMe indépendants : +37,5 % décodage. Préfetch une couche en avance : routage prévisible à 71,6 %.
- Pas de taux de cache hit global par taille de RAM publié (mécanisme adaptatif décrit qualitativement, pas chiffré par palier de RAM).

**llama.cpp** (discussion #23324, pread direct `--no-mmap` contre mmap par défaut) :
- Apple M3 Pro 36 Go, Qwen3-30B : 38,1 tok/s sans offload → 29,1 tok/s (80 slots) → 15,7 tok/s (16 slots).
- Apple M1 Pro 16 Go : 13 tok/s rapporté (config allégée, un seul témoin, source unique).
- Windows/CUDA RTX 4080 : mmap de base 0,43 tok/s → pool de lecture asynchrone optimisé 0,75 tok/s.
- Cache hit experts : calibré sur 10 jetons de texte 42 %, oracle top-8/couche 45 %, 56 % combiné avec LRU — **seule mesure chiffrée de hit rate trouvée, source unique**, pas de second projet avec un chiffre comparable pour confirmer.
- Guide HF (Doctor-Shotgun, llamacpp-moe-offload-guide) : aucun débit MoE/disque chiffré, seule mention anecdotique Kimi K2.5 (~14 contre ~12 T/s, contexte NUMA, hors sujet NVMe).

**kTransformers** (repris du lot d'un pair, non revérifié directement ce tour) : ~4,2 tok/s sur Q4_M 377 Gio + 256 Go RAM, disque ~4 Go/s ; borne théorique top-4 routing à 3,4 Go/jeton sur 5,7 Go/s = 1,55 tok/s avant tout calcul. **Source unique**, à confirmer si réutilisé ailleurs.

**PowerInfer-2** (arXiv 2406.06282, primaire) : I/O réduit à 13,7 % du temps contre 76,7 % pour LLMFlash ; TurboSparse-Mixtral-47B 9,96 tok/s. Smartphone (OnePlus 12/Ace 2), pas de carte RTX — **contexte matériel différent du nôtre, à ne pas extrapoler à une RTX 5090**.

**Conclusion Q1** : aucun débit publié pour la config exacte demandée (1 SSD PCIe 4 ~7 Go/s, O_DIRECT contre page cache, cache hit par palier de RAM, sur UN de ces 4 projets). Le signal le plus solide et le plus proche : colibrì +34 % O_DIRECT (source unique, drive-dependent assumé par l'auteur) et llama.cpp 42-56 % hit rate (source unique). Aucun recoupement à 2 sources indépendantes sur aucun chiffre de cette question.

## Q2 — KAT-Coder-V2.5-Dev : benchmarks et écarts par quantification

Model card HF (Kwaipilot/KAT-Coder-V2.5-Dev, primaire) : base Qwen3.6-35B-A3B, MoE 256 experts routés (top-8 + partagé), 35B total / 3B actifs, contexte 262 144.

Scores publiés (bf16, seul type de tenseur donné par la card) :
- SWE-bench Verified 69,40 ; SWE-bench Multilingual 63,00 ; SWE-bench Pro 45,96.
- Terminal-Bench 2.1 41,02 ; PinchBench 93,43 ; Scicode 44,20 ; KAT-Code-Bench 46,21.
- LiveCodeBench : **non trouvé** dans la card (ni ailleurs lors de cette recherche) — à rendre comme absent, pas à déduire d'un score voisin.

Écart par quantification : **aucune évaluation officielle publiée**. La card ne couvre que le bf16 ; elle renvoie vers des quantifications communautaires (GGUF Q4_K_M/Q6_K/Q3_K ~19,2 Gio, int4 asymétrique MoE ~11,7 Go, colibrì int4-gs64 ~22 Go, NVFP4/NVFP4A16, W4A16, EXL3 4bpw) sans aucun chiffre de perte (PPL, benchmark recalculé) pour aucune d'elles. Aucune des pages de quantification consultées (HF) n'affiche de tableau qualité bf16-vs-quantifié.

**Conclusion Q2** : benchmarks bf16 solides (source primaire unique mais officielle — la card du modèle). Écart par quantification : question sans réponse fiable, rendue comme telle — aucune source, ni communautaire ni officielle, ne chiffre la perte.

**RESTE** : rien à moi. Si chef veut une 2ᵉ source sur O_DIRECT colibrì ou sur le hit-rate llama.cpp #23324, le dire — ce tour n'en a trouvé qu'une par chiffre.
