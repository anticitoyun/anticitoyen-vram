# duck.ai 02/10-c — colibrì vs Marlin, MLA bf16 sm_120, gabarit ChatML/Mistral (ordre chef)

Sources primaires (README colibrì GitHub, docs/issues vLLM FlashMLA, GitHub llama.cpp #8522/#20668, HF discussions chat template) ; pas de passage duck.ai (3 modèles de raisonnement) sur ce lot — WebSearch seul, chiffres croisés entre au moins 2 sources quand possible.

## Q1 — colibrì vs Marlin MoE : leviers de chargement à froid après prélecture B2

colibrì (README GitHub JustVugg/colibri, cité tel quel) place les experts sur 3 niveaux VRAM(chaud)/RAM(tiède)/NVMe(froid), avec un **préfetch guidé par routage** : "router-lookahead prefetch (PILOT=1)" annoncé à 71,6 % de prédiction correcte — une couche en avance. Sous ce mécanisme, les leviers encore cités par le projet au-delà d'une simple prélecture séquentielle (B2) sont : (a) O_DIRECT contournant le cache de pages (+34 % décodage mesuré, source unique, drive-dependent) ; (b) deux NVMe indépendants en parallèle (+37,5 % décodage, même lot que 02-10-b) ; (c) le lookahead de routage lui-même (prédire le prochain expert avant le jeton courant, pas seulement après).

Aucune comparaison directe colibrì-vs-Marlin n'existe (projets distincts, pas de banc commun trouvé) : colibrì vise l'offload disque d'un MoE entier hors VRAM, Marlin (notre usage) vise des GEMM quantifiés déjà en VRAM — périmètres différents, pas de recoupement possible sur cette question précise. **Rendu comme non comparable plutôt que de forcer un chiffre.**

Levier retenu comme potentiellement absent côté Marlin : le lookahead de routage (prédire l'expert avant le jeton, pas seulement prélire la couche suivante) — colibrì le nomme explicitement, aucune trace de l'équivalent dans le code Marlin consulté par le groupe jusqu'ici (à vérifier par qui tient le code Marlin, pas vérifié ce tour).

## Q2 — fenêtre causale MLA, pièges bf16 sur sm_120

**Point central, à ne pas confondre** : la désactivation documentée de FlashMLA ("FlashMLA is temporarily disabled on Blackwell (SM 10.0). Please use CUTLASS_MLA or TRITON_MLA instead.", docs.vllm.ai flashmla.html) cible **SM 10.0 = sm_100** (Blackwell datacenter, B200/GB200), **pas sm_120** (RTX 5090/6000, compute capability 12.0). Les deux puces sont physiquement différentes : sm_120 n'a pas le sous-système "tensor memory" qu'exige FlashAttention-4/FlashMLA avancé (confirmé indépendamment, cohérent avec le motif déjà noté le 01/10 : mma.sync sm_120 ≠ tcgen05.mma sm_100, vLLM #31085). Donc la désactivation FlashMLA citée dans la littérature vLLM **ne s'applique pas tel quel** à notre carte — risque de confusion si quelqu'un la cite comme preuve d'un bug MLA sm_120.

Aucun piège numérique (overflow bf16, biais LOWER_RIGHT, troncature causale) spécifiquement documenté pour sm_120 trouvé. Ce qui est confirmé : sur sm_120, le repli est Triton (FlashAttention-2 via Triton, pas de FlashAttention-4), chemin plus lent mais aucune source ne documente une sortie numériquement fausse — à distinguer d'un repli "lent et correct". **Question Q2 rendue sans réponse fiable** sur le piège numérique propre à sm_120 ; seul fait solide = l'architecture diffère de sm_100 et le repli Triton est confirmé.

## Q3 — ChatML vs [INST] Mistral : symptômes d'un mauvais gabarit

Cas confirmé et sourcé (GitHub llama.cpp PR #8522, ggerganov/llama.cpp) : un modèle Mistral family a reçu par défaut le gabarit ChatML au lieu de son propre format [INST] — corrigé par un PR ajoutant la détection du bon template. Symptôme générique documenté ailleurs (HF discussions, model card Mistral-7B-Instruct-v0.1 #53) : quand le gabarit ne correspond pas, "le modèle voit la conversation comme un bloc de texte confus plutôt qu'un dialogue structuré" — tokens spéciaux du mauvais format ([INST] vu par un modèle ChatML, ou `<|im_start|>` vu par un modèle Mistral) non reconnus comme délimiteurs, traités comme texte littéral.

Symptôme de répétition en boucle trouvé (GitHub llama.cpp #20668, Mistral-Small-4 sur Metal) : sortie répétitive en mode chat — mais la cause reste **non confirmée** dans l'issue (marquée "bug-unconfirmed", peut venir du backend Metal, pas forcément du gabarit). **Écarté comme preuve directe du lien gabarit→répétition** ; cité seulement comme symptôme possible, pas comme cause établie. Pas de second cas indépendant trouvé reliant explicitement un mauvais gabarit à une boucle de répétition pour du 24B — rendu "source unique et non confirmée" plutôt que retenu comme fait.

**RESTE** : rien à moi. Q1 : vérification directe du code Marlin (lookahead présent ou non) à faire par qui le tient. Q2 : aucun second signal possible trouvé, question reste ouverte si chef veut creuser plus loin (SGLang/FlashInfer sm_120 pas consultés en détail ce tour). Q3 : cas de répétition sous gabarit faux pour un modèle ≥20B non trouvé, seul le cas générique [INST]/ChatML est solide.
