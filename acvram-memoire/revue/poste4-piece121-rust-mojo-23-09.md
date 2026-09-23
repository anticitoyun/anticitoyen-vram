# Pièce 121 — vérification de l'affirmation du chef sur Rust et Mojo, à sec

Affirmation posée mot pour mot aux trois modèles duck.ai (raisonnement) : « Rust et Mojo : aucun chiffre
vérifiable n'existe pour Rust ni Mojo sur RTX 5090/sm_120. Rust apporterait de la robustesse côté hôte, pas du
débit, puisque le coût hôte est déjà couvert par les CUDA Graphs. Mojo reste sans mesure publiée sur sm_120. »

## Verdict — **l'affirmation tombe pour Rust, tient pour Mojo**

Un chiffre vérifiable, réel, existe pour Rust sur RTX 5090/sm_120 et il montre du DÉBIT, pas seulement de la
robustesse hôte : **arXiv:2606.15991, « Fearless Concurrency on the GPU »** (Melih Elibol, Jared Roesch, Isaac
Gelado, **Eric Buehler** — l'auteur de `mistral.rs` — Michael Garland ; soumis 14/06/2026, révisé 17/09/2026).
Vérifié directement sur `arxiv.org/html/2606.15991v2` (pas seulement via duck.ai) :

> « **Grout**, a Qwen3 inference engine built on **cuTile Rust**, reaches **171 generated tokens/s for Qwen3-4B
> on the NVIDIA GeForce RTX 5090** and 82 for Qwen3-32B on the B200 in batch-1 decode, **at parity with vLLM and
> SGLang** and consistent with a memory bandwidth roofline sanity check. »

**cuTile Rust** est un système présenté dans ce même papier : un langage à tuiles pour écrire des noyaux GPU en
Rust sûr (garanties d'appartenance étendues au noyau, compilé vers Tile IR), pas une simple liaison FFI vers du
C++/CUDA existant (à la différence de `cudarc`/`tch-rs`, qui sont des bindings). Le papier rapporte aussi, pour
des micro-bancs génériques (non spécifiques à l'inférence) : débit élément-par-élément et **2,1 PFlop/s en GEMM
(98 % de cuBLAS), à égalité avec cuTile Python** — donc un langage-cadre où Rust n'est pas en retrait sur le
calcul brut non plus.

**Nuance à noter** : la synthèse de Gemma disait « Grout **bat** vLLM et SGLang » — le papier dit **« à
parité »**, pas « bat ». Correction faite ici avant de verser ce chiffre ailleurs.

Pour Mojo : **aucune source, parmi les trois modèles et la recherche indépendante, ne produit un chiffre débit/
latence sm_120/RTX 5090 vérifiable** — cette partie de l'affirmation du chef **tient**.

## 1. Réponses duck.ai (3/3, mode raisonnement, recherche web activée quand proposée)

**GPT-5.6 Luna** (19s, sourcé) : affirmation « partiellement vraie, mais trop absolue ». Vrai pour Mojo/MAX
(aucune mesure publique reproductible trouvée sur RTX 5090/sm_120/FP4). Faux pour Rust *au sens général* :
`mistral.rs` publie des comparatifs contre llama.cpp, mais **sur GB10, B200 et H100, pas sur RTX 5090** —
exemples chiffrés cités (Gemma 4 E4B) : GB10 préremplissage Q8 7 395,7 tok/s (mistral.rs) contre 3 973,7
(llama.cpp) ; B200 préremplissage Q8 27 705,6 contre 11 992,4. Conclut : « il n'existe pas, à ma connaissance,
de banc public complet et équitable des principaux moteurs Rust sur RTX 5090/sm_120 » — plus prudent que Gemma,
n'a pas trouvé le papier Grout (question de timing de recherche, pas d'erreur). Insiste sur la rigueur de
mesure attendue (modèle, quantification, GPU/pilote, CUDA/cuBLAS/CUTLASS, p50/p95/p99 TTFT et inter-token) avant
toute comparaison sérieuse. Sources : `github.com/ericlbuehler/mistral.rs`, `gigagpu.com`, `hanlab.mit.edu`,
`nikolasent.github.io` (bancs RTX 5090 génériques, pas Rust).

**gpt-oss 120B** (recherche web activée sur demande, répond en anglais malgré la question en français — anomalie
déjà notée en pièce 119/106) : verdict « the statement is false », mais ses propres chiffres cités
(`mistral.rs` sur **RTX 3090/4090**, pas 5090) ne satisfont pas non plus l'exigence sm_120 — il le reconnaît
lui-même en fin de réponse (« Rust numbers exist for RTX 3090/4090 ... Mojo/Modular still lacks any publicly
released RTX 5090/SM120 benchmark »). Cite un lien réel et pertinent trouvé indépendamment par la recherche
profonde ci-dessous : `forums.developer.nvidia.com/.../mxfp6-w6a8-on-rtx-5090-sm120-qwen3-8-27b...` (MXFP6,
pas du Rust ni du Mojo, mais un vrai chiffre sm_120 à noter pour un autre chantier).

**Gemma 4 31B** (raisonnement 73s, recherche web activée) : verdict « l'affirmation est fausse » — **seul des
trois à avoir trouvé le papier Grout/cuTile Rust**, avec un chiffre RTX 5090 réel et vérifié indépendamment
ci-dessus. Confirme aussi que Mojo/MAX n'a aucun chiffre publié sur sm_120 (FP4 compris) — la partie Mojo de
l'affirmation tient selon les trois modèles ET la vérification indépendante.

## 2. Recherche indépendante (au-delà de duck.ai)

* **arXiv:2606.15991** vérifié directement (§ ci-dessus) — la pièce centrale.
* **Aucun autre moteur Rust** (candle, burn, Ratchet, cudarc nu) n'a produit de chiffre RTX 5090/sm_120
  vérifiable dans cette recherche — seul `mistral.rs`/Grout (via cuTile Rust, pas `cudarc`) en a un.
* **Mojo/MAX** : aucune source (documentation Modular, blog, GitHub, forum) ne publie de débit/latence LLM sur
  RTX 5090 ou sm_120, FP4 ou non, au 23/09/2026 selon les trois modèles et confirmé par l'absence de résultat
  dans les recherches de la pièce 119 (Modular @ GTC 2026 mesurait sur B200, pas RTX 5090).
* **MXFP6 W6A8 sur RTX 5090/SM120, Qwen3.8-27B** (`forums.developer.nvidia.com/.../381093`, trouvé par gpt-oss) :
  hors sujet Rust/Mojo mais un vrai chiffre sm_120 supplémentaire, à verser à la pièce 119 si utile (qualité/
  débit/mémoire d'un format NVIDIA propre à Blackwell).

## Conclusion pour chef

L'affirmation, telle qu'écrite, **ne tient pas dans son ensemble** : elle est correcte pour Mojo, mais **fausse
pour Rust** — un chiffre RTX 5090/sm_120 vérifiable existe (`arXiv:2606.15991`, Grout/cuTile Rust, 171 tok/s
Qwen3-4B décodage b=1, **à parité** avec vLLM/SGLang, pas au-dessus). Le point utile pour nous : ce résultat
vient d'un langage à tuiles **compilé vers des noyaux natifs** (cuTile Rust), pas d'un simple binding Rust→CUDA
(`cudarc`) — cohérent avec la leçon de la pièce 119/120 (c'est l'instruction émise qui compte, pas le langage
hôte). Rien ici ne dit qu'un binding Rust nu (`cudarc` sans cuTile) apporterait du débit — l'affirmation du chef
serait vraie si elle visait spécifiquement les bindings Rust plutôt que Rust en général.
