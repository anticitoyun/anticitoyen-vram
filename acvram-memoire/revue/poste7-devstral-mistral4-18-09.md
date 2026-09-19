# poste7 — Deux modèles demandés par l'utilisateur : Devstral-Small-2-24B = oui, tout de suite, à sec (dense llama-like, 3 lignes de support à ajouter) ; Mistral-Small-4-119B-NVFP4 = format Mistral natif + MLA + 70 Go > 32 Go de VRAM : chantier de format de 3-5 j pour ≤ 10 t/s prévisibles — décision utilisateur avec ce chiffre, après P1/P2 (18/09)

Entrée : chef — demande utilisateur, disque cible 221 Go libres (94 %).

## 1. Devstral-Small-2-24B-Instruct-2512 (lu sur HF : `config.json`)

Mistral3ForConditionalGeneration, `text_config` ministral3 : dense, 40 couches, hidden 5 120, 32 têtes / 8 KV, head_dim 128, vocab 131 072, rope yarn (facteur 48, `llama_4_scaling_beta` 0,1), source **FP8 statique par tenseur** (vision + projector + lm_head en bf16). 24 Go source, ≈ 13 Go en NVFP4 : tient en VRAM, classable.
Support acvram, vérifié dans le code : `config.py:481` connaît `MistralForCausalLM → llama` mais **pas** `Mistral3ForConditionalGeneration` (une ligne dans la table) ; `text_config` est déjà déplié (`:538`) ; yarn est servi (`layers.py:653`) mais **`llama_4_scaling_beta` n'apparaît nulle part** (0 occurrence) — à lire dans transformers `ministral3` avant de convertir : s'il ne touche que les positions > 8 192, la PPL à 2 047 ne le voit pas et on le note ; sinon il se porte ; source FP8 : `hfquant.py:49-147` lit `quant_method` et des échelles e4m3 — poste2 vérifie que `fp8` statique par tenseur est un cas servi, sinon déquant → bf16 → NVFP4 (24 Go → 48 Go transitoires : le disque le permet, 221 − 24 − 48 − 13). Tour de vision ignorée (comme Gemma-4).
**Ordre** : poste2, à sec, ≤ 3 h : table d'archis + contrôle `llama_4_scaling_beta` + import FP8 + conversion calibA (bras A) → `Devstral-Small-2-24B-nvfp4-calibA` ; poste3 : PPL 3 tranches privé, témoin bf16 HF (device_map=auto) même run, **classé si ≤ 1,020** (prédiction 1,010-1,016 : dense 24B, même famille que Coder-14B 1,0143) ; puis cellules b=1/b=12 harnais égal dans un bloc. Disque : 37 Go, aucun nettoyage requis.

## 2. Mistral-Small-4-119B-2603-NVFP4 (lu sur HF : `README`, `ls`)

MoE **128 experts, 4 actifs, 6,5 B actifs / 119 B**, attention **MLA** (vLLM le sert en `TRITON_MLA`), multimodal, NVFP4 par llm-compressor (compressed-tensors) mais **format Mistral natif** : `params.json` + `consolidated-*.safetensors` (13 fichiers, **70,5 Go**), pas de `config.json` ni de noms HF — acvram n'a **aucun lecteur** de ce format (0 occurrence de `params.json`/`consolidated`). Trois chantiers avant une PPL : lecteur de format (noms de tenseurs Mistral → nôtres, 1-2 j), MLA de Mistral (variante à lire : ce n'est ni DeepSeek ni Kimi, 1-2 j), tokenizer tekken. Et l'arithmétique de service : 70 Go de poids pour 31,4 Go de VRAM → ~40 Go en RAM hôte (93 Go, 80 libres : tient), experts exilés servis par PCIe : 4 experts × ~0,45 Go par jeton ≈ 1,8 Go/jeton à ≥ 25 Go/s ⇒ **≤ 14 t/s à b=1 en borne, 5-10 réalistes** (le 70B dense exilé a rendu 1,56 j/s au prefill), prefill inutilisable. Qualité classable peut-être, vitesse non.
**Décision utilisateur, une ligne** : « 119B = 3-5 j de chantier de format/architecture pour un modèle qui ne tient pas dans la carte (≤ 10 t/s prévisibles, prefill inutilisable), pendant que P1/P2 occupent poste4. Oui après P1/P2, ou non ? » Téléchargement (70,5 Go) seulement après le oui **et** ≥ 150 Go libres (221 − 37 Devstral = 184 ; la conversion vers notre format en ajoute ~70 : libérer d'abord, liste `poste7-tri-hdd-15-09` à la main de l'utilisateur).

## Ordre

* poste2 : § 1 à sec (Devstral) ; verdict conversion (SNR, manifeste, sha256).
* poste3 : PPL Devstral + cellules, dans un bloc après l'in situ P1.
* chef : ligne § 2 à l'utilisateur ; rien ne se télécharge pour le 119B avant son oui.

## Correctif (utilisateur, 18/09) : la contrainte disque était fausse — je n'ai regardé que le disque cible (221 Go) ; 2TO_SSD_2025_IA (643 Go libres) et 4TO_STEAM0_2025 (690 Go) ont la place ; taille exacte 70,85 Go (API HF). Aucun nettoyage requis, téléchargement vers l'un des deux après le oui. Le reste de § 2 (3-5 j, ≤ 10 t/s, après P1/P2) tient.
