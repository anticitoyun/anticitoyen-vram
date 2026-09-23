# Sage — AirLLM et DeepSeek-V4-Flash : ce que la machine permet, chiffré avant tout téléchargement (17/09)

Entrées : `reference-airllm-17-09` (Jérôme) ; demande utilisateur « télécharger DeepSeek V4 Flash ».

## 1. Le dépôt existe : `deepseek-ai/DeepSeek-V4-Flash-0731`

https://hf.co/deepseek-ai/DeepSeek-V4-Flash-0731 (MIT ; variante de base `DeepSeek-V4-Flash`, 22/04/2026 ; `-0731` = révision du 31/07, la plus téléchargée). `config.json` lu : architecture **`deepseek_v4`**, 304 G paramètres, 43 couches, hidden 4 096, **256 experts routés × 6 actifs + 1 partagé**, `moe_intermediate 2048`, **experts déjà en FP4** (`expert_dtype: fp4`), reste en FP8 blocs 128 ; attention à rangs bas (q/o lora 1 024, 1 tête kv, head_dim 512, rope 64) **avec compression (`compress_ratios` 4/128 alternés), indexeur épars (`index_topk 512`), couches de hachage, `hc_*` (hyper-connexions), `dspark` (prédiction multi-jetons)**. 48 fragments ≈ 3,5 Go ⇒ **≈ 165 Go**.

## 2. La borne physique, indépendante du moteur

Actifs par jeton : experts 7 × 3 × 4 096 × 2 048 × 43 ≈ **7,6 G paramètres ≈ 4,3 Go en FP4** ; poids d'experts totaux ≈ 277 G ≈ **140 Go en FP4** — ne tiennent ni dans 32 Go de VRAM, ni dans **96 Go de RAM hôte**. Donc le flux vient du SSD (NVMe ~7 Go/s) : 4,3 Go par jeton ⇒ **≤ 1,6 jeton/s à b = 1, plus probablement < 1**. Le lot n'aide pas : 12 jetons touchent ~60 experts distincts par couche, soit ~10× les octets pour 12× les jetons — même coût par jeton. AirLLM annonce des Go de VRAM, jamais des jetons/s : ses 3,72 Go pour Kimi K3 sont exacts et sans intérêt pour servir. Notre exil mesuré aujourd'hui (70B : 1-1,5 j/s prédit) dit la même chose depuis l'autre bout.

Deux choses seulement changeraient la borne : **RAM ≥ 192 Go** (experts en hôte épinglé ⇒ PCIe 21 Go/s ⇒ ~4-5 t/s), ou une carte de 80 Go+ (hors sujet).

## 3. Ce que ça coûterait au circuit

`deepseek_v4` n'existe pas dans acvram (`config.py:487-488` : V2/V3 seulement) ; les mécanismes du § 1 (attention compressée, indexeur épars, hachage, hyper-connexions) sont **chacun un chantier** : semaines, pour ≤ 1,5 t/s. Pas de cellule classable (pas de bf16 servable : 600 Go), pas de comparaison honnête avec llama.cpp (qui a un GGUF `ds4` communautaire, lui aussi à la vitesse du SSD).

## 4. Réponse pour l'utilisateur, une ligne par choix

* **Ne pas télécharger maintenant** : 165 Go pour un modèle que la machine sert à < 1,6 t/s et que le moteur ne lit pas — c'est ma recommandation.
* **Télécharger quand même** (2,6 To libres sur `/mnt/16TO_LORAS_2025`, hors de la convention HDD/SSD des convertis) : seulement si l'objectif « MoE géant en flux par expert » devient un objectif du projet *après* les cellules classées — avec la borne du § 2 écrite dans ETAT avant, pour qu'un futur chiffre ne surprenne personne.

## 5. AirLLM et notre exil : rien à changer

Nous streamons **déjà par expert** sur les MoE (`layers.py:308-322`, `ExpertPool` : copie après le routage, jamais la couche entière) ; la couche entière ne concerne que les denses (`StreamedWeight`, `layers.py:177`). La différence réelle d'AirLLM est la source (disque au lieu de RAM épinglée) — précisément ce qui le rend lent. À garder en référence, pas en chantier.
