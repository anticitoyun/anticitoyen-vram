# AirLLM (github.com/lyogavin/airllm) — référence externe, 17/09

Bibliothèque de streaming de couches disque→GPU : charge un décodeur à la fois
(un seul layer résident en VRAM au moment du calcul), permet de faire tourner
des modèles bien plus gros que la VRAM disponible, sans quantification,
distillation ni élagage obligatoires (quantification 4/8 bits en option,
block-wise). Licence Apache-2.0.

Mécanisme proche par l'esprit de notre exil couche par couche
(`acvram/engine/loader.py`, `layers.py` streaming dense/MoE), mais AirLLM va
plus loin : streaming par EXPERT (charge seulement les experts réellement
routés pour un jeton donné, pas la couche entière) sur les MoE géants
(Kimi K3 2,8T annoncé à 3,72 Go VRAM ; DeepSeek-V3 671B à ~12 Go). Prefetching
en option pour recouvrir chargement et calcul (seul AirLLMLlama2 le supporte
selon le README). Piste à examiner pour notre chantier exil/pool dense si le
volume d'un futur modèle géant (>100B) redevient un sujet — n'est PAS un
remplacement de notre pipeline NVFP4/AWQ, c'est un mécanisme de streaming pur
(bf16/quantifié brut, pas notre format de quantification maison).

Repères disque/VRAM cités par le projet (à vérifier avant de s'y fier, chiffres
en évolution rapide dans leur README) : 671B (DeepSeek-V3) ~12 Go VRAM,
405B (Llama 3.1) ~8 Go, 70B (Llama 3.x) ~4 Go.

Pas de modèle téléchargé à ce stade — référence documentaire seulement.
