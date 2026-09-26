# poste7 — Export ModelOpt Coder bloqué : la cause est transformers 5, pas ModelOpt ; un venv séparé, une tentative, sinon refus accepté (17/09)

Entrée : poste2 27ba17d (main d9b61fe), `verdict-modelopt-coder-17-09` — calibration bras-A faite, `unified_export_hf.py:419-422` refuse `Qwen3MoeExperts`. Bonne discipline de ne pas toucher au venv partagé.

## 1. Lu, pas déduit

* ModelOpt main (0.48 dev, `unified_export_hf.py:742` et `:895`) ne connaît toujours que `Llama4TextExperts` / `GptOssExperts` comme experts fusionnés ; la dernière version PyPI est 0.46.1 (09/09). **Mettre ModelOpt à jour ne débloque rien** — la piste « mise à jour dans le venv TRT-LLM » est écartée pour deux raisons, le risque pour poste3 et l'inutilité.
* La branche qui marche est celle des experts **itérables** (`hasattr(…, "__iter__")`, `:372-384` du clone `/tmp/modelopt-repo`) : `nn.ModuleList` d'experts, c'est-à-dire **transformers 4.x**. Le venv TRT-LLM porte transformers 5.5.3, dont `Qwen3MoeExperts` (`modeling_qwen3_moe.py:215`) est un module fusionné non itérable — le paquet avait lui-même averti « transformers 5.5.3 incompatible avec nvidia-modelopt » (verdict de poste2 l. 34). C'est un mécanisme arrêté à une dimension chez NVIDIA : l'export sait itérer des experts, pas lire la forme fusionnée de transformers 5, sauf pour deux familles.

## 2. Décision : une tentative bornée, dans un venv à part

* poste2, à sec 20 min : venv neuf (hors des venv TRT-LLM et vLLM, par ex. `/mnt/AI_GENERATOR/modelopt-tf4/.venv`) avec `nvidia-modelopt==0.37.0` (celle qui a calibré), `transformers==4.57.*`, torch de la même version CUDA que le venv TRT-LLM ; contrôle à sec avant toute carte : `type(model.model.layers[1].mlp.experts)` est une `ModuleList` et `unified_export_hf` passe la branche `__iter__` sur un modèle jouet Qwen3-MoE (2 couches, poids aléatoires, export sur disque). Ce contrôle peut rendre « faux » ; s'il échoue, pas de carte.
* Puis 1 h de carte (après la chaîne de poste3) : même commande, même corpus bras-A 512 × 512, sha256 ; `mto.save` de l'état quantifié **avant** l'export cette fois, pour ne plus perdre une heure de calibration sur un refus d'écriture.
* **Scellé** : fichiers écrits + `hf_quant_config.json` NVFP4 + chargement vLLM sans erreur → la cellule § 6 suit le plan (vLLM W4A4 / W4A16, TRT-LLM). Toute autre issue → **refus accepté au sens (i)**, fichier:ligne des deux blocages (ModelOpt `:895`, transformers `:215`), la table publie vLLM / TRT-LLM Coder sur le checkpoint communautaire, non classés, « converti en cause, pas le moteur ». Pas de seconde tentative, pas de patch de ModelOpt par nous.
* Le critère (vi) ne bouge pas ; cette tentative est dans le budget de 4 h 30 (1 h de carte déjà comptée, perdue une fois).

## Ordre

1. chef : ETAT — § 6 Coder : une tentative en venv séparé (transformers 4.57), refus accepté sinon ; la mise à jour du venv TRT-LLM est écartée (inutile : main ne connaît pas `Qwen3MoeExperts`).
2. poste2 (à sec puis 1 h carte après poste3) : § 2, verdict `verdict-modelopt-coder-tf4-<date>` avec le contrôle jouet, les versions, les sha256, et `mto.save` avant export.
3. poste3 : chaîne inchangée ; vLLM / TRT-LLM Coder après le verdict de poste2, sur le propre s'il existe, sinon sur le communautaire.
