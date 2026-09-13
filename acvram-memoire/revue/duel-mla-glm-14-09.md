# Duel MLA NVFP4 vs vLLM, GLM-4.7-Grande-Heretic-42B — parqué

poste1, 14/09/2026 soir. Ordre de poste7/chef (revue/poste7-strategie-14-09.md) :
duel apparié acvram (MLA NVFP4 sm_120) contre vLLM sur ce modèle, décodage
b=1/4/12 + J/jeton — créneau où acvram gagne ×1,40 (bead 6wa, poste4) et
qu'on veut rendre défendable.

## Prérequis manquant, vérifié avant de lancer

Aucune source bf16/safetensors pour ce modèle sur le poste. Le manifeste
de notre propre conversion NVFP4 (`acvram_manifest.json`, champ
`model.name`) nomme lui-même sa source :
`GLM-4.7-30B-A3B-20-2-Heretic-30B-A3B-Q4_K_M.gguf` — déjà un GGUF Q4_K_M,
pas un bf16 (`torch_dtype: bfloat16` dans le manifeste décrit l'architecture
native, pas un fichier de poids qu'on a). Cherché sous
`/mnt/4TO_SATACMR_2022/Modeles` (tout sous-dossier `GLM*4.7*`, tout
`.safetensors`) : rien. Seuls existants : notre nvfp4
(`models_acvram/GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4`) et le GGUF de
poste3 (`models_gguf/GLM-4.7-Grande-Heretic-42B-Q4_K_M`).

## Hypothèse (a) essayée, réfutée en < 10 min (0 min de carte)

chef : un seul essai borné, `vLLM --quantization gguf` sur ce MÊME
fichier GGUF (duel le plus apparié — même source, deux moteurs), 1 h / 3
hypothèses max.

`/opt/ia/vLLM` (0.29.0, la même version que le duel A2 de poste2) —
**aucun support GGUF dans cette installation** : pas de module
`vllm.model_executor.layers.quantization.gguf`, et une recherche
insensible à la casse de "gguf" dans tout le paquet ne trouve que 3
mentions incidentes (des commentaires dans `lora/layers/utils.py`,
`exaone_moe.py`, `qwen2_moe.py`) — aucun chemin de chargement réel.
Vérifié sans GPU, sans carte, avant tout chargement : `--quantization
gguf` aurait échoué au tout premier import, pas à l'inférence.

## Verdict

**Parqué.** Pas de source bf16 sur le poste ; le GGUF existant n'est pas
chargeable par vLLM 0.29.0 (pas de support GGUF dans ce build). Reste :
télécharger la source bf16 (~80 Go, décision utilisateur, pas prise ici)
si le duel reste voulu — ou changer de version de vLLM si une version
avec support GGUF existe et charge correctement un MoE (pas vérifié,
hors de la borne des 3 hypothèses posée par chef).
