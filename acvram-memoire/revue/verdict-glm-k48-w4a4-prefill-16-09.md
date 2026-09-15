# Verdict — GLM-4.7-Flash-k48-w4a4 (métrique W4A4 des experts) : PPL prefill RÉFUTÉE, pire que sans le correctif

poste2, 16/09. Ordre chef/poste7 : reconversion GLM avec la métrique W4A4
des experts (`quantize_activation_nvfp4`, `c6963ec`), puis PPL NOMINAL
MMA=1 scellée ≤ 1,010 en PREFILL ET DÉCODAGE (les deux). Prefill mesuré
d'abord.

## En-tête de mesure

Instrument : `acvram eval`, une carte (`outils/carte.sh`, `CUDA_VISIBLE_
DEVICES=0`), preuve de configuration publiée (`--json`). Modèle : `GLM-
4.7-Flash-srcbf16-nvfp4-k48-w4a4` (reconverti ce soir, commit `2f4224a`
vérifié en tête de journal — `acvram.__file__` = `.../anticitoyen-vram/
acvram/__init__.py`, contient `c6963ec`). Corpus `wiki-gptq.txt`, fenêtre/
pas 2048/2048, min-context 256, régime NOMINAL confirmé avant mesure
(`acvram serve --regime` : graphes=on, couches_exilées=0/47, experts_
exilés=0/2944, piles_ok=True).

## Résultat

```
PPL prefill (k48-w4a4)     8,3293
référence bf16 (établie)   8,1427   (outils/glm-ppl-bf16-hf.py, verdict-glm-ppl-finale-15-09.md)
ratio                      1,02292
```

**RÉFUTÉ** (> 1,010). Fait notable, à publier sans l'atténuer : la version
SANS le correctif métrique (`-k48`, ordre gate!=up seul, sans `quantize_
activation_nvfp4`) donnait un ratio de **1,0183** (poste7 §8) — la métrique
W4A4 rend le résultat **pire**, pas meilleur (1,0183 → 1,02292). L'hypothèse
« la métrique choisit un alpha qui élargit l'étendue intra-bloc » (poste7-
glm-mma0-verdict §2, point 1b) ne suffit donc pas à expliquer la perte
mesurée en isolation — soit la métrique corrige un biais qui existe mais
introduit une variance plus grande sur d'autres tenseurs, soit un autre
facteur (SNR moyen sortie légèrement plus bas : 21,7 dB contre 21,8 dB
sur `-k48`, à confirmer si c'est le même bruit ou un effet réel) domine.

## Suite

Le seuil exige les DEUX régimes (prefill et décodage) ≤ 1,010 — le prefill
seul suffit déjà à réfuter, sans attendre le décodage. Je n'ai pas encore
lancé la mesure décodage (pas de protocole établi pour GLM en décodage
contre témoin bf16, à concevoir avant de consommer plus de carte) — je
demande à chef/poste7 s'il faut la mesurer quand même à titre diagnostique
avant de passer à l'option Hadamard par bloc (note séparée, déjà annoncée
comme repli par poste7).
