# Verdict final — GLM-4.7-Flash-srcbf16-nvfp4 : juge ≤×1,01 PASSÉ

poste2, 15/09 soir. Juge final scellé par poste7
(`revue/poste7-glm-equivalence-15-09.md`) : PPL du converti NVFP4 ≤
PPL bf16 × 1,01.

## Chaîne de mesure

| étape | modèle | PPL | outil |
|---|---|---:|---|
| référence | bf16 pur, source | **8,1427** | `outils/glm-ppl-bf16-hf.py` (HF `transformers`, `device_map="auto"`, accelerate) |
| livrable | NVFP4, AWQ (correctif poste1) | **8,1275** | `acvram eval` |

Même corpus (`wiki-gptq.txt`), même régime (fenêtre/pas 2048,
min-context 256), même encodage (`add_special_tokens=False`, mêmes
fichiers `tokenizer.json`), 4 fenêtres, 7164 jetons notés côté HF.

## Ratio

**0,99813** (≤ 1,01). **PASSÉ, avec marge confortable** (0,187 point sur
1,0 de tolérance) — le converti NVFP4/AWQ n'est même pas simplement
« dans le seuil », il est légèrement MEILLEUR que la référence bf16
sur ce corpus précis (8,1275 < 8,1427), écart attendu comme du bruit de
mesure à cette échelle (deux implémentations distinctes, HF vs acvram,
mêmes jetons mais chemins de calcul différents) plutôt qu'un signe que
la quantification améliore la qualité.

## Chaîne complète de ce soir, pour mémoire

1. Équivalence 2 couches CPU : RÉFUTÉE puis expliquée par un bogue de
   routage (softmax au lieu de sigmoid+biais) et un bogue MLA
   (`glm4_moe_lite` absent de trois listes) — corrigés par poste1.
2. Conversion NVFP4 sans AWQ (bogue de calibration signalé, corrigé) :
   PPL 8,4415.
3. Conversion NVFP4 avec AWQ (corrigé) : PPL 8,1275 — AWQ vaut −3,72 %
   sur ce MLA (`revue/verdict-glm-awq-mla-15-09.md`).
4. PPL bf16 par `acvram eval` : impossible (P6, marge de plan
   insuffisante, bogue tiering `--host-exec` à part).
5. PPL bf16 par HF direct (ce document) : 8,1427 — juge final PASSÉ.

## Suite

Le converti `GLM-4.7-Flash-srcbf16-nvfp4` (SSD,
`/mnt/2TO_2023_980PRO/Modeles/models_acvram/`) est le livrable validé.
`-sansawq` conservé pour comparaison (mesure gratuite d'AWQ sur MLA,
déjà utilisée). Débit/énergie non mesurés ce soir — hors périmètre de
la conversion+PPL demandée.
