# Verdict — pièce 153d (KL/argmax + ABBA, instrument de poste2 repris tel quel)

Modèle : `Qwen3.8-27B-nvfp4-attn-gdn-i8c` (sha256 vérifiés contre `prise11.txt` de poste5 avant toute mesure — intact). Branche `poste4-153d` depuis origin/main (post-201/216).

## KL(HF bf16 ‖ i8c) + argmax — instrument poste2 (scratchpad/poste2-piece174-bf16-reduction-25-09/, repris tel quel sur ordre chef)

Deux harnais maison ont échoué avant cette reprise (voir scratchpad/poste4-p153d-25-09/verdict-mesure-25-09.md) : ForwardBatch manuel (gdn_store={} vide, jamais éprouvé à 2048 jetons single-séquence) puis kl-gabarit.py (bras HF-CPU incompatible avec fla/GDN, aucun repli CPU Triton). Repro serveur i8c au défaut (une requête 2048 jetons) : HTTP 200, tient — confirmé, ce n'était pas un défaut servi.

Dump HF (`device_map="auto"`, carte + cpu cap 14 Gio — fla exige CUDA, séquentiel avec le bras acvram sous la même prise) puis instrument acvram forcé sur les mêmes ids :

| L | KL(HF‖i8c) dernière position | KL(T1 seule‖T2 lot de 5) | KL(répétition) | argmax = HF |
|---|---|---|---|---|
| 512 | 0,002087 | 0,000039 | 0,0 | oui |
| 1024 | 0,000597 | 0,000084 | 0,0 | oui |
| 2048 | 0,015472 | 0,000623 | 0,0 | oui |

`kl_hf_max` = 0,0155, très en-dessous de la référence Coder (0,519 int8 / 0,735 nvfp4, ordre de grandeur seulement). `kl_repetition` nul aux trois fenêtres (déterminisme confirmé). Argmax HF/acvram identiques aux trois fenêtres.

Point à noter : `kl_hf` (2048) est ~25× `kl_t1_t2` (composition de lot) — le témoin de composition est quasi nul, l'écart réel HF/acvram est faible en absolu mais pas nul au sens du témoin. Pas d'issue nommée franchie (le seuil de la scellé initiale portait sur un protocole per-pas différent, non reproduit ici — cet instrument mesure la dernière position de chaque fenêtre, pas le max sur toute la séquence).

## ABBA b=1/b=8, graphes=on, i8c contre Qwen3.8-27B-nvfp4 (102)

- b=1 : 70,545 t/s (i8c) / 80,93 t/s (102) → **+14,7 %** de ralentissement — SOUS le plancher prédit [+23 %,+82 %].
- b=8 : 522,9 t/s (i8c) / 591,48 t/s (102) → **+13,1 %**, dans la fourchette prédite [+2 %,+15 %].
- Surprise non nommée dans le scellé initial : écart b=1/b=8 quasi identique (14,7 % vs 13,1 %) alors que le modèle bande-passante (b=1)/calcul (b=8) prédisait un écart marqué.

## PPL (acquis en 153b, non refait ici)

`acvram eval`, fenêtre 2048/2048, wiki-gptq : **7,2157** (8 188 jetons, 0 OOM) — dans le prédictible du scellé initial sous réserve de la référence HF bf16 exacte (non recalculée séparément par `acvram eval` sur ce même corpus dans cette pièce).

## Bilan

KL négligeable en absolu, argmax parfait 3/3, répétition déterministe — pas de signal de défaut qualité sur ce chemin. Seule anomalie : le rapport b=1/b=8 de l'ABBA, plus resserré que prévu par le modèle bande-passante/calcul — à documenter, pas à trancher ici (hors scope de cette pièce).
