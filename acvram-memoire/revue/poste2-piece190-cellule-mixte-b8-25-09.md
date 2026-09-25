instrument : outils/carte.sh + `scratchpad/poste2-p190-25-09/cellule-190.sh` (banc-llamacpp-16-09.py decode, ABAB×5)
commit : A 923700e4 (ex-74bcdd07, remappé post-purge) — B be837ca1 (main + 175b)
régime : plein (session 12 %, hebdo 50 %)
scellé : `revue/poste2-piece190-scelle-25-09.md` — prédiction banc mixte b=8 : +1 à +3 % débit, −2 à −6 % J/jeton
mesuré : Qwen3.8-27B-unsloth-mixte-i8c, b=8, banc chat. Débit médian A 323,4 t/s, B 394,8 t/s → **B/A +22,08 %**. J/jeton net médian A 0,8784, B 0,8125 → **−7,50 %**. 5/5 paires sans chevauchement (A∈[322,5;323,7], B∈[393,4;396,5]). Horloges comparables (moy A≈2620 MHz, B≈2555-2590 MHz), les deux bridées puissance à 400 W.
verdict : **falsificateur du scellé déclenché, mais faute de PÉRIMÈTRE du scellé, pas de mesure** (chef 25/09 11h3x) — le scellé listait cinq pièces (172/175/176/179/182) mais pas deux autres déjà dans B : **175b** (`ACVRAM_GDN_AB=auto` au défaut, −12 %/pas à b=8) et **187** de poste5 (tranche int8 6/6, +6,65 % au banc b=8). Composition correcte avec ces deux pièces en plus : 1/(1−0,12) × 1,0665 × 1,02 (fenêtre 179) ≈ **+24 %**, cohérent avec le +22,08 % mesuré. Recoupement indépendant : la 188 d'poste1 donnait 373 t/s avec la 175 seule → 373 × 1,0665 ≈ **398**, contre 394,8 mesuré ici. Écart net, propre, ABAB sans ambiguïté — PAS un artefact de mesure, la prédiction du scellé était incomplète, pas fausse comme méthode. Cellules restantes (nvfp4, gemma31, b=1) relancées en file (`ACVRAM_ATTENTE=5400`).

## Qwen3.8-27B-nvfp4, b=8 — 25/09 11h44
ABAB×5 (mêmes A/B). Débit médian A 481,9 t/s, B 499,0 t/s → **+3,55 %**. J/jeton net médian A
0,6611, B 0,6366 → **−3,71 %**. Prédiction scellé pour ce modèle : +1 à +4 % débit, −1 à −4 %
J/jeton — **TENU, dans la fourchette, aucun falsificateur**.

## gemma-4-31B-it-nvfp4-vision, b=8 — 25/09 12h01
ABAB×5. Débit médian A 402,3 t/s, B 406,2 t/s → **+0,97 %**. J/jeton net médian A 0,7944, B
0,7819 → **−1,57 %**. Prédiction scellé : NEUTRE 0 ± 2 % — **TENU, aucune des cinq pièces sources
ne touche ce chemin, cohérent avec le rôle de falsificateur direct que lui donnait le scellé**.

## Qwen3.8-27B-unsloth-mixte-i8c, b=1 — 25/09 12h19
ABAB×5. Débit médian A 62,2 t/s, B 64,3 t/s → **+3,38 %**, dans la fourchette prédite (+1 à
+4 %). J/jeton net médian A 4,8628, B 4,8604 → **−0,05 %**, quasi plat — la prédiction (−2 à
−6 %) n'est pas tenue sur l'énergie à ce B (le poste dominant à b=1 est la puissance de repos,
pas le débit ; pas un falsificateur du mécanisme, juste hors de portée du gain énergie observé
à b=8). **Bilan des quatre cellules 190 clos** : mixte b=8 falsificateur attribué (175b+187 hors
périmètre), nvfp4 b=8 TENU, gemma31 b=8 TENU (falsificateur neutre confirmé), mixte b=1 TENU en
débit / plat en énergie.
durée : prévu ≤ 15 min, tenu par carte.sh — 4 prises consécutives 11:03→11:22 (~19 min, file d'attente ~13 min avant la 1ʳᵉ prise incluse dans le journal, hors prise)
