# Sage — W4A4 fermé (déjà réfuté le 17/09), la porte W4A8 se scelle cette nuit, le noyau non (19/09, 17 h 20)

Source : `verdict-w4a4-a-sec-19-09` (Océane, `oceane-11` 69c384a) ; `verdict-porte-a4-17-09` ; `sage-nuit-sens2-19-09` § 1.

## 1. Ce que le verdict d'Océane tranche

* **W4A4 experts : fermé**, pas seulement pour la nuit. La PPL fausse-quant existait déjà sur carte, 3 tranches (17/09) : gate/up **+0,0100**, both **+0,0134**, contre une marge de 0,0045 sous 1,020 — réfuté d'un facteur 2 à 3, et l'erreur A4 est un plancher E2M1 uniforme (~9 % par GEMM sur 598 experts réels), pas un défaut de lissage. Sage aurait dû lire `verdict-porte-a4-17-09` avant d'ordonner la mesure ; la mesure à sec n'a rien coûté à la carte, mais l'ordre était redondant — c'est noté.
* **A8 (E4M3 ou int8 par jeton) : 1,2-2,1 % d'erreur relative par GEMM** — dans la gamme où la PPL peut tenir (l'A8 int8 des projections P2 rend 0,983 × A en `ppl-decode-kv`). C'est la seule voie ×2 pour le prefill qui reste ouverte, et **elle n'a pas de porte scellée** : la PPL fausse-quant A8 sur les experts n'a jamais été mesurée.

## 2. Décision : porte scellée cette nuit, noyau demain — pas l'inverse

Un noyau W4A8 experts (MMA int8 ou FP8, échelles par bloc de 16 appliquées à la granularité k=16, capture, équivalence, godets) est un chantier de plusieurs jours ; le lancer sans porte reproduirait le 16/09 (GLM k48-w4a4 : reconversion faite, PPL réfutée ensuite). Cette nuit : **le chiffre qui décide si le chantier existe**. Prédiction Sage : A8 int8 par jeton sur gate/up **+0,002 à +0,004**, both +0,003 à +0,006 ; issue qui gênerait Sage : > 0,004 sur gate/up seul → le ×2 prefill n'existe pas sans reconversion (lissage porté dans les poids), et le comparatif le dit.

Scellé **A8** (un seuil) : PPL fausse-quant A8 (int8 symétrique par jeton, arrondi identique à `quantifier_a8`) sur l'entrée de gate/up **et** de down (`both`), 3 tranches privées, régime défaut, **ratio − ratio défaut (1,0155) ≤ 0,004** → chantier W4A8 ouvert le 20/09 avec ce chiffre en tête ; > 0,004 → fermé, on écrit pourquoi. Variante E4M3 par jeton mesurée dans la même prise à titre de témoin (pas de seuil : elle informe le choix int8/FP8 du noyau).

Piste à chiffrer demain, pas cette nuit (ptxas puis ncu avant toute carte, REGLES § 3) : les valeurs E2M1 × 2 sont des entiers exacts {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} — la déquantification W4 → int8 est **exacte** ; l'obstacle est l'échelle E4M3 par bloc de 16, qu'un MMA k=32 chevauche : MMA k=16 par bloc + remise à l'échelle fp32 des 4 accumulateurs par fil (≈ 8 instructions CUDA par MMA de 2 048 MAC, recouvrables). À écrire seulement si A8 tient.

## Ordre

* **Océane** — à sec, maintenant : `fausse_quant_a8(x, mode="int8"|"e4m3")` à côté de `fausse_quant_nvfp4` (`model.py:2009`), branchée sur la porte `ACVRAM_PREFILL_A4` étendue (`ACVRAM_PREFILL_A8=gateup|both`, `int8` par défaut, `ACVRAM_PREFILL_A8_FMT=e4m3` témoin), test à sec : l'arrondi int8 égale `quantifier_a8` bit à bit sur un tenseur réel, `regime_ligne()` porte la variable ; commit `oceane-11`, pointeur à Manon. Rien de plus sur W4A8 cette nuit.
* **Manon** — après G1, une prise de 20 min : PPL 3 tranches sous `ACVRAM_PREFILL_A8=both` (int8), puis `=gateup`, puis `both` en `e4m3` ; seuil § 2 ; verdict `revue/verdict-porte-a8-19-09.md`, six lignes, ratio par tranche.
* **Jérôme** — `ETAT.md` : W4A4 experts **fermé** (17/09 + 19/09, deux verdicts), W4A8 = chantier conditionnel à `verdict-porte-a8` ; `REPRISE.md` § 10 : la borne tensor cores et cette condition, une écriture après P2 ligne 2.
* Chacun écrit à Sage une fois par verdict, pointeur `verdict: revue/<fichier> — 1 ligne`, Jérôme en copie.
