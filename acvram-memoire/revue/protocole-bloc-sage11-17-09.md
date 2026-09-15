# Protocole — bloc poste7 § 11 (main 524ccd8) : AWQ experts, temps du duel, PPL décodage préfixée, arbitre

Instrument : `scratchpad/bloc-sage11-17-09/chaine.sh` (arbre poste3 780a067 = main 3082bda fusionné, extension recompilée hors verrou) ; PPL par `ppl-acvram-17-09.py` (fenêtres `encode_brut`, médiane) ; temps par `certifie-b12-15-09.py` (rondes, SLOTS=12, régime duel W4A16 prefill+décodage, MIN_T=5, graphes) ; PPL décodage par `ppl-decode-prefixe-17-09.py` (copie de `outils/ppl-narrow-b12-coder30b.py` : invite = `encode_brut("")` + premier jeton, 12 séquences × 2 047 cibles, médiane des séquences) ; arbitre par `ties-moe-decode-15-09.py` (`TIES_PREFIXE_BRUT=1` : invite = préfixe + 126 jetons tirés) + `arbitre-prefill-mla-16-09.py`.

## Scellés (poste7)
1. `-avant-alpha-experts-fix` (AWQ sans les experts) : médiane privée ≤ 1,022 ET |privé − public| ≤ 0,01 → l'AWQ des experts est retirée de la recette GLM ; réfuté → autre cause à nommer. Témoin ajouté : `-sansawq` (awq=False, même source), mêmes critères lus, non scellé.
2. Temps b=12 (−k48) : 21,1 ± 0,3 ms/pas (deux passes).
3. PPL décodage préfixée (privé, 3 tranches, médiane des 36 séquences) : mon scellé : rapport à la PPL prefill médiane du même converti (`verdict-ppl-refonte`, 1,029 × bf16) dans ±0,01 — le décodage égale le prefill (arbitre ≥ 80/84 le 16/09).
4. Arbitre b=12 avec préfixe : ≥ 80/84, cos ≥ 0,9999 (comme prise B).
