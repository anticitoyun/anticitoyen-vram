# Verdict — pièce 209 (c) : les 4 couches MoE du Coder-30B servies en Marlin (facteur par ligne) — KL tenue (0,32 ≤ 2 × 0,29), PPL meilleure (12,80 → 12,61), ABBA b=8 **+5,77 %** de débit et **−8,58 %** de J/jeton, b=1 +0,52 % (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p209-25-09/kl-decode-209.py` (KL de décodage b=8, 8 × 78 + 32 pas forcés, en DEUX processus :
  A = `ACVRAM_MARLIN_PAR_LIGNE=0`, B = 1 — la pile Marlin se construit au premier passage ; T1 = 8 séquences seules à b=1 sous A,
  T2 = rejeu ; preuve du bras dans chaque processus : couches refusées « sous-normales » 4/48 en A, 0/48 en B), `prise-abba.sh`
  (banc chat 102, A B B A A B B A A B, serveur neuf, 20 s, -lgc 2700, cpu-safe 100) ; résultats `kl-*.json`, `cellule-*-b{8,1}.jsonl`,
  journaux `*.txt` ; logits hors git (A f0d43134a1ebb197…, B ec088d14abd4fc80…).
* **commit** : a6d2c3878 (ABBA) / commit suivant (KL, preuve déplacée) — poste6-209 = (a) a256676f8 + (b) 010945747 + (c).
* **régime** : Coder qkvo-i8c ; A = 44 piles Marlin + 4 naturel (`decode_mma` W4A4 à b=8, GEMV naturel à b=1, préfill « groupe »)
  = le servi d'aujourd'hui ; B = 48 piles Marlin (w13 par colonne pour les 4). Journaux serveur : A « pas de piles Marlin » × 4,
  `experts_layout=naturel` ; B 0 refus, `experts_layout=marlin-w13`. Drapeau « PASSE NULLE » de ma prise = artefact (motif de grep
  sur le mauvais jeton), preuve par les journaux ; KL : une 1re chaîne nulle (preuve posée avant la construction des piles).
* **scellé** : `scelle-c.md` (avant les prises) — KL AB max ≤ 2 × max(T1, T2), argmax, PPL par fenêtre ; ABBA b=8 +1,5 à +2,5 %,
  b=1 +0,3 à +0,8 %.
* **mesuré** : KL AB max **0,324** (moy 0,0125, p99 0,113), T1 max **0,286** (moy 0,0126, p99 0,145), T2 0 → seuil 0,572 : **TENU** ;
  argmax AB 0,961 ≥ T1 0,957 : tenu ; **PPL A 12,80 → B 12,61** (T1 12,66) : B est plus proche du bf16 (poids exacts, activations non
  requantifiées en E2M1 dans les 4 couches) ; fenêtres de 16 pas : B dans l'écart A/T1. ABBA b=8 : A 1 669,9 t/s (1 664,6-1 673,2),
  B 1 766,3 (1 763,9-1 768,6) = **+5,77 %** (étendue A 8,6 ≪ gain 96), J/jeton net 0,131 → 0,119 (**−8,58 %**), W 292 → 285 ;
  b=1 : A 336,9, B 338,7 = **+0,52 %**, J −0,54 %.
* **verdict** : (c) TENUE. Prédictions FAUSSES sur l'ampleur, dans les deux sens : KL 10 × plus grande que prévue (0,32 contre
  ≤ 0,03 — mais T1 aussi : c'est la variance b=1/b=8 du Coder, dont le W4A4 des 4 couches) ; débit b=8 +5,8 % contre +1,5-2,5
  prévus (le préfill « groupe » des 4 couches naturelles pesait plus que le décodage seul de la 206). b=1 dans la bande.
  **Recommandation : `ACVRAM_MARLIN_PAR_LIGNE=1` au défaut** (c'est le défaut de la branche ; 0 = témoin nommé) — décision chef.
* **durée** : KL 20 + 16 s, ABBA 8 + 8 min, une chaîne KL nulle (1 min) ; à sec 40 min.

## Ce qui entre (si défaut)
La sortie servie du Coder-30B change dans 4 couches sur 48 (poids exacts au bit, noyaux Marlin au lieu de naturel/W4A4) : KL
dans le témoin, PPL meilleure, +5,8 % / −8,6 % J à b=8. Aucun autre alias du parc n'a de pile écrasante (157 : GLM 0) : ailleurs,
préparation d'avant au bit (tests 209 a/b). CHANGELOG à écrire à la fusion (je le propose dans ce commit).
