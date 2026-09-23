# Protocole — chantier MLA, commit 1 de Laurine (d352fe2, `ACVRAM_MLA_BATCH=2` par défaut) : tests, nsys, équivalence, PPL

Laure, 16/09/2026, avant mesure. Ordre : Jérôme. Arbre mesuré : **travail/laure-qa @ d352fe2**
(branche laurine ; `mla_ecrit_latent` remplace 1 104 copies/pas ; projections, normes, RoPE,
cat au lot 12) ; scripts depuis travail/laure (`campagne-mla1-16-09.sh`), régime du duel
(prefill W4A16, décodage MMA=1 MIN_T=5), `-k48`, une prise de carte, sorties `scratchpad/mla1-16-09/`.

## Scellés
- Sage (relayés) : **lancements/pas ≤ 2 500, pas ≤ 22 ms** ; > 2 500 → marche suivante nommée (RoPE + cat en un noyau).
- Laurine (scellée, à sec) : 2 200-2 600 lancements, 24-28 ms.
- Moi : (1) tests verts ; (2) **lancements 2 300-2 800, pas 22-27 ms** sous nsys (référence 670cd63 :
  15 534 / 43,9 ms, MLA_BATCH=1) — je rejoue aussi BATCH=1 sur le même arbre pour que l'écart soit à
  code égal ; réfuté si < 2 300 (Laurine a fusionné plus que les copies) ou > 28 ms ; (3) équivalence
  BATCH=2 contre BATCH=1 (12 × 256 greedy, logits des 64 premières positions × 12 = 768) : **top-1 ≥ 99,5 %,
  cos ≥ 0,9999 partout** ; le critère d'Océane (5 ulp) rendra « refusé » sur quelques positions comme
  sur Coder (plancher 7 ulp au décodage, `verdict-ties-moe-decode`) — publié, pas jugé sur lui ;
  réfuté si top-1 < 99 % ou une divergence avant la position 32 sur plus d'une séquence (alors ce
  n'est pas de l'arrondi : un jeton ou une tête mal indexé) ; (4) PPL décodage 3 tranches
  (`ppl-narrow-b12`, GLM -k48, 12 × 2 047) : **B/A = 1,000 ± 0,002 par tranche**, réfuté > 1,004.
Durée ≈ 2 + 6 + 3 + 15 min ; unité `mla1-laure`.
