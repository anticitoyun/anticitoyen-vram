# Protocole — rejuger le latent fp8 et le chemin batché avec `ACVRAM_HYBRID_SLOTS=12` (main dcb09ba)

Laure, 16/09/2026, avant mesure. Ordre : Jérôme / Laurine (dcb09ba) : sans `ACVRAM_HYBRID_SLOTS ≥ b`, 8 des 12
séquences passaient par `forward_batch` eager bf16 (`model.py:1583-1596`) — **mes équivalences ties / PPL / arbitre
du jour à b=12 n'exerçaient les créneaux statiques que pour 4 séquences sur 12** ; seuls les nsys (qui posent
`HYBRID_SLOTS=12`) mesuraient le chemin batché entier. Arbre : travail/laure au commit de ce protocole (`acvram/`
= main dcb09ba, extension recompilée hors verrou). Régime : W4A16 prefill et décodage, `-k48`, b=12, prompt 128.

## Montage
(a) ties (768 logits) + arbitre prefill (84 points) pour **C** défaut, **D** `ACVRAM_MLA_LATENT_FP8=1`, **B0** boucle ;
(b) PPL décodage teacher-forcé tranche 0 (`ppl-narrow-b12`, jeton par jeton sous créneaux) pour C et D.
Scellé (Sage § 9) : arbitre ≥ 80/84, cos ≥ 0,9999 ; fp8 : PPL D/C dans 1 ± 0,004.

## Mes prédictions (scellées)
1. **D ≠ C cette fois** : logits non identiques (bras qui DOIT réagir) ; réfuté si encore identiques → le fp8 n'est
   toujours pas exercé, retour au code avant tout chiffre.
2. Arbitre : C **80-82/84** ; **D 77-81/84**, |Δ| médian +0,1 à +0,4 au-dessus de C ; B0 **79-81**.
3. PPL décodage D/C **1,002-1,006** (E4M3 par ligne, 3 bits de mantisse sur le latent de rang 512) ; tenu ≤ 1,004,
   réfuté au-delà → bf16 gardé.
Unité `slots12-laure`, ≈ 12 min, sorties `scratchpad/arbitre-slots12-16-09/`.
