# Laurine — rotation de Hadamard à l'exécution (ordre Sage `sage-hadamard-16-09`), 16/09

Périmètre : Manon tourne les poids d'experts (W·H, `hadamard_block: 512` par tenseur, sans AWQ
experts) ; moi : l'entrée tournée partout où les poids tournés la lisent.

## Fait (à sec)
- `nvfp4_quant_act(x, awq, e_sorted, compteurs, hadamard)` : FWHT fp32 par bloc de `hadamard`
  colonnes sur la ligne déjà en mémoire partagée (log₂ étages papillon, chaque sortie = une
  somme de deux valeurs → reproductible au bit), × 1/√bloc en fp32 RN (`__fsqrt_rn` +
  `__fdiv_rn`, pas le `sqrtf` approché de fast_math), arrondi bf16, PUIS division AWQ — l'ordre
  de `ChannelScaler.apply` (hadamard(x) / s) ; puis amax de ligne, E4M3, E2M1 comme avant.
  Aux deux sites (x pour gate/up, act pour down), décodage et prefill MMA.
- `fwht_activations(x, bloc)` (`quant/calibrate.py`) : la même arithmétique en torch (fp32,
  normalisation tenseur/tenseur), utilisée par `ChannelScaler.apply` (boucle par expert) et par
  les chemins torch de la pile (GEMV W4A16 décodage, prefill direct / `_grouped_mm`).
- `_try_build_stacks` : la pile n'est plus refusée sur `hadamard_block` ; acceptée si tous les
  experts d'une projection portent le même bloc (gate = up) → `awq["hadamard"][nom]`.
- Témoin torch (ROUTE_PACK=0) : rotation en torch avant la division, noyau à 0.
- Chemin fusionné (`ACVRAM_MOE_DECODE_FUSED`, témoin OFF) exclu quand une rotation est active.
- Tests `tests/test_moe_hadamard_pile.py` : noyau == référence Python au bit (blocs 16/64/256,
  avec et sans table ; sans drapeau les codes diffèrent) ; `fwht_activations` == référence et
  orthogonale ; pile tournée (experts W·H à canaux aberrants) : GEMV et prefill direct ≤ 1 ulp
  de la boucle, MMA/prefill MMA au bruit W4A4 avec témoin de faute (sans rotation de l'entrée,
  > 4× l'écart). Non exécutés sur carte (nvcc OK, CPU OK).

## Ce que Manon doit écrire pour que ça tourne
`hadamard_block: 512` dans l'entrée de manifeste de chaque `gate_proj`/`up_proj`/`down_proj`
d'expert (clé déjà lue par le chargeur, `loader.py:143` → `ChannelScaler.hadamard_block`), poids
`W·H` avec H = FWHT normalisée bloc-diagonale sur K (2 048 = 4 × 512 ; 1 536 = 3 × 512) — la
même transformée que `fwht_activations(w, 512)` (H symétrique : W·H = fwht des lignes de W).

## Scellés (Sage § 4) et prédiction
PPL NOMINAL MMA=1 ≤ 1,010 prefill et décodage ; W4A16 tourné = non tourné ± 0,002 ; pas b=12
≤ 19,0 ms ; prefill ≥ 12 000 j/s ; réfuté > 1,015 → W4A4 abandonné sur GLM. Coût du FWHT dans
le noyau : 9 étages × K/2 sommes en shared par ligne, ≈ +10-20 µs/couche à b=12 → +0,5-0,9
ms/pas (Sage +0,3) — à mesurer, témoin = converti non tourné.
