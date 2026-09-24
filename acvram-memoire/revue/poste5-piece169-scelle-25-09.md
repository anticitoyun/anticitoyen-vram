# Scellé — pièce 169 : pourquoi GDN_PREFILL_LOT=1 change la sortie (poste5, 25/09 00 h 4x, AVANT la prise)

Ordre de chef. Instrument : `scratchpad/poste5-p169-25-09/diag169.py`, Qwen3.8-27B-nvfp4 (défaut), C1 8 × 78 et C2
mêlées, un processus, LOT basculé à chaud.

**Lecture du code avant la prédiction** : LOT=0 appelle `GatedDeltaNet.forward` par séquence (`couches.py`, boucle sous
l'embranchement LOT) ; LOT=1 appelle `forward_lot` (`gdn.py`), qui fait les quatre projections (qkv, gate, beta, alpha)
en UN appel sur Σ t lignes, puis la convolution, la règle delta et la norme par séquence sur des tranches, puis out_proj
sur les lignes concaténées. Seuls les linéaires voient donc un M différent. Au préfill (M > 32), un NVFP4 en disposition
Marlin est dépaqueté en bf16 puis passé à `F.linear`, donc cuBLAS (`kernels/__init__.py`, branche « pièce 134 ») ; à
M ≤ 32, il prend le GEMV Marlin, un autre noyau.

**Prédiction** :
1. Première divergence au bit : couche 0, `qkv`, le premier linéaire exécuté. Tout ce qui précède est identique au bit :
   plongements et normes, qui ne dépendent pas du lot.
2. Ordre de grandeur : 1 ulp bf16 (max ≤ 2), sur 0,1 à 10 % des éléments ; conv, état et norme n'ajoutent aucune
   divergence propre (entrées égales → sorties égales).
3. Cause : cuBLAS choisit un autre noyau, avec un autre découpage de la réduction sur K, pour M = 78 et M = 624.
   Micro-test : `F.linear` par séquence ≠ sur le lot. Couper `allow_bf16_reduced_precision_reduction` ne suffit PAS
   (prédit : toujours ≠).
4. Variante au bit, **B'** : garder les appels par séquence (le M de LOT=0), mais dépaqueter chaque poids UNE fois par
   passage (cache le temps du passage). Au bit par construction : mêmes valeurs de W, mêmes appels cuBLAS. Elle garde
   ≥ 70 % du gain de B, si le gain vient du dépaquetage (poste6 164 : 1 008 dépaquetages de moins par passage).
   Prédit, en C1 : B' ≤ A − 0,7 × (A − B).

**FAUX** si : la première divergence n'est pas dans un linéaire de la couche 0 ; ou B' ≠ A au bit ; ou B' garde < 50 %
du gain. Dans ce dernier cas, le gain de LOT viendrait du GEMM lui-même : pavage à M plus grand, efficacité du lot.

**Ajout 25/09 00 h 4x — 1re prise (817867ca), partielle, OOM à la suite de C1** : C1 relevé avant l'OOM. **Prédiction 1
FAUSSE** : `qkv` est égal AU BIT entre A et B ; la première divergence est `gate` (27 % des éléments, max 0,125,
ordre de l'ulp bf16 à ces valeurs), puis `beta_proj` et `alpha` (≈ 25 %), puis out_proj et toutes les couches.
(Mon « ulp_max » était faux : une différence d'entiers de part et d'autre de zéro n'est pas un compte d'ulp.)
L'instrument est corrigé AVANT la 2e prise : mémoire libérée entre les compositions, `expandable_segments`, et un
micro-test par linéaire (appel du module par séquence contre sur le lot, chemin NVFP4 pris, `F.linear` sur le W exact
avec et sans `allow_bf16_reduced_precision_reduction`). Nouvelle prédiction, écrite avant : gate, alpha et beta prennent
un chemin, ou un noyau cuBLAS, qui dépend de M ; qkv non. B' reste la variante au bit, et sa prédiction de gain est
inchangée.
