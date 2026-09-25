# Verdict — pièce 169 : pourquoi GDN_PREFILL_LOT=1 change la sortie (poste5, 25/09) — cuBLAS bf16 à réduction bf16 selon M

* **instrument** : `scratchpad/poste5-p169-25-09/diag169.py` (crochets couche 0 et sorties de couche, micro-test par
  linéaire, variante B'), `prise.sh` → `prise.txt` (2e prise), `prise-1.txt` (1re, partielle) ; `diag169.json`
* **commit** : 952df8a4 (2e prise ; 817867ca pour la 1re) · **régime** : Qwen3.8-27B-nvfp4 défaut, -lgc 2700
* **scellé** : `revue/poste5-piece169-scelle-25-09.md` (avant ; ajout écrit entre les deux prises)
* **durée** : 1re 00:33:23-00:33:31 ; 2e → 00:47:46

## Où et combien

| comp. | 1re divergence | puis | ordre de grandeur |
|---|---|---|---|
| C1 8 × 78 | couche 0 `gate` (qkv AU BIT) | beta, alpha, out_proj, toutes les couches | 27 % des éléments, max 0,125 (≈ 1 ulp bf16 à ces valeurs) ; beta 25 %, alpha 28 % |
| C2 mêlées | couche 0 `qkv` (6,6 %, max 0,5) | gate 22 %, beta 25 %, alpha 28 % | idem |

Les quatre projections suivent le même chemin dans A et dans B : `marlin_depaquete_prefill` pour qkv et gate, le chemin
naturel pour alpha et beta, qui sont hors Marlin. Ce n'est donc pas un changement de chemin acvram. Même W exact
(module(I_K)) dans les deux cas : **c'est `F.linear` (cuBLAS bf16) qui rend des lignes différentes selon M.**

## Cause (micro-test, W exact, lot contre par séquence)

| linéaire | `allow_bf16_reduced_precision_reduction` = True (défaut torch) | = False |
|---|---|---|
| qkv [10240 × 5120] | égal (C1) | égal |
| gate [6144 × 5120] | **27 % différents** | **égal au bit** |
| alpha [48 × 5120] | **28 %** | **égal au bit** |
| beta [48 × 5120] | **25 %** | 0,01 % |

Avec la réduction en bf16 autorisée, le défaut de torch, cuBLAS découpe la réduction sur K (split-K) selon M, et somme
les morceaux en bf16. D'où une sortie qui dépend du nombre de lignes du lot. En la coupant, gate et alpha deviennent
invariants en M, au bit. Il reste 0,01 % sur beta. Mais la sortie change alors par rapport au défaut d'aujourd'hui
(`seq_egal_module` faux) : ce serait un changement de sortie servie.

## Prédictions
* 1 (première divergence : qkv) : **FAUSSE** en C1 (qkv au bit, gate d'abord), juste en C2.
* 2 (1 ulp, 0,1-10 % des éléments) : ordre de l'ulp juste, fraction **fausse** (22-28 %).
* 3 (cuBLAS selon M ; couper la réduction réduite ne suffit PAS) : cause juste, seconde moitié **fausse** : la couper
  suffit pour gate et alpha.
* 4 (B' au bit, ≥ 70 % du gain) : **au bit : oui** (C1 et C2, logits égaux) ; gain **52 % (C1) et 55 % (C2)**, prédiction
  fausse, mais au-dessus du seuil FAUX de 50 %.

| comp. | A (LOT=0) | B (LOT=1) | B' (LOT=0 + dépaquetage en cache) | part du gain gardée |
|---|---|---|---|---|
| C1 | 371,2 ms | 276,3 | **322,1** (au bit) | 52 % |
| C2 | 415,5 | 335,8 | **371,9** (au bit) | 55 % |

## Deux variantes pour chef
1. **B', au bit** : dépaqueter chaque poids UNE fois par passage de préfill, puis garder les appels par séquence. Le
   défaut ne change pas, donc aucune KL n'est requise : test d'équivalence au bit plus bras cassant, comme F4/F5. Gain :
   −49 ms (C1) et −44 ms (C2) par forward de préfill, soit ≈ la moitié de LOT=1, sans coût de qualité. Le prototype
   de la prise est un cache de `depaqueter_marlin` (clé : pointeur et forme) ; à écrire proprement dans le moteur,
   portée d'un passage.
2. **Réduction fp32 sur les projections GDN du préfill** (`allow_bf16_reduced_precision_reduction=False` localement) :
   LOT=0 et LOT=1 deviennent quasi au bit l'un de l'autre (reste beta, 0,01 %), et LOT=1 récupère tout son gain. Mais
   le défaut change une fois : sommes en fp32, probablement plus justes, à qualifier par KL contre témoins, avec un
   coût de vitesse à mesurer.
