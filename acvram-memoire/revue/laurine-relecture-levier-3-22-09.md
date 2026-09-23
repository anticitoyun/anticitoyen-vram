# Relecture levier 3 (Océane `oceane-levier-3-conception-22-09` ccc07d43) — B et C retenus, reduce-dans-down écarté à raison, µs tenus par module, tests au bit manquants (Laurine, 22/09, à sec)

instrument : lecture croisée `oceane-levier-3-conception-22-09.md`, `model.py:1085-1125` (chemin plain/marlin_c17, `nvfp4_quant_act`/`moe_act`/`moe_reduce_trie`), `model.py:1760-1780` (`_MOE_FUSED_ATOMIQUE`, `_moe_fused`) ; 0 min de carte

## Reduce dans down : je retiens l'écart d'Océane, mais pas pour la raison qu'elle cite
`_MOE_FUSED_ATOMIQUE` (`model.py:1772`, « témoin, non reproductible ») ne dit PAS que fusionner un reduce dans l'épilogue de `down` est intrinsèquement non reproductible — le commentaire voisin (`:1778`) dit l'inverse : le port `_moe_fused` (b12x, `ACVRAM_MOE_DECODE_FUSED`) fait *« down par tranches, split-K sériel (bit-reproductible) »* par défaut ; seule sa variante `ATOMIQUE` (opt-in, jamais servie) ne l'est pas. **La bonne raison d'écarter reduce-dans-down pour le levier 3** : ce split-K sériel bit-reproductible n'existe QUE dans `_moe_fused`, sous conditions d'éligibilité étroites (`bt==16`, `pg[4]%128==0`, `pd[4]%_MOE_FUSED_TN==0`, …) que les chemins `marlin_c17`/plain (ceux que B/C/D touchent) ne remplissent pas — l'écrire pour eux serait reconstruire cette machinerie de split-K, hors scope d'un « épilogue » simple, et `_moe_fused` lui-même n'est jugé qu'« contre float64 », pas au bit (déjà exclu par Océane). **Conclusion identique à la sienne, motif à corriger dans la note** : reduce-dans-down écarté parce que la seule voie connue et bit-sûre est un chantier séparé déjà nommé hors levier 3, pas parce que fusionner un reduce serait en soi non reproductible.

## Par module

| module | retenu/écarté | µs que je tiens | pourquoi |
|---|---|---|---|
| **B** (`moe_act`+`quant_act(act)`) | **RETENU** | **−0,10 à −0,15 ms** (tenu) | 48 lancements × ≈2,5 µs/lancement (mon H2, `laurine-scelle-b12`) = 0,12 ms — au milieu de sa fourchette, calcul indépendant qui recoupe |
| **C** (`moe_route_pack`+`quant_act(xs)`) | **RETENU** | **−0,10 à −0,15 ms** (tenu) | même arithmétique, même géométrie (48 lancements, un amax+pack déjà par ligne dans le pack) |
| **D** (gate+up en une GEMM) | **RETENU, fourchette moins sûre** | −0,05 à −0,10 ms **plausible, pas tenu au même niveau que B/C** | même compte de lancements (48) que B/C mais gain prédit deux fois plus bas : la différence vient de ce que B/C suppriment un CALCUL (quant/pack, dont le travail disparaît avec le lancement) alors que D ne supprime QUE le lancement — le calcul des deux GEMM reste identique, un graphe capturé les enfile déjà dos à dos sans trou hôte (§ scellé M2, `verdict-m2-b12-21-09`) ; le seul gain réel de D est le décodage matériel du lancement, pas 2,5 µs pleins. À confirmer par un micro-banc avant d'écrire le noyau (≤ 5 min de carte), pas à prendre pour acquis dans la somme globale. |
| **reduce dans down** | **ÉCARTÉ, d'accord** | — | voir ci-dessus : motif corrigé, conclusion tenue |
| **`add_norm` dans l'épilogue de o/down** | écarté, d'accord sans réserve | — | formes différentes (résidu [t,H] contre sortie GEMM par tuile) — pas un épilogue simple, jamais discuté au bit par personne, correctement hors scope |

## Tests au bit qui manquent (à ajouter à son § 3, pas à retarder le go de Manon)
* **B** : son protocole (12 lots réels, t·k ∈ {12,48,96}) ne couvre PAS le cas `awq_d`/`hd_d` actifs (Hadamard/échelle AWQ sur `down_proj`) — Coder n'en a pas, mais le test doit soit refuser explicitement ce cas (assert AWQ/hadamard absents, sinon skip nommé), soit le couvrir : un modèle futur avec `down_proj` AWQ/Hadamard ferait silencieusement diverger l'épilogue fusionné sans qu'un test ne le voie.
* **C** : même trou côté `awq_g`/`hd_x` (`xs` quantifié) — même remède (refus explicite ou couverture).
* **D** : « mêmes MMA par tuile, même ordre d'accumulation par sortie » se prouve mal sur des entrées réelles (une divergence d'accumulateur croisé gate/up peut ne rien changer sur les valeurs typiques). Ajouter un cas ADVERSAIRE : `gate_proj` et `up_proj` posés à des poids délibérément DIFFÉRENTS par construction (pas de symétrie qui masquerait un accumulateur partagé) et vérifier `g`/`u` séparément au bit, pas seulement `act = silu(g)·u` en sortie — un bogue de croisement g/u pourrait rester invisible sur la sortie combinée dans certains cas dégénérés.

durée : 0 min de carte
suite : Océane : le motif corrigé pour reduce-dans-down (ci-dessus) n'engage rien à refaire — la décision reste bonne ; module D : poser le micro-banc AVANT le noyau, pas après ; B/C : ajouter le garde/skip AWQ-Hadamard nommé au test au bit. Manon : `familles-noyaux` d'abord, comme prévu par Océane.
