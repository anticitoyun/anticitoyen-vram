# Verdict — pièce 165 : GDN_PREFILL_LOT=1 requalifié contre témoins (poste5, 24-25/09) — NON TENU (qualité) ; TTFT −16 à −19 %

* **instrument** : `scratchpad/poste5-p165-24-09/kl-lot165.py` (kl-lot de la 150 bis, A/B basculés à chaud, T1 seules,
  T2 ordre inverse, textes neufs) ; `ttft-charge.py` (8 requêtes simultanées, 12 tours, A B B A A B B A A B, serveur neuf) ;
  `prise.sh` → `prise-kl.txt`, `prise-ttft.txt`, `kl-*.json`, `ttft.jsonl`
* **commit** : 73abfc20 (KL), 3f1e30c9 (TTFT ; seul `ttft-charge.py` diffère, voir l'ajout au scellé)
* **régime** : défaut du soir (Marlin `seuls=305`, F1-F6) ; TTFT : -lgc 2700, `--no-prefix-cache`, 10/10 passes
  `repli_eager=0`, B porte `GDN_PREFILL_LOT=1` à sa ligne de régime ; 0 passe nulle ; Qwen3.5-35B : disposition Marlin
  refusée au chargement (31 échelles sous-normales dans down_proj, repli nommé, 157), deux avertissements de l'allocateur
  sur les logits de ~0,8 Go, réessayés (rejeux au bit)
* **scellé** : `revue/poste5-piece165-scelle-24-09.md` (critère de chef, écrit avant)
* **durée** : KL 23:43:32-23:44:5x ; TTFT → 00:18:39

## Qualité (critère : KL_max ≤ 2 × max(T1, T2) ; accord argmax A/B ≥ A/T1 − 0,5 pt ; |ΔPPL| par séquence ≤ 2 × max témoins)

| modèle | comp. | KL_max B / T1 / T2 | accord argmax B / T1 / T2 | max\|ΔPPL\| B / T1 / T2 | tenu |
|---|---|---|---|---|---|
| Qwen3.8 | C1 8×78 | 0,0079 / 0,0047 / 0 | **98,38** / 99,19 / 100 % | 0,73 / 0,54 / 0 % | **NON (argmax −0,81 pt)** |
| Qwen3.8 | C2 | 0,0050 / 0,0252 / 0,0005 | 98,96 / 99,09 / 100 | 0,45 / 0,71 / 0,13 | oui |
| Qwen3.8 | C3 | 0,0122 / 0,0191 / 0 | 98,89 / 99,26 / 100 | 0,32 / 0,21 / 0 | oui |
| Qwen3.5-35B | C1 | **0,319** / 0,146 / 0,057 | 94,32 / 93,67 / 96,59 | 2,63 / 2,49 / 2,31 | **NON (KL 0,319 > 0,292)** |
| Qwen3.5-35B | C2 | **1,240** / 0,568 / 0,594 | 94,65 / 94,65 / 96,09 | 3,25 / 3,31 / 1,53 | **NON (KL 1,240 > 1,187)** |
| Qwen3.5-35B | C3 | 1,208 / 1,659 / 0,377 | **94,59** / 95,33 / 94,96 | 2,06 / 0,84 / 1,84 | **NON (argmax −0,74 pt)** |

Rejeux A et B au bit partout. Qwen3.8 : T2 au bit en C1 et C3 ; Qwen3.5-35B : T2 jamais au bit (MoE).

## TTFT servi sous charge (Qwen3.8 ; TTFT du tour = le plus lent des 8 ; médiane par passe, moyenne de 5 passes)

| comp. | A (σ) | B (σ) | B/A | prédit | FAUX si |
|---|---|---|---|---|---|
| C1 8 × 78 | 431,5 ms (1,8) | 351,1 (1,1) | **−18,6 %** | −15 à −25 | > −8 % |
| C2 mêlées | 466,6 (2,1) | 393,6 (1,1) | **−15,7 %** | −12 à −22 | > −8 % |

Tenu et dans la prédiction, A dans sa fourchette (380-480). Le gain du forward mesuré par poste6 (−18/−19 % sous
Marlin) se retrouve presque entier dans le TTFT servi.

## Verdict

* **NON TENU** au critère de qualité, sur les deux modèles (4 cellules sur 6). Ma prédiction (tenu 6/6) était fausse.
  L'issue nommée d'avance (accord d'argmax de C1 sur Qwen3.8 à plus de 0,5 pt sous T1) s'est produite : B 98,38 %
  contre T1 99,19 %. Sur le MoE, la KL de B dépasse 2 × témoins en C1 et C2, de 9 % et 4 %.
* **Vitesse tenue** : TTFT sous charge −18,6 % (C1) et −15,7 % (C2).
* **Conséquence** : `GDN_PREFILL_LOT` reste opt-in. La décision revient à chef ou à l'utilisateur, qui disposent des
  deux côtés : qualité hors critère, de peu, et gain de TTFT réel.
* **Lecture** : l'écart de B ne vient pas des GDN seuls. B change le M des projections (Σ t au lieu de t), donc le noyau
  (GEMV ou GEMM, Marlin par M) et l'ordre des sommes de TOUTES les projections GDN, alors que T1 change le chemin
  entier. À noter : sur le MoE, B et T1 ont des PPL au même niveau (2,6 contre 2,5 % en C1, 3,3 contre 3,3 % en C2).
