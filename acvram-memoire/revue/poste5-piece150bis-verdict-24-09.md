# Verdict — pièce 150 bis : préfill GDN du lot (poste5, 24/09) — NON TENU à la lettre

* **instrument** : `scratchpad/poste5-p150bis-24-09/kl-lot.py` (A/B basculés à chaud, même processus, témoins T1-T3) ;
  banc chat de la 150 (`banc-service.py`, 3 lots de 8 × 256, invite 78 jetons), serveur b=8, ordre A B B A ; `prise.sh`
* **commit** : d194aafe (branche poste5), `acvram.__file__` = worktree ; tests 17 verts sous mon verrou
* **régime** : -lgc 2700, cpu-safe 100, ACVRAM_ECO=off ; défaut `Qwen3.8-27B-nvfp4` ; mixte sous PROJ_MARLIN.
  Carte 0 (5090, bus 01) seule à nous ; un llama-server (pid 4627, 5,6 Gio) tournait sur la 3080 Ti (bus 02) pendant
  les deux prises, présent sous les quatre bras de chaque ABBA
* **scellé** : `revue/poste5-piece150bis-scelle-24-09.md` (écrit avant, précision KL de chef ajoutée avant la prise)
* **durée** : défaut ≈ 13:1x-13:21:33, mixte → 13:26:24 ; prévu ≤ 41 min au total, tenu

## Vitesse (préfill par lot, `/metrics`)

| alias | A1 | B1 | B2 | A2 | Δ B − A | prédit | seuil | débit A → B |
|---|---|---|---|---|---|---|---|---|
| défaut | 0,4119 s | 0,3411 | 0,3403 | 0,4173 | **−0,074 s/lot** | 0,22-0,32 s/lot (−0,09 à −0,19) | −0,08 | 293,3 → 296,8 t/s (+1,2 %) |
| mixte | 1,3627 s | 0,5518 | 0,5518 | 1,3658 | **−0,812 s/lot** | 0,45-0,8 s/lot | −0,4 | 293,3 → 333,4 t/s (**+13,7 %**) |

Pas de décodage inchangé (25,5 et 21,7-21,8 ms/pas sous les deux bras). Mixte : 6 → 5 pas de préfill pour 3 lots.

## Qualité (KL max par position contre A-lot ; seuil = 2 × max des témoins)

| alias | comp. | KL_max(A‖B) | T1 seul | T2 ordre inverse | T3 int8 GEMM | seuil | argmax A/B | ΔNLL (z) |
|---|---|---|---|---|---|---|---|---|
| défaut | C1 8×78 | 0,0034 | 0,0065 | 0 | — | 0,0130 | **98,24 %** | −0,0018 (−1,65) |
| défaut | C2 mêlées | 0,0061 | 0,0117 | 0 | — | 0,0233 | **98,32 %** | −0,0002 (−0,18) |
| défaut | C3 | 0,0239 | 0,0324 | 0 | — | 0,0649 | 99,63 % | −0,0004 (−0,39) |
| mixte | C1 | 0,0047 | 0,0047 | 0 | 0,0060 | 0,0120 | 99,68 % | −0,0014 (−1,26) |
| mixte | C2 | 0,0636 | 0,0356 | 0 | 0,0049 | 0,0713 | **98,19 %** | +0,0008 (+0,48) |
| mixte | C3 | 0,0361 | 0,0914 | 0 | 0,0009 | 0,1828 | **98,90 %** | +0,0019 (+1,36) |

Rejeu A et B : 0 partout. A ≠ B au bit sur les deux alias. T2 = 0 partout (l'ordre du lot ne change rien au bit).

## Verdict

* **KL : tenue** sur les 6 cellules (KL_max(A‖B) ≤ 2 × max des témoins ; sous T1 seul dans 5 cas sur 6, mixte C2 à
  1,8 × T1). ΔNLL : aucun |z| ≥ 2, signe mêlé.
* **argmax ≥ 99 % : NON tenu sur 4 cellules sur 6** (98,19-98,90 %). Ce seuil, je l'ai écrit sans témoin : l'accord
  d'argmax de T1 (séquences seules contre lot, servi aujourd'hui) n'a pas été relevé. Je ne le reconstruis pas. Le
  critère est donc manqué à la lettre, et rien ne dit encore si 98 % est au-dessus ou au-dessous de ce que le préfill
  groupé déjà servi fait lui-même.
* **Vitesse défaut : NON tenue** (−0,074 contre −0,08 s/lot). Prédiction fausse : j'attendais 0,22-0,32 s/lot, mesuré
  0,34. Aucune issue nommée n'est prise (entre (a) < −0,05 et le seuil) : le reste du préfill du défaut (≈ 0,34 s pour
  ~1,7 passage) n'est pas dans les projections GDN, et les règles delta fla par séquence restent le premier suspect.
* **Vitesse mixte : TENUE** et dans la prédiction (0,55 s/lot, +13,7 % de débit au banc, prédit +8 à +14 %). Les GEMV
  int8 par canal des couches GDN étaient bien la cause du 1,37 s de la 150.
* **Décision** : l'opt-in reste à 0 par défaut. Aucune proposition de défaut.

## Ce qui trancherait (pas lancé : aucune nouvelle pièce sans le mot de chef)

1. Relever l'accord d'argmax de T1 (et de T3) dans la même prise. Si le préfill groupé déjà servi descend lui-même sous
   99 %, le seuil scellé était mal posé, et la qualité se lit sur la KL, qui tient.
2. Défaut : fla en varlen (`cu_seqlens`, une règle delta pour le lot) pour le reste du préfill.
