# Verdict — pièce 156 (d) : F1, F3, F6 du décodage Qwen3.8, opt-in (poste5, 24/09) — TENU ; prédictions de gain FAUSSES (trop basses)

* **instrument** : `scratchpad/poste5-p156d-24-09/prise.sh` — (tests) suite 156 d + GDN + table des variables, trois bras
  cassants dans une copie ; (abba) `frontiere-pas.py` 300 pas ; (kl) `kl-decode.py`, un processus par bras, moteur servi
  (pipeline, graphes), 8 fenêtres wiki par composition, 512 jetons forcés, log-probs complètes des 64 premiers pas
* **commit** : 408d9058 (tests, ABBA), 7c874a72 (tests avec le bras cassant F6 remplacé, KL ; seul `prise.sh` diffère)
* **régime** : `Qwen3.8-27B-nvfp4`, config par défaut (`ACVRAM_GEMV_LAYOUT=marlin`, F2/F4/F5 au défaut), ctx 2 048,
  invite 256, -lgc 2700 (horloge relevée 2 655-2 662), cpu-safe 100. Carte 0 seule à nous ; llama-server sur la 3080 Ti
  (bus 02) pendant toute la prise, présent sous tous les bras
* **scellé** : `revue/poste5-piece156d-scelle-24-09.md` (21f9a858, avant le code)
* **durée** : tests + ABBA 18:25:18-18:31:02 ; tests + KL 18:55:54-19:04:29

## Tests

201 verts (aucun sauté). Bras cassants : F1 `dt_bias` omis → ROUGE ; F3 poids oublié → ROUGE ; F6 (2e prise)
arrondi bf16 intermédiaire oublié → ROUGE sur les 21 formes.
**Faute d'instrument, corrigée avant la KL** : le bras cassant F6 écrit au scellé (somme d'un fil prise à l'envers)
est resté VERT. Ce n'était pas une faute : sur des entrées bf16, les ≤ 8 carrés d'un fil (16 bits de mantisse chacun)
se somment EXACTEMENT en fp32, dans n'importe quel ordre. Émulation : inverser l'ordre des warps change ss sur 65 % des
lignes mais 0 élément de sortie bf16 sur 932 k. L'ordre de la réduction n'est pas observable en sortie : le test juge
la sortie, qui est le contrat. Remplacé par une faute visible (7c874a72).

## Vitesse (pas GPU médian, µs)

| b | A1 | B1 | B2 | A2 | Δ B − A | prédit | seuil FAUX |
|---|---|---|---|---|---|---|---|
| 8 | 14 400,4 | 13 323,2 | 13 325,1 | 14 402,4 | **−1 077 (−7,5 %)** | −560 à −740 | > −390 |
| 1 | 13 049,8 | 12 301,3 | 12 301,3 | 13 049,9 | **−749 (−5,7 %)** | −320 à −460 | > −220 |

Passes seules à b=8 (une chacune, indicatif) : F1 14 053,3 (−348 ; prédit −200 à −260), F3 13 977,9 (−423 ; −210 à
−270), F6 14 097,3 (−303 ; −150 à −210). Somme −1 074 ≈ B −1 077 : les trois gains s'additionnent. Trou entre pas
inchangé (23-26 µs). A1 ≈ A2 et B1 ≈ B2 à 2 µs près.

**Gain tenu** (loin des seuils FAUX), **prédictions FAUSSES** : toutes trop basses, d'un facteur 1,4 à 2. J'ai compté
le gain comme la somme des durées des noyaux retirés (dossier 156 b, nsys). Or chaque nœud retiré d'un graphe rend
aussi son écart de lancement, que nsys ne compte pas dans la durée d'un noyau. C'est une hypothèse, pas une mesure : elle
se vérifierait par une trace nsys des deux bras (nombre de nœuds et mur du graphe).

## Qualité (KL contre A ; seuils = 2 × max des témoins mesurés dans la même prise)

Instrument valide : A2 = A au bit (log-probs et NLL) sur les deux compositions. T2 (ordre inverse) = A au bit.

| comp. | bras | KL_max | seuil | désaccords argmax (/512) | seuil | PPL fenêtre max | ΔNLL moy (z) |
|---|---|---|---|---|---|---|---|
| C1 | T1 (témoin) | 0,0803 | — | 7 | — | +0,38 % | +0,0002 (0,35) |
| C1 | B | 0,0261 | 0,161 | 9 | 14 | +0,41 % | +0,0008 (1,41) |
| C1 | F1 | 0,0121 | 0,161 | 6 | 14 | +0,21 % | −0,0003 (−0,51) |
| C1 | F3 | 0,0235 | 0,161 | 6 | 14 | +0,33 % | +0,0006 (1,01) |
| C1 | F6 | 0 (au bit) | — | 0 | — | 0 | 0 |
| C2 | T1 (témoin) | 0,0161 | — | 8 | — | +0,30 % | +0,0003 (0,27) |
| C2 | B | 0,0169 | 0,032 | 8 | 16 | +0,28 % | +0,0002 (0,34) |
| C2 | F1 | 0,0193 | 0,032 | 6 | 16 | +0,23 % | −0,0007 (−1,13) |
| C2 | F3 | 0,0071 | 0,032 | 8 | 16 | +0,31 % | +0,0006 (0,97) |
| C2 | F6 | 0 (au bit) | — | 0 | — | 0 | 0 |

* **KL, argmax, PPL : tenus** pour B, F1 et F3 dans les deux compositions ; aucun |z| ≥ 2 ; signes mêlés.
* **F6 : AU BIT sur le modèle servi** (16 fenêtres, 512 pas), comme visé.
* Prédictions qualité : « KL ≤ T1 » FAUX en C2 pour F1 (0,0193 > 0,0161) et pour B (0,0169) ; « argmax < T1 » FAUX en
  C2 pour F3 (8 = 8) ; « |ΔPPL| < 0,1 % » FAUX par fenêtre (0,2-0,4 %). Le témoin T1 fait pourtant autant (+0,38 % et
  +0,30 %) : sur 512 jetons, une fenêtre bouge de quelques dixièmes de pour cent pour un seul ulp en amont. Conséquence :
  **le seuil fixe de +0,5 % par fenêtre n'a qu'une marge de 0,1 point au-dessus du témoin**. Il ne départagerait pas
  une fusion réellement nocive d'un simple changement de chemin déjà servi (leçon de la 150 bis, dans l'autre sens).

## Verdict

* **TENU** : les trois fusions, seules et ensemble, dans les seuils de qualité posés contre les témoins ; F6 au bit ;
  gain **−1,08 ms/pas à b=8 (+8,1 % de débit de décodage)** et **−0,75 ms/pas à b=1 (+6,1 %)**.
* **Prédictions de vitesse fausses**, toutes sous-estimées. Cause supposée : le coût de nœud des graphes, non compté.
* **Proposition au chef** : F6 au défaut sans condition de qualité (au bit : test et rejeu). F1 et F3 au défaut sur ton
  mot. Avant toute bascule : suite complète sous verrou et capture des godets 1 à 8 (conditions de la 156 c).
* Limites : Qwen3.8-27B-nvfp4 seul (mixte et Qwen3.5 non mesurés, même code) ; vitesse du préfill non mesurée (F3 et F6
  y passent, la KL les couvre) ; une seule passe par fusion seule.

## Addendum 24/09 19 h 3x — F6 au défaut (décision de chef), conditions écrites AVANT la prise

F6 passe à 1 par défaut (0 = témoin) ; F1 et F3 restent en opt-in (décision de l'utilisateur, soumise par chef).
Conditions avant le push :
1. **Suite complète** sous mon verrou, HEAD contre la base origin/main (e5c95667) dans un worktree détaché : FAUX si un
   échec existe sur HEAD et pas sur la base. Prédit : différence vide.
2. **Capture des godets 1 à 8** (`capture-godets.py`, Qwen3.8, défaut) : capture ok, `graphes=on`, 0 repli eager pour
   chacun. Prédit : 8/8, ms/pas ≈ 0,30 de moins que la 156 c à chaque godet (13,05 → ≈ 12,75 à b=1, si la config
   Marlin qualifiée d'alors vaut le défaut d'aujourd'hui, ce que je ne sais pas : chiffres relevés, pas jugés).
