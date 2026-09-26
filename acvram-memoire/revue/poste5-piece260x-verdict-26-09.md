# Pièce 260x — copie signée xor au défaut : VERDICT (poste5 26/09 07 h)

Scellé `poste5-piece260x-scelle-26-09.md` (8c28d63dd, poussé avant mesure). Prise `poste5-p260x` sur 8c28d63dd, 06:44-06:57,
Coder -qkvo-i8c servi, A (int16) B (xor) ×4 en A B B A A B B A, serveur neuf par passe, -lgc 2700, ACVRAM_ECO=off ; carte 0
sans PID hors prise avant/après. Brut : `scratchpad/poste5-p260x-26-09/mesures.jsonl`, `prise.log`. Tests dans la prise : verts.
Preuve de prise : `regime_ligne` A porte `ACVRAM_I8C_COPIE=int16`, B non ; `int8_chemins` : cublas 5 328, i8c_fabrique ≈ 20 400
par passe dans les deux bras.

| TTFT b=1 (médiane de 40, ms) | A (int16) ×4 | B (xor) ×4 | B − A (médianes) | étendue A | seuil |
|---|---|---|---|---|---|
| L = 78 (contrôle, GEMV) | 36,70-37,69 | 36,71-37,05 | −0,24 (−0,6 %) | 0,99 | \|Δ\| ≤ 2 % : **tenu** |
| **L = 512** | 46,67-48,35 | 44,95-45,56 | **−1,98** | 1,68 | ≥ 2,5 et > 3,36 : **NON TENU** |
| **L = 2047** | 101,36-101,42 | 98,71-98,83 | **−2,62** | 0,06 | ≥ 2,5 et > 0,12 : **tenu** |
| banc chat b=8 (t/s, contrôle) | 1 595,6-1 605,6 | 1 602,3-1 608,2 | +0,20 % | — | \|Δ\| ≤ 1 % : **tenu** |

* **Seuil scellé (≥ 2,5 ms aux DEUX longueurs) : NON TENU** à L = 512 (−1,98 ms). Prédiction −3,5 à −5,5 ms : **FAUSSE** aux deux
  longueurs (gain réel 2,0-2,6 ms ; mon estimation de 5,3 µs par M de poids venait des grandes formes du mixte, pas des
  petites du Coder).
* Le gain est RÉEL : à L = 512 et 2047, les quatre médianes de B sont toutes sous les quatre de A (distributions disjointes).
  Au bit (tests). Jamais plus lent (contrôles tenus).
* Réserve : `fenetre_valide` = false partout — elle juge la moyenne d'ÉNERGIE (fenêtre 1,5-4 s < 10 s, bridage puissance),
  pas le TTFT (médiane de 40 requêtes) ; aucun chiffre d'énergie n'est revendiqué.

Issue nommée (i) du scellé, écrite avant mesure : « gain < 2,5 ms → reste au défaut quand même (au bit, jamais plus lent),
sans revendication ». Décision à chef (sa règle : 260x FAUSSE → défaut int16).

**DÉCISION DE chef (26/09 07 h)** : le scellé fait foi (issue (i) écrite AVANT la mesure ; la règle « int16 si FAUX » est
venue après). Verdict FAUX par la lettre, prédiction fausse, écrits tels quels ; **la copie xor RESTE au défaut, sans
revendication de gain** (au bit, jamais plus lente, 4/4 médianes sous A aux deux longueurs).
