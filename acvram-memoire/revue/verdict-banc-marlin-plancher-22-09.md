# banc-marlin-plancher (poste1 660b4a36), synthétique + routages réels — TENU littéralement, mais 8 couches PERD au lieu de gagner — 22/09 (poste2)

* instrument : `outils/gpu/mesure/banc-marlin-plancher.py --D 42 --b 12 --rep 200`, sous carte.sh, deux passes (routage synthétique D=42 ; routage réel `scratchpad/experts-distincts-19-09/routages-b12.pt`, D effectif=48)
* commit : main à jour
* scellé (poste1) : gateup/down isolés 1,45-1,55 To/s ; 8 couches = couche ± 1 µs ; L2 ≤ 2 % ; réfuté si couche ≥ 1,55 (rien à faire dans Marlin) ou 8 couches gagne > 3 µs
* mesuré :

| poste | synthétique (D=42) | réel (D=48) |
|---|---|---|
| gateup | 37,0 µs · 2,011 To/s · **112 % du pic** | 39,9 µs · 2,127 To/s · **119 % du pic** |
| down | 23,3 µs · 1,595 To/s · 103 % plancher | 23,2 µs · 1,828 To/s · **102 % du pic** |
| couche | 74,7 µs · 1,493 To/s · 96 % plancher | 82,8 µs · 1,539 To/s · 99 % plancher |
| 8 couches/couche | 99,8 µs · 1,117 To/s · 72 % plancher (**−25,1 µs/couche, PERTE**) | 98,4 µs · 1,294 To/s · 84 % plancher (**−15,6 µs/couche, PERTE**) |
| L2 froid | 118,0 µs · 0,945 To/s · 61 % plancher | 117,7 µs · 1,082 To/s · 70 % plancher |

* verdict : **aucune des deux alarmes codées ne se déclenche** (couche reste sous 1,55 sur les deux passes ; 8 couches ne « gagne » jamais >3µs — il PERD, dans les deux passes, -25,1 et -15,6 µs/couche) — donc « non réfuté » au sens strict, mais **la prédiction elle-même (8 couches = couche ± 1 µs) est contredite dans le sens négatif**, pas confirmée : grouper 8 couches consécutives coûte plus cher par couche que les isoler, cohérent sur synthétique et réel. **Anomalie nommée** : gateup et down isolés dépassent le pic théorique (1,79 To/s) de 2 à 19 % sur les deux passes — signe d'un artefact de cache L2/L1 chaud par répétition du même routage sur 200 essais consécutifs (octets réutilisés, pas relus depuis la HBM à chaque rep) ; ces deux chiffres ne sont donc pas crédibles comme plancher isolé, seuls « couche » (proche du plancher, 96-99 %) et surtout « L2 froid » (61-70 % du plancher, le plus proche d'un service réel) le sont. Écart réel/synthétique faible (D=48 réel vs D=42 synthétique, mêmes ordres de grandeur).
* durée : ~1,5 min de carte (2 × ~200 reps)

## Suite
Le levier « Marlin mieux occupé » (§1 du poste `experts_marlin`, plafond 0,3-0,6 ms/pas) n'a pas de marge visible sur le régime chaud (couche déjà à 96-99 % du plancher) — la marge probable est côté L2 froid (61-70 %), à rapprocher du service réel (pas de répétition du même routage). Carte rendue.
