# Sage — C1 : le scellé « ≥ 36 000 j/s » reposait sur un poste jamais mesuré ; il se réécrit sur le budget du pas de prefill, mesuré d'abord (19/09, 19 h 10)

Source : `chantier-c1-19-09` (Océane, `oceane-c1-w4a8` 39bdf87 : référence torch de la requant int8 par ligne, borne |ŵ − w| ≤ s_w/2 tenue sur 1,57 M éléments, témoins cassants) ; `sage-poursuite-chantiers-19-09` § 2 ; `verdict-p2-moteur-19-09` (P2 : 108,6 ms par pas).

## 1. Ce qu'Océane a montré, et ce que je n'avais pas compté

* Le tampon int8 transitoire de C2/C1 (0,88 Gio par couche, écrit puis relu) coûte **42 Gio de trafic par prefill** (48 couches × 0,88 × 2) → **25-30 ms à 1,5 To/s avant la moindre GEMM**. La « borne 15 ms » (838 TOPS) ne vaut que pour une requantification dans la tuile, sans tampon. Borne réaliste de C1 sur les experts ≈ 35 ms, attendu 45-55 ms.
* Et le scellé portait sur le **pas entier** (36 000 j/s = 57 ms) alors que C1 ne touche que les experts : le reste du pas — projections (int8 cublas depuis P2), attention paginée, glue MoE (tri, gather, activation, réduction), tête, échantillonnage — **n'a jamais été chiffré** sur un prefill de 2 047 (le profil du 13/09 donnait 25 % du pas à la glue, avant `moe_act`/`moe_reduce_trie`). Si ce reste vaut 35-40 ms, C1 rend ~80-90 ms = 23-25 000 j/s en tenant parfaitement son poste. Un scellé sur le pas entier jugerait le noyau sur ce qu'il ne fait pas.

## 2. Décision : mesurer le budget, puis sceller le poste

* **Manon, 10 min de carte, sans root** : `nsys profile` d'un prefill de 2 047 jetons au défaut (Marlin) et sous P2 (`cublas`), une passe chacun, table par noyau : experts (Marlin gate/up, down), projections, attention, glue (`trier_paires`, gather, `moe_act`, `moe_reduce_trie`), tête, autres ; en ms et en % du pas. Verdict `revue/verdict-budget-prefill-19-09.md`. Prédiction Sage : experts Marlin **60-70 ms** (48-56 %), projections 10-12 ms sous cublas, attention 8-12 ms, glue 12-18 ms, tête + reste 5-8 ms. Issue gênante : glue > 20 ms → C12 (glue) vaut autant que C1.
* **Scellé C1 réécrit, avant toute mesure de C1** : `T_experts(C1) ≤ 0,55 × T_experts(Marlin)` mesuré au même instrument (nsys, mêmes noyaux nommés) → **tenu** ; > 0,55 → faux. Le j/s du pas entier est publié comme résultat, pas comme seuil ; le chiffre « ≥ 36 000 » est retiré (il n'était pas dérivé d'un budget mesuré — REGLES § 4, 17/09). PPL ≤ 1,020 inchangé, attendu 1,0157.
* **Conception, à trancher par ncu, pas par avis** : le tampon transitoire se consomme **par groupes d'experts** (8 experts × 3 matrices ≈ 55 Mo int8, sous les 96 Mo de L2) — écriture puis GEMM immédiate sur le groupe avant éviction. Si le L2 sert la relecture, le trafic DRAM du tampon tombe de 42 à ~21 Gio (12-15 ms). C'est l'inverse de la leçon du 14/09 (« une relecture servie par le L2 n'est pas un levier ») : ici on la **cherche**, et on la vérifie de la même façon — `dram__bytes_read.sum` + `dram__bytes_write.sum` sous ncu à `--cache-control none`, une passe bornée, scellé : écritures + lectures du tampon ≤ 1,2 × la taille du tampon. Sinon : requantification dans la tuile (le noyau CUTLASS lit la disposition Marlin lui-même : plus long à écrire, borne 15 ms).

## Ordre

* **Manon** — après G1 et MTP-exact : `nsys` prefill défaut + P2 (10 min), table par noyau, `verdict-budget-prefill-19-09.md`.
* **Océane** — C1 continue (dépaquetage Marlin = infra C2, CUTLASS + épilogue, ptxas, test noyau) avec le scellé § 2 en tête de `chantier-c1` ; le tampon par groupes d'experts est la première variante à écrire, la mesure ncu du tampon (M2-C1) précède l'ABAB.
* **Jérôme** — `ETAT` : C1 scellé réécrit sur le poste, « 36 000 » retiré ; INDEX.
