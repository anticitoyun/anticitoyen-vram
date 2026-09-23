# Sage — C16-bis réfuté (hôte 0,35-0,38 ms) et « 30 ± 8 » réfuté (45,4) ; le vrai poste : le même graphe fait 7,0 ms en rafale et 8,6 ms sous 400 W soutenus — le pas servi b=12 est borné par la puissance, donc l'énergie des experts EST la vitesse ; Mesure 1 à l'unité 45 décide tout (19/09, 19 h 46, heure du commit)

Source : `verdict-experts-distincts-c16bis-19-09` (Manon 1853343) ; `verdict-nsys-coder-b12-19-09` ; `verdict-ncu-m1-m2-19-09` ; `verdict-eco-lgc-b12-19-09` (E1) ; `sage-nsys-coder-c16-mma2-19-09` et son addendum.

## 1. Deux prédictions réfutées, tenues pour réfutées
* **Experts distincts par couche à b=12, invites réelles : 45,4** (médiane 45, σ 5,3, 26-64 ; couches 0-4 à 51-55, 45-47 à 36-38). Mon « 30 ± 8 » (chiffre du 14/09 jamais remesuré) est faux ; le synthétique 27,4 sous-estimait de 40 %. Conséquence arithmétique : Marlin déplace **45,4 × 48 × 2,655 Mo = 5,8 Go en 3,73 ms = 1,55 To/s** — au plafond DRAM de la carte (copie 1,5) ET au plafond de puissance (398 W). mma2 n'a été vu qu'à 27 distincts, 0,99 To/s : **à 45 il ferait 5,9 ms si sa bande ne monte pas avec le nombre de blocs** — l'unité 45 n'est pas un détail, c'est la question.
* **C16-bis : hôte 0,38 ms (certifie) / 0,35 ms (rafale), 4 %** — réfuté (≥ 1,0 prédit) ; C16 est mort sous ses deux formes. L'instrument est juste : les rejeux lus dans le SQLite nsys `graph-trace` rendent le pas et le hors-GPU exacts (à écrire dans REGLES § 6 à côté du « temps mural ×6 faux » du `node-trace`).

## 2. Ce que Manon a trouvé à la place : 19 % du pas servi est le bridage
Le même graphe : **7,00 ms en rafale** (horloge libre) et **8,59 ms sous 400 W soutenus, SM 2 524 MHz moyens**. Le pas servi à b=12 n'est borné ni par l'hôte ni par les octets seuls : il est **borné par la puissance** — Marlin (54 % du temps, 398 W, 1,3-2,6 inst/octet le 14/09) est co-limité par ses instructions, et l'horloge qui tombe ralentit tout le pas, glue et projections comprises. Cela relit E1 : un 2 700 plat évite l'oscillation boost/bridage, vitesse égale à −10 % de J. Et cela change la valeur de Mesure 1/2 : **un noyau d'experts sous le plafond ne gagne pas seulement des joules, il rend au pas entier jusqu'à 19 % de vitesse** — ou il perd 40 % s'il est affamé en bande à 45 distincts. Les deux issues sont écrites avant la mesure :

| Mesure 1, unité 45 (et 27 en second point), boucle soutenue ≥ 6 s, rapport cyclique publié | Marlin gate·up + down | mma2 gate/up/down + quant_act |
|---|---|---|
| t par couche (prédit) | **78 µs** (nsys) | **90-125 µs** (0,99 → ≤ 1,3 To/s selon les blocs en vol) |
| W (prédit) | 398 | **280-340** |
| Go/s | 1,5 | 1,0-1,3 |

Règle écrite maintenant : **Mesure 2 seulement si t_mma2(45) ≤ 1,15 × t_marlin(45) ET W_mma2 ≤ 350**. Mesure 2 = `certifie` soutenu, ABAB contre défaut, `GEMV_LAYOUT=naturel` + prefill nommé, cellule prefill du régime dans la même fenêtre : prédiction **t/s 1 300-1 600** (le bridage levé contre le noyau plus lent), **J net 0,17-0,20** ; scellé inchangé : J net ≤ 0,205 ET t/s ≥ 1 290. Si t_mma2(45) > 1,15 × t_marlin(45) : mma2 est affamé en bande → **C17 change d'objet** : « mma2 à ≥ 1,3 To/s à 45 distincts » (étages cp.async, blocs par expert) avant toute disposition, et Marlin reste — pas cette nuit.

## 3. Ce qui engage l'utilisateur, à porter par Jérôme, pas à décider ici
Le pas servi est bridé à 400 W par choix de plafond ; le 2 700 plat rend la même vitesse pour −10 % de J (E1, E1-bis : b=1 −2 %). **Question à l'utilisateur** : `acvram eco 2700` comme régime servi par défaut de la 5090 (engage : `sudo nvidia-smi -lgc`, horloge de la carte, autres usages) ; tant qu'il n'a pas dit oui, le défaut reste « off » et les cellules publiées le disent. Toute cellule b=12 porte désormais **l'horloge SM moyenne du pas** dans son en-tête (nsys ou `nvidia-smi dmon`), pas seulement le plafond : deux cellules à horloges moyennes différentes ne se comparent pas.

## Ordre
* **Océane** — **Mesure 1 en premier créneau, avant le contrôle (b)** : unité 45 puis 27, pour chaque noyau t/couche, W, rapport cyclique, Go/s (§ 2), boucle ≥ 6 s soutenue ; verdict deux lignes ; puis la règle du § 2 dit si Mesure 2 a lieu ; C17 en fiche seulement après, avec son objet fixé par le résultat.
* **Manon** — Mesure 2 selon la règle (15 min, cellule prefill du régime dans la fenêtre, horloge SM moyenne en tête) ; en-tête de toute cellule b=12 : horloge SM moyenne ; C13-b quatre bras comme lancés ; file inchangée sinon.
* **Jérôme** — ETAT : C16 fermé (deux formes), 45,4 distincts (unité Mesure 1), pas servi borné par la puissance (7,0 / 8,6 ms), règle de Mesure 2, C17 conditionnel ; **question à l'utilisateur** : éco 2700 par défaut ; REGLES § 6 (graph-trace exact) et § 3 (horloge SM moyenne dans l'en-tête b=12) ; INDEX ; commit + push.
