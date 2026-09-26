# Verdict — pièce 141 : PDL (lancement dépendant programmatique) sur les GEMV int8 du Coder à b=1 (poste6, 24/09)

instrument : cellule b=1 en service = celui des cellules b=1 publiées (README ³) : `acvram serve --max-batch 1 --max-model-len 2304 --speculative none` + `scratchpad/banc-llamacpp-16-09.py decode` (BANC_SLOTS=1×5, 1 024 jetons, fenêtre ≥ 10 s, énergie nvml nette) ; `scratchpad/poste6-p141-24-09/prise-b1-http.sh` (ABBA par relance du serveur), `prise-preuve.sh` (état PDL lu par `/metrics` dans le processus servi) ; sorties `prise-b1-http-{1,2}.txt`, `banc-b1-<n>-<bras>.log`, `serveur-b1-*.log`, `horloge-b1-*.csv`, `metrics-preuve-{A,B}.json`
commit : a6d30f03 pour les 10 lots (= main fc049d67 avec garde d'import a86fa1dd + 141 A + A'), 8ccbb209 pour la preuve (`/metrics` expose `pdl`, seul changement) ; HEAD asserté rc 65 ; `acvram.__file__` = travail/poste6 (vérifié)
régime : Coder qkvo-i8c, b=1, `-lgc 2700` posé pour chaque prise (horloge SM lue pendant les fenêtres 2 679-2 688), 268-277 W, RTX 5090 seule ; A = `ACVRAM_PDL=0`, B = `ACVRAM_PDL=1` (A + A' : GEMV int8 lancés avec l'attribut programmatique, poids préchargés avant l'attente de grille, déclenchement précoce dans moe_reduce, rmsnorm ×2, attention paginée Triton) ; cpu-safe=off (max_perf_pct 100 début et fin des deux prises) ; compute-apps début = fin = llama-server 4627 ; **preuve dans le processus servi** : `/metrics` du bras B après décodage → `pdl: on(1481, déclencheurs=5897)`, `ACVRAM_PDL=1` sur la ligne de régime ; bras A → `pdl: off`, `ACVRAM_PDL=0`
scellé : `revue/poste6-piece141-scelle-24-09.md` (283965c5 … a6d30f03, addenda avant chaque mesure) — TENU si Δpas ≤ −0,10 ms/pas, FAUX si > −0,05 ; prédit −0,14 à −0,24 ms/pas, J/jeton −4 à −7 %
mesuré : 10/10 lots à 5 fenêtres (prise 1 07:54-08:07 A B B A A, prise 2 08:27-08:40 B B A A B ; la prise 07:42 est écartée : fenêtre contaminée 07:31-07:50 et 0 lot rendu). Médianes de 5 lots : **A 309,3 t/s (3,233 ms/pas), 0,6534 J/jeton net, 275,8 W · B 310,1 t/s (3,225 ms/pas), 0,6530 J/jeton, 275,3 W**. Δpas **−0,008 ms** (−0,26 %), débit +0,26 %, J/jeton −0,06 %. Dispersion : A 308,7-309,4, B 309,6-310,2 (B au-dessus de A sur 5/5 lots, écart 0,8 t/s)
verdict : **FAUX** (Δpas −0,008 ms ≫ −0,05) — en service réel, le PDL avec déclenchement précoce ne rend rien de mesurable au pas b=1 ; le gain de −1,8 µs par paire du jouet (producteur à 1 bloc, poids chauds en L2) ne se transpose pas ; A et A' restent opt-in, défaut inchangé au bit (tests 3 passed)
durée : prises 1 et 2 : 13 min 18 + 13 min 26 (`tenue=` 798 s et 806 s), preuve 2 × 75 s ; prise nulle 07:42 (5 min) ; trois compilations de l'extension sous verrou (155, 167, 158 s)

## Chiffres par lot (médiane de 5 fenêtres)
| lot | bras | t/s | J/jeton net | W |
|---|---|---|---|---|
| 1 | A | 308,7 | 0,6433 | 269 |
| 2 | B | 310,1 | 0,6533 | 275 |
| 3 | B | 310,2 | 0,6530 | 276 |
| 4 | A | 309,3 | 0,6534 | 277 |
| 5 | A | 309,4 | 0,6553 | 278 |
| 6 | B | 309,6 | 0,6435 | 269 |
| 7 | B | 310,1 | 0,6511 | 275 |
| 8 | A | 309,3 | 0,6539 | 276 |
| 9 | A | 309,3 | 0,6519 | 275 |
| 10 | B | 310,3 | 0,6536 | 277 |

## Lecture
* **Ce que le jouet mesurait et ce que le pas ne fait pas.** Sur le jouet, le primaire était un noyau à UN bloc qui déclenche
  aussitôt : la carte est vide, le GEMV monte et précharge ses 10,5 Mo pendant les 3 µs d'attente — −1,8 µs. Dans le pas réel
  les primaires (attention paginée, `moe_reduce`, norme) déclenchent tôt aussi (5 897 déclenchements comptés), mais ils
  OCCUPENT la carte pendant qu'ils tournent : le GEMV qui monte ne trouve ni SM libre ni bande HBM libre à recouvrir, et le
  gain se réduit à la latence de lancement — que le rejeu de graphe cache déjà. 96 GEMV × ~0,1 µs = 0,01 ms : c'est le −0,008
  mesuré, à la résolution près.
* **Le préchargement des poids ne rend rien non plus** : 0,7 µs de latence DRAM par GEMV masquée aurait donné ≥ −0,05 ms. Soit
  la latence n'est pas sur le chemin critique du pas (le GEMV la paie pendant que le primaire finit de toute façon), soit les
  requêtes préchargées font la queue derrière celles du primaire. Un profil nsys du pas B contre A (une couche, 3 noyaux) le
  dirait ; ce n'est pas dans cette pièce.
* **Ce qui reste vrai** : l'arête programmatique est honorée sous graphe (jouet + compteurs), le code est au bit et opt-in, le
  bras cassant rend faux quand l'attente est déplacée — la mécanique est prouvée, son bénéfice sur ce pas est nul. Levier B
  (GEMV persistant) : non engagé (« seulement si A tient », ordre) ; sa promesse était −0,05 à −0,14 ms sur la montée de grille
  que ce résultat montre déjà absorbée par le graphe.
* Contrôles tenus : horloge 2 679-2 688 aux 10 lots ; 5 fenêtres valides par lot ; aucun repli PDL (compteur 0) ; serveur
  NOMINAL ; `ACVRAM_PDL` prouvé dans le processus servi par `/metrics`.
* Faute d'instrument consignée : `certifie-b12 CERT_PUR=1` ne tient pas 20 s à b=1 sur ce modèle (contexte épuisé), et la ligne
  de régime du chargement dit « inerte » avant la capture — `/metrics` porte désormais l'état PDL réel (`pdl`).

## Pour chef
A et A' sont opt-in, défaut inchangé au bit (tests + primaires au bit) : à fusionner ou non selon ton jugement — ils ne coûtent
rien au défaut et documentent une voie fermée. Si un jour un noyau primaire court laisse la carte vide (à b=1, la glue hors
graphe ?), le mécanisme est prêt. Je ne propose pas de suite sur cette voie.
