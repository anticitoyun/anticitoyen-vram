# Verdict — 173 : écart restant contre NInfer, alias mixte (Qwen3.8-27B-unsloth-mixte-i8c) à b=8 — 25/09 (poste1)

* **scellé** : `scratchpad/poste1-p173-25-09/scelle.md` (30ad9bed, avant les prises). Prémisse corrigée : le 315,5 de poste5
  était AVEC Marlin, sans F1/F3 (324,8 avec) ; F1/F3 au défaut depuis.
* **(1) ABBA** (banc chat de la 102, copie de la prise de poste5, serveur neuf par passe, fenêtre 20 s, -lgc 2700 ; A = F1/F3
  coupés, B = défaut actuel ; 10 passes, toutes `marlin(doubles=0,seuls=112,…,replis=0)` ; prise poste1-p173-abba, 0f4795f1) :
  **A 316,2 t/s** (314,8-317,4 ; 1,138 J/jeton, 360 W), **B 326,0 t/s** (324,5-328,7 ; 1,132 J, 369 W), **B/A +3,1 %** ; prédit
  A 310-322, B 318-334, +2 à +4 % : TENU. **Écart à NInfer (463,3, 139, non remesuré) : −29,6 %.** Une 1re prise n'a rien
  mesuré (script vide : sed cassé par un # non échappé).
* **(2) nsys** (décodage pur b=8 sous graphes, 50 pas, poste1-p173-nsys) : **19,90 ms/pas** (402 t/s ; le banc HTTP donne 326 :
  ≈ 4,6 ms/pas de préfill des requêtes qui se renouvellent, même banc que NInfer), noyaux 19,56, trous 0,34. **Idéal en octets
  lus : 12,1 ms** (19,1 Go de poids dont 10,6 int8, état GDN 2,4 Go).

| famille | ms/pas | % | idéal | gain max |
|---|---|---|---|---|
| GEMM int8 étroit (_etroit_reduit, 193 appels) | 8,30 | 42,6 | 5,9 | ≈ 2,4 |
| GEMM NVFP4 Marlin (112 appels) | 5,75 | 29,4 | 4,7 | ≈ 1,0 |
| GEMM bf16 cutlass wmma = α/β GDN (96 appels, grille 4 × 32 fils, 31 µs pour 0,5 Mo) | 2,99 | 15,3 | ≈ 0,03 | ≈ 2,9 |
| récurrence GDN (fla) | 1,06 | 5,4 | ≈ 1,35 (état) | ≈ 0 |
| attention | 0,60 | 3,1 | | |
| normes, copies, reste | 0,86 | 4,4 | | |

* **(3) les 3 plus gros postes** : int8 étroit (≈ 2,4 ms), α/β bf16 (≈ 2,9 ms, confié à poste6, 175), Marlin (≈ 1,0 ms).
  Prédiction : int8 premier poste TENU ; GDN troisième FAUX (c'est α/β, non vu).
  **Décomposition de l'int8 étroit** (régression t = a + octets/débit sur les formes à une vague) : coût FIXE ≈ 5,9 µs par
  appel × 193 ≈ 1,1 ms (occupation : 146 registres, 3 blocs/SM = 510 places, grilles de 384 à 480 blocs, 0,75-0,94 vague ;
  pas du lancement, les trous du graphe font 0,34 ms au total) ; lecture en régime à 87 % (1,56 To/s) ≈ 0,9 ms ; vague de
  gate‖up (544 blocs = 1,07 vague) ≈ 0,26 ms. Suite : pièce 176 (qkv‖gate GDN en un appel, au bit), grille de gate‖up, puis
  PDL ou persistant (après lecture de la 141 d'poste6).
* **dumps** : trace nsys et CSV hors git (dossier de session).
