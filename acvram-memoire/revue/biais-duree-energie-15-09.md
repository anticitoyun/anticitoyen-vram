# Biais de durée d'`energie.py` : mes fenêtres, l'excès mesuré, ce qui reste valable

Laure, 15/09/2026, sans carte. Biais trouvé par Laurine (b11b8a3, main 5239f27) :
`Energie.__exit__` relevait `time.time()` APRÈS le `join` du fil de sonde
(`sleep(periode)` = 1 s) → la durée portait l'attente du prochain tic ; J et
J/jeton (compteur) sont justes, **durée, ms/pas, t/s et W sont biaisés**.
Ordre : Sage § 8 [`sage-reprise-15-09-b.md`](sage-reprise-15-09-b.md).

## Tableau des excès (128 fenêtres de mes JSON, 14-15/09)

    fenêtres à nominal CONNU (repos 30,0 s, `sleep(30)` dans `Energie`) :
      6 fenêtres : 30,01 / 30,01 / 30,01 / 30,01 / 30,02 / 30,02  → excès 0,01-0,02 s
      (nominal entier : le tic tombe juste après, l'attente est nulle)
    fenêtres à nominal INCONNU (rondes de décodage, durée = n_pas × pas réel) :
      110 fenêtres, partie fractionnaire de la durée relevée :
        0,00-0,02 : 38     0,03-0,05 : 76     autres : 7 (P5 < 2 s et quatre
        fichiers mesurés APRÈS le correctif : 0,89 / 0,94 / 0,12 / 0,15)
      → 114/118 fenêtres biaisées tombent sur ⌈t⌉ + 0,01-0,04 s : la durée
        relevée est le PROCHAIN TIC ENTIER, l'excès vaut (⌈t⌉ − t) + 0,03,
        **uniforme dans [0,03 ; 1,03) s, inconnu par cellule**, déterministe
        (mêmes pas → même t → même excès : c'est pourquoi mes ABAB se
        reproduisaient à 0,01 s près malgré lui)

**Étendue de l'excès : ≈ 1,0 s > 0,2 s → cas « variable » de Sage.** Sur des
fenêtres de 23-36 s : **durée biaisée de +0,1 à +4,3 %**, différente entre
A et B d'une même paire (leurs t diffèrent, donc leurs fractions).

## Ce que ça change dans mes verdicts

* **J/jeton : tous justes** (compteur d'énergie, indépendant de la durée).
  Tenus tels quels : modes (max +31 % J, eco −8,7 % J), courbes MMA en J,
  1aj D en J, repos serve (16 W : W = J/30 s exact, nominal entier), narrow
  b=12 J −16 %.
* **ms/pas, t/s, W : ±1 s / fenêtre, soit ± 3-4 % par cellule, non
  corrélé entre A et B**. Les écarts A/B en ms plus petits que ~5 % ne sont
  plus tranchés par ces campagnes :
  - recourbe MIN_T route+pack : b=4 **+3,5 %** (le seuil 1,02×) est DANS le
    biais — le godet 4 n'est ni tenu ni réfuté en ms ; b=6 −10,5 % et J −10,9
    tiennent ; b=1/2/3 (+17/+9/+8) tiennent en signe.
  - courbe MMA sans route+pack : +18 à +56 % tiennent.
  - 1aj D : b=1 −4,2 % **non tranché en ms** (J −6,6 % juste) ; b=12 +10 % tient.
  - bissection 7f3f422 : +0,187 ms = +4,4 % — au bord ; MAIS good/bad et
    parent/suspect ont donné exactement 4,297 → 4,484 quatre fois, et
    v0.6.4 a rendu 4,297 quatre fois : le signe et l'ampleur sont
    confirmés par cinq répétitions, pas par une paire. Tient.
  - v0.6.4 4,297 ms ; narrow b=1 4,297 : valeurs absolues à ±0,2 ms près
    (24 s de fenêtre, excès ≤ 1 s = 4 %) — le « ≤ 4,30 » n'est pas
    décidable à cette précision ; l'égalité A = B à 0,001 ms l'est
    (même t → même excès).
  - narrow b=12 −16 % ms : entre −12,5 et −20 % ; tient en signe.
  - officiels absolus (16,25 / 17,37 / 14,01 / 11,77 ms ; 674 / 781 / 930
    t/s) : biaisés de 0 à −4 % (le vrai pas est plus court).

## Ce que je fais

1. INDEX : annotation « durée biaisée ≤ +4 %, J/jeton juste » sur mes notes
   de mesure du 14-15/09 (pas de réécriture des notes : le régime est dit).
2. Re-tampon de l'officiel 0.6.6 seulement (B narrow + témoin A, une
   session, `energie.py` corrigé — dans mon worktree depuis 4debbfb), sur
   go de Sage/Jérôme.
3. Le protocole des rondes gagne un chrono hôte `perf_counter` autour de la
   fenêtre, publié à côté de la durée `Energie` : deux instruments, l'écart
   doit être < 0,05 s, sinon la cellule est invalide (REGLES §3 : rendre
   le contrôle impossible à sauter).
