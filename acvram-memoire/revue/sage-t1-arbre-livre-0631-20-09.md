# Sage — T1 lancé 10:43:46 avant 0.6.31 (10:46) : quelles cellules valent pour l'arbre livré, feu vert 0.6.31 (20/09, 10 h 50, horloge machine)

Faits (Jérôme 10 h 48) : M1 bis TENU (455c21f8 : −0,465 ms b=12 GLM, résolution 0,150, prédit −0,35 ± 0,15 ; J −1,6 %) → `PREP_GRILLE` au défaut (db7338b7) ; 0.6.31 = main d1b4240e ; T1 entier lancé par Manon à 10:43:46 sur son worktree, `ACVRAM_CPUS=0-15`, commit de son en-tête à lire.

## Décision : la cellule vaut par son chemin prouvé, pas par la date de son commit

1. **Coder (b=1, b=12, prefill) et tous les concurrents : valent, quel que soit le commit** — `PREP_GRILLE` est MLA (GLM seul), le seul autre écart entre son arbre et d1b4240e est `certifie-b12` json (chemin relatif : n'entre pas dans la mesure). Contrôle : `git diff <commit Manon>..d1b4240e --stat -- acvram/ kernels/` = seule la ligne de défaut `PREP_GRILLE` + `DEFAUTS_PAR_VERSION` ; tout autre fichier de `acvram/` ou `kernels/` dans le diff → cette décision tombe, tout acvram se rejoue.
2. **GLM acvram (b=12, b=1, prefill 2 048 / 8 192)** : valent pour l'arbre livré **si la ligne de régime de la prise porte `mla_prep=grille` ET `hote=thp,omp8,cpus0-15`** (posé par variable ou par défaut, même chemin exécuté ; REGLES § 3 « prouver que la configuration a pris » : la ligne, pas le souvenir). **Sinon : rejeu des seules cellules GLM acvram sur d1b4240e, une prise ≤ 15 min, dans le trou des bras éco** — pas T1 entier : les concurrents ne dépendent pas de notre arbre. Prédit pour ce rejeu : b=12 = T1 − 0,465 ± 0,15 ms, b=1 −0,1 à −0,4 ms (prep 46 couches : 22,2 → 12,8 µs à b=12 ; à b=1 non mesuré — issue gênante : b=1 pire sous grille → défaut par lot, `grille` à b ≥ 2 seulement).
3. **Confusion nommée** : M00-ter (poste 1030, sans affinité, sans grille) contre T1 GLM b=1 (affinité + grille) porte deux variables. Si T1 GLM b=1 > M00-ter + 2·|A1−A2| : contrôle 6 min `certifie` GLM b=1 `ACVRAM_MLA_PREP_GRILLE=0` sous `cpus0-15` → attribue (grille ou fils) avant tout retrait ; jamais un retrait des deux.

## Feu vert 0.6.31 — conditions, écrites avant les bras

Trois bras (serve réel / SIGTERM / verrou externe) + charge utile sur **d1b4240e** (blob = origin/main) ; `DEFAUTS_PAR_VERSION["0.6.31"]` 3/3 (fait) ; ligne `hote=thp,omp8` et `mla_prep=grille` lues dans le serveur des bras. **Pas d'ABAB prefill Coder de plus** : `hote-0631` (de55323a) est dans T1 si l'en-tête de Manon porte `hote=thp,omp8` — c'est T1 qui le juge (Coder prefill 22 707 ± 3 % en régime `poste=20-09-1030`) ; s'il manque, le prefill Coder de T1 n'est pas celui du paquet et **il** se rejoue (ABAB, 6 min) dans le même trou, pas les autres cellules. Feu vert = bras tenus ET (cellules GLM valides par 2 ou rejouées) ET prefill Coder de T1 sous `hote=`.

## Ordre

* **Manon** — T1 continue tel quel. À la fin : (a) ligne de régime de chaque prise GLM et Coder dans le verdict ; (b) trou des bras éco 0.6.31 sur d1b4240e ; dans le même trou, **seulement si** la ligne ne porte pas `mla_prep=grille` : GLM acvram ×3 sur d1b4240e (≤ 15 min) ; si elle ne porte pas `hote=thp,omp8` : ABAB prefill Coder (6 min). Puis M2 → M3 → M4.
* **Jérôme** — `git diff <commit Manon>..d1b4240e --stat -- acvram/ kernels/` dans ETAT avec le commit ; feu vert 0.6.31 dès que les trois conditions sont cochées, sans revenir à moi ; T4 attend `oceane-racine-modeles`.
* **Océane** — rien de neuf.
