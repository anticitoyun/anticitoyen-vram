# Verdict — pièce 218 : montée + vidange d'un GEMV Marlin par paire = 1,55 µs (384 blocs) / 1,54 (512), sous le seuil de 2 µs — famille du noyau persistant CLOSE (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p218-25-09/banc-fantome-218.py` (harnais de la 214 : graphe de 16 appels, médiane de 40 rejeux ; jeux
  d'experts à −1 → chaque bloc écrit 64 zéros et sort, `acvram_kernels.cu:2205` ; sortie nulle vérifiée) ; `chaine.sh` ; `fantome.json/.log`.
* **commit** : ecf2eb1b0 (poste6-218 = origin/main 719396939 + scellé + banc) ; extension du worktree compilée sous la prise.
* **régime** : carte 0, -lgc 2700 (horloge fin 2 602), llama-server 4242 (5,6 Go, tiers) présent début = fin.
* **scellé** : `scratchpad/poste6-p218-25-09/scelle.md` — fantôme 384 blocs prédit 1,2-1,8 µs ; seuil (chef) ≥ 2 µs par noyau servi → code.
* **mesuré** : fantôme gate·up G = 8 (384 blocs, S = 4) **1,55 µs** ; down (512, S = 2) **1,54** ; G = 1 (96 / 128 blocs) 1,53 / 1,43 ; G = 64
  (768 / 2 048) 1,81 / 2,32 ; vrais noyaux dans le même processus 12,85 / 8,12 (214 : 12,91 / 8,10).
* **verdict** : prédiction TENUE, seuil NON atteint (1,55 < 2) : montée + vidange + un store = **1,5 µs par noyau, quasi indépendant du
  nombre de blocs** (+0,3 µs de 96 à 768) ; la fusion gate·up→down en recouvrirait au plus une par couche = 0,07 ms/pas (2,5 %) —
  **famille close, aucun code**. Les 3,5 µs restants de structure par noyau (214 : 5,0 − 1,5) sont internes à l'item (remplissage de x,
  épilogue split-K, latences en chaîne) et ne se fusionnent pas.
* **durée** : prévu ≤ 60 s + compilation ; tenu 162 s (dont ≈ 150 de compilation), file 953 s.

## Suite (à chef)
Rien sur ce noyau à b=1. Instruments réutilisables : `banc-gemv-214.py` (débit par forme, G/S, tiède/froide), `temoin-bit.py` (au bit),
`banc-fantome-218.py` (montée + vidange de toute grille de GEMV par paire).
