# I5 mesuré : gdn_tuiles AU BIT partout mais plus lent (T4) ; fla est déjà au plancher lecture + écriture — I5 clos (poste5, 01/10)
* instrument : `outils/gpu/mesure/banc-gdn-recurrence.py` (48 états distincts par graphe, 200 rejeux, médiane) + `tests/test_gdn_tuiles_au_bit.py` ; prise `scratchpad/poste5-i5-30-09/prise.sh`, sortie `prise.txt`, `i5.json`
* commit : 016568be9 (poste5-leap, HEAD asserté par la prise et par le banc)
* régime : carte 0 (RTX 5090) sous `carte.sh` mesure, horloge SM non relevée (diagnostic de noyau, pas une cellule publiable), cpu-safe 85 ; avant/après : seul le llama-server permanent (PID 4436, 5 614 Mio) ; Qwen3.8 HV 48, dimensions réelles, entrées aléatoires
* scellé : `revue/poste5-i5-verdict-30-09.md` § 3 (T1-T4, GO ≥ 2 µs/couche à b=12)
* mesuré : µs/couche ci-dessous ; σ des bras fla 0,03-0,06 µs ; fla avant/après identique à 0,13 µs près
* verdict : AU BIT VRAI (55/55, contrôle BV=16 rouge comme exigé) ; vitesse **T4** (tuiles plus lentes) → défaut inchangé, I5 clos
* durée : prévu ≤ 10 min / tenu 20 s (journal `tenue=20s`, 05:13:46 → 05:14:06)
## Mesures (µs par couche, Qwen3.8)
| b | fla | tuiles2 | tuiles4 | plancher lecture+écriture | plancher lecture | fla r+w |
|---|---|---|---|---|---|---|
| 1 | 4,37 | 10,16 | 6,88 | 3,94 | 2,89 | 1,44 To/s |
| 2 | 8,35 | 17,09 | 10,03 | 7,95 | 4,75 | 1,51 |
| 8 | 32,51 | 52,18 | 33,88 | 36,99 | 15,88 | 1,55 |
| 12 | 49,25 | 75,28 | 53,31 | 56,77 | 23,41 | 1,53 |
| 16 | 65,91 | 100,43 | 73,36 | 78,38 | 30,97 | 1,53 |
## Lecture
* **Au bit** : sortie et état `torch.equal` à fla sur 3 pas, b ∈ {1, 2, 8, 12, 16}, J ∈ {1, 2, 4}, HV 48 F1 on/off et HV 32 F1 on ;
  le banc le redit au premier pas (J 2 et 4, tous b). Le contrôle à sec (PTX = fla × J) a prédit juste.
* **Vitesse** : tuiles2 +53 %, tuiles4 +8 % à b=12 : les tuiles sérialisées dans un programme coûtent plus que la grille
  qu'elles épargnent. T4 → abandon de la voie Triton ; `ACVRAM_GDN_TUILES` reste opt-in, défaut 0.
* **Prédiction RÉFUTÉE** (dite telle quelle) : fla prédit à 30-31,5 µs à b=12, mesuré 49,25. Le profil servi (30-31)
  cachait les écritures : en service, la réécriture de l'état se vide pendant les GEMV suivantes ; au banc, noyaux
  dos à dos, elle se paie : fla lit ET écrit 75,5 Mo à 1,53 To/s, AU plancher (sa copie nue fait 56,77, plus lente).
* T2 lu à la lettre (plancher de lecture 23,4 ≤ fla − 4) ouvrirait une transcription CUDA ; je ne la propose pas : un
  noyau en place ne descend pas sous lecture + écriture, et fla y est déjà. I5 clos sans gain ; décision à chef.
* **Pour LeapQuant** : l'écriture de l'état se paie bien en bande (ici dans le noyau, en service sur les noyaux
  suivants) → l'estimation M2 du verdict leap (2,0-2,4 ms/pas à b=12) devient la référence, M1 (0,75-1,0) le bas.
  Au banc, ~1,2 Mo/couche/séq au lieu de 6,29 donnerait ~10 µs contre 49 à b=12 (prédiction, à sceller avec la garde).
