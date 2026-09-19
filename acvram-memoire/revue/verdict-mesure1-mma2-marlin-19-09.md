# Verdict — Mesure 1 à unité égale (Coder b=12, même liste de 96 lignes, 45 puis 27 experts distincts) : mma2 est **plus rapide** que Marlin à octets égaux (×0,88 à 45, ×0,75 à 27) mais **au même plafond de 400 W** (401 W ≥ 380 : réfutation de poste7 atteinte) → **Mesure 2 : NON** tel que scellé (`poste7-c16bis` § 2 : t ≤ 1,15 t_marlin TENU, W ≤ 350 FAUX) ; la règle de l'addendum 20 h 15 (W·t ≤ 0,85·398·t_marlin) rend faux de 4 % à 45 et tenu à 27 — poste7 tranche

instrument : `outils/energie_par_poste.py 12 6` mode `BANC_SEULEMENT=mesure1` (poste1-11 après 7cc9d5fa, ce commit) — 5 noyaux sur la MÊME liste d'experts (générateur figé : B=12 jetons × top_k 8 = 96 lignes, exactement u distincts, 8 distincts par jeton, `liste_experts`), 48 couches en tampons tournants (DRAM, pas L2), boucle soutenue 6 s par poste, W = compteur NVML, **t noyau** = Σ durées CUPTI (`torch.profiler`, passe séparée après la boucle) / lancements, rapport cyclique = t noyau / t mur, Go/s = octets d'expert (0,885 Mo par projection, lus dans les piles) × u / t noyau ; témoins copie DRAM 1 Gio (339-342 W, 1 405 µs = 0,76 Go/ms… 1,53 To/s) et GEMM bf16 8192³ (400 W). Chaîne `donnees-mesure1-19-09/chaine-mesure1.sh` (scellé en tête), journaux `journal-{marlin,naturel}.log`
commit : poste1-11 7cc9d5fa (arbre), 19:50-19:57, sous `carte.sh`, **deux processus** (les deux dispositions ensemble, `ACVRAM_DOUBLE_DISPOSITION_DIAG=1`, sortent de la mémoire : 24,1 Gio alloués + 6,5 réservés + **PID 4284 `llama-server` 5,6 Gio, pas à moi, hors verrou, présent avant et après**, laissé en place) — bras Marlin `[régime] NOMINAL … experts_layout=marlin` (défaut), bras naturel `ACVRAM_GEMV_LAYOUT=naturel ACVRAM_PREFILL_GROUPED=groupe … experts_layout=naturel` ; repos 48,8 W (bras 1) / 63,1 W (bras 2) contre 34,5 le matin — le processus étranger ; les W publiés sont absolus (c'est ce que la règle compare à 350), W_net dans les journaux
régime : `Qwen3-Coder-30B-A3B-nvfp4`, b=12, BT=16 (`ACVRAM_MOE_MMA_BT=16`, la forme <16, 4> du ncu M2), pas de graphes sur les postes (noyaux bouclés), horloge libre (pas de `acvram eco`)
scellé (poste7 `poste7-c16bis-puissance-mesure1-19-09` § 2, en tête de la chaîne avant la mesure) : Mesure 2 seulement si t_mma2(45) ≤ 1,15 × t_marlin(45) ET W_mma2 ≤ 350 ; t_mma2 = gate+up + down + quant_act ×2, t_marlin = gate·up + down ; prédictions poste7 : Marlin 78 µs / 398 W / 1,5 To/s, mma2 90-125 µs / 280-340 W / 1,0-1,3 To/s ; réfutation poste7 : W_mma2 ≥ 380 ; prédiction poste1 : t_mma2/t_marlin 1,1-1,4 à 45 ; contrôle : rapport cyclique ≥ 0,9 par noyau, alarme Marlin gate·up hors ± 15 % de 398 W
mesuré :

| u | noyau | t noyau µs | t mur µs | W | cycl. | To/s | mJ/lancement | MHz SM |
|---|---|---|---|---|---|---|---|---|
| 45 | Marlin gate·up (`nvfp4_gemv_marlin_gateup`) | **73,5** | 80,3 | **397,8** | 0,92 | 1,08 | 31,9 | 2 422 |
| 45 | Marlin down (`nvfp4_gemv_marlin`) | **37,3** | 41,5 | 401,5 | 0,90 | 1,07 | 16,7 | 2 025 |
| 45 | mma2 gate+up (2 GEMM, 45 tuiles) | **64,0** | 65,3 | **401,3** | 0,98 | 1,24 | 26,2 | 2 730 |
| 45 | mma2 down (45 tuiles) | **28,3** | 29,1 | 399,0 | 0,97 | 1,41 | 11,6 | 2 625 |
| 45 | quant_act ×2 | 4,9 | 15,3 | (138 = boucle) | **0,32** | — | (2,1) | 3 000 |
| 27 | Marlin gate·up | 55,8 | 62,3 | 400,2 | 0,90 | 0,86 | 24,9 | 2 025 |
| 27 | Marlin down | 33,7 | 37,8 | 398,0 | 0,89 | 0,71 | 15,0 | 2 107 |
| 27 | mma2 gate+up (27 tuiles) | 43,7 | 44,7 | 400,2 | 0,98 | 1,09 | 17,9 | 2 880 |
| 27 | mma2 down | 18,4 | 19,1 | 402,5 | 0,96 | 1,30 | 7,7 | 2 760 |
| 27 | quant_act ×2 | 4,9 | 14,9 | (142 = boucle) | 0,33 | — | (2,1) | 2 985 |

Par couche (t noyau) : **u=45 Marlin 110,8 µs, mma2 97,2 µs (×0,877)** ; **u=27 Marlin 89,5, mma2 67,0 (×0,749)** ; J par couche à 45 : Marlin 48,6 mJ, mma2 39,9 (−18 %, quant_act compté au W de la boucle : borne haute). Rapport cyclique ≥ 0,89 sur les 8 noyaux d'experts (Marlin down 0,89-0,90 à la limite : son W est à 1 % près celui du noyau) ; quant_act 0,32 : 4,9 µs de noyau pour 15 µs de boucle, son W n'est pas publié comme W du noyau. Alarme non déclenchée : Marlin gate·up 397,8 W.
verdict : **Mesure 2 : NON** — la ligne t est tenue (0,88 ≤ 1,15 ; ma prédiction 1,1-1,4 réfutée : mma2 n'est pas affamé en bande, il lit 1,24-1,41 To/s contre 1,07-1,08 pour Marlin à octets égaux), la ligne W est fausse (401 W, plafond ; poste7 280-340 réfuté, son seuil de réfutation 380 dépassé : **la déquantification par la MMA ne rend pas de watts à b=12, elle rend du temps**). Ce que les horloges disent : au même plafond de 400 W, Marlin tourne à **2 025-2 422 MHz** et mma2 à **2 625-2 880** — Marlin est co-limité par ses instructions (l'horloge tombe sous le bridage de puissance, c'est le § 2 de `c16bis` mesuré noyau par noyau), mma2 tient l'horloge et gagne 12-25 % de temps, donc autant de joules à W égal. Règle de l'addendum 20 h 15 : W·t / (0,85 · 398 · t_marlin) = **1,04 à u=45 (faux de 4 %), 0,89 à u=27 (tenu)** — deux règles, deux verdicts : je n'en choisis pas, poste7 tranche ; ce que je propose est C17 avec **l'objet que le résultat fixe** : ni « mma2 sous 350 W » (il n'y est pas) ni « mma2 à ≥ 1,3 To/s » (il y est déjà au down) mais « la disposition unique côté mma2 » ne vaut que si le pas servi rend ces 12-25 % (Mesure 2 = `certifie` ABAB, la seule qui compte les 1,9 ms hors noyaux et le bridage 19 % de poste2)

## Non vérifié / à dire
* Deux processus et non un : même liste d'experts (générateur figé, `assert` sur les distincts), mêmes 48 couches, même carte ; les témoins copie/GEMM des deux passes coïncident à 1 % (339/342 W, 400/400 W) — pas de dérive entre les bras.
* Le processus étranger 4284 (llama-server, 5,6 Gio) était sur la carte et son repos diffère entre les deux passes (48,8 W bras Marlin, 63,1 W bras naturel, contre 34,5 W carte vide le matin) : les W absolus des deux bras ne sont comparables qu'à ± 15 W. Ramenés à un repos propre (W_net + 34,5) : **mma2 370-373 W, Marlin 384 W** — la réfutation de poste7 « ≥ 380 » tiendrait alors de 7-10 W pour mma2 (non tranché à cet instrument), le seuil « ≤ 350 » reste faux dans tous les cas, donc Mesure 2 reste NON. Une remesure carte vide (3 min) lève le 372/401, elle ne change pas le verdict.
* Unité : 45 et 27 sont des listes construites (8 distincts par jeton), pas le routage réel ; le compte réel par couche à b=12 vient de la trace C9-M1 (poste2, prédiction poste7 30 ± 8).
* mma2 à M=96 lignes sans le tri/`bincount`/`.item()` de la glue hôte (`_forward_prefill_grouped`) : le pas servi paie cette glue, Mesure 2 la compte.

## Mesure 1-bis (20:10-20:13, `donnees-mesure1-19-09/chaine-mesure1bis.sh`, u = 8 à b=1 et u = 16 à b=2, 4 s par poste, sans témoins) : **Marlin garde b=1 et b=2** (prédiction poste7 tenue)

| u (b) | noyau | t noyau µs | W | cycl. | To/s |
|---|---|---|---|---|---|
| 8 (1) | Marlin gate·up | **15,0** | **376** | 0,98 | 0,94 |
| 8 (1) | Marlin down | 7,2 | 402 | 0,96 | 0,98 |
| 8 (1) | mma2 gate+up (8 tuiles) | 15,6 | 383 | 0,97 | 0,91 |
| 8 (1) | mma2 down | 6,8 | 398 | 0,96 | 1,04 |
| 16 (2) | Marlin gate·up | **21,2** | 400 | 0,97 | 1,34 |
| 16 (2) | Marlin down | 11,2 | 404 | 0,96 | 1,26 |
| 16 (2) | mma2 gate+up (16 tuiles) | 28,2 | 398 | 0,99 | 1,00 |
| 16 (2) | mma2 down | 12,1 | 400 | 0,97 | 1,17 |
| 8 / 16 | quant_act ×2 | 4,9 | (boucle) | 0,31 | — |

Par couche : **u=8 Marlin 22,2 µs, mma2 27,3 (×1,23 ; ×1,01 sans la quantification)** ; **u=16 Marlin 32,4, mma2 45,2 (×1,40 ; ×1,24 sans)**. Le croisement est entre u=16 et u=27 (×0,75) : à 8-16 tuiles mma2 n'occupe que 8-16 blocs sur 170 SM (latence, pas débit : 28,2 µs à 16 tuiles contre 43,7 à 27 — non linéaire) ; Marlin gate·up à b=1 tourne sous le plafond (376 W). La route native ne vaut qu'à b ≥ ~4 (u ≥ ~24) ; à b=1 le GEMV Marlin reste.

## Contrôle (b) de C10 (20:12, 3 s, carte) : **21 passed** — `tests/test_gemv_marlin.py` sous `ACVRAM_MARLIN_DISTINCT=1` : formes GLM ajoutées (bce00f71 : gate·up 2048×1536 act 0/1, down 1536×2048 en 3 modes, à côté des formes Coder), bloc MoE décodage par GEMV Marlin, bras cassant (échelles décalées / poids permutés). Extension = cache faf1f7b (`ACVRAM_CUDA_HOME=/usr/local/cuda-13.4` obligatoire sous `carte.sh`, sinon « Error building extension » et repli référence : le premier passage a été un 21 skipped, dit ici). Pointeur à chef : fusion de C10 opt-in (`ACVRAM_MARLIN_DISTINCT`) possible.

## Mesure 1-ter (21:23-21:25, sous éco 2 700 posé par l'instrument, `eco=2700(2685)` sur les deux bras ; `donnees-mesure1-19-09/chaine-mesure1ter.sh`, journaux `mesure1ter-*.log`) : **le -lgc 2700 est un plafond, pas un plancher — Marlin reste bridé par la puissance sous 2 700, mma2 tient 2 692 : 0,893 à u=45, 0,771 à u=27**

| u | noyau | t noyau µs | W | MHz | cycl. | To/s |
|---|---|---|---|---|---|---|
| 45 | Marlin gate·up | 74,0 | 397,6 | **2 265** | 0,94 | 1,08 |
| 45 | Marlin down | 36,9 | 401,3 | **1 950** | 0,91 | 1,08 |
| 45 | mma2 gate+up | 64,8 | 388,6 | **2 692** | 0,99 | 1,23 |
| 45 | mma2 down | 28,8 | 398,2 | 2 692 | 0,98 | 1,38 |
| 45 | quant_act ×2 | 5,4 | (boucle) | 2 692 | 0,27 | — |
| 27 | Marlin gate·up | 55,2 | 403,9 | 1 942 | 0,90 | 0,87 |
| 27 | Marlin down | 33,6 | 397,3 | 2 040 | 0,91 | 0,71 |
| 27 | mma2 gate+up | 44,2 | **358,0** | 2 692 | 0,98 | 1,08 |
| 27 | mma2 down | 18,9 | 383,9 | 2 692 | 0,99 | 1,27 |

Par couche : u=45 Marlin 110,9 µs, mma2 99,0 (**×0,893**) ; u=27 88,8 contre 68,5 (**×0,771**). Scellé poste7 (`poste7-c17-scelle-mesure1-ter-20-09`, prédiction 0,95-1,04 : **réfutée** ; C17 s'écrit si ≤ 0,92 : **tenu**) ; ma prédiction 0,93-0,98 réfutée aussi. Mon contrôle « MHz des quatre noyaux à 2 640-2 700, sinon éco non effectif, non publiable » rend **faux pour Marlin (1 942-2 265 MHz)** — mais la cause est physique et vérifiée dans le même processus : l'éco est effectif (2 685 lu sous charge légère, mma2 à 2 692 sous 390 W), c'est **le plafond de puissance qui tire Marlin sous le verrou** (398-404 W à 2 000 MHz : ses instructions par octet le rendent co-limité, `c16bis` § 2) ; `-lgc` ne peut pas tenir une horloge que le bridage refuse. Lecture : sous le défaut servi (éco 2 700), l'écart mma2/Marlin n'est PAS un écart d'horloges qu'on égalise, c'est l'écart de ce que chaque noyau obtient sous 400 W ; le −11 % (u=45) est le chiffre du régime servi. poste7 tranche C17 avec ce chiffre ; mma2 à u=27 tombe à 358 W (sous le plafond) : le seul poste qui rend des watts.
