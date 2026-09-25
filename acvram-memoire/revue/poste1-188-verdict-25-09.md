# 188 — verdict (poste1, 25/09) : mixte b=8 sur main à jour + GDN_AB=auto (à la main) : 17,48 ms/pas (NInfer 15,98) ; banc 373,3 t/s (NInfer 463,3)

* instrument : `scratchpad/poste1-p188-25-09/` — `prise-nsys.sh` + `profil.py` + `familles.py` (copies de la 173 ; famille α/β
  bf16 ajoutée en tête), `prise-abba.sh` (banc chat de la 102, serveur neuf par passe, 10 passes, fenêtre 25 s, `energie.py`)
* commit : 67782c01 (poste1-mtp = main ecc25f97 + scellé), ATTENDU vérifié par les deux scripts
* régime : Qwen3.8-27B-unsloth-mixte-i8c, b = 8, carte 0 seule, -lgc 2700 (horloge moyenne 2 632-2 645 MHz), ECO=off,
  cpu-safe 100 début/fin ; **ACVRAM_GDN_AB=auto posé À LA MAIN** (175b pas au défaut), preuve `ab=auto(48)` (ligne PAS nsys,
  /metrics des 5 passes B ; aucun `ab=` en A) ; graphes on, repli_eager=0, aucune PASSE NULLE ; PID hors verrou : 4242 seul
* scellé : `scratchpad/poste1-p188-25-09/scelle.md` (67782c01, avant toute mesure)
* mesuré : pas nsys 17,476 ms (noyaux 17,172, trous 0,297) ; banc A 336,1 / B 373,3 t/s (médianes)
* verdict : banc TENU (A, B et B/A dans leurs fourchettes) ; pas nsys FAUX de 0,08 ms (au-dessus de la fourchette) ; 3e poste FAUX
* durée : prévu ≤ 15 min par prise ; tenu 23 s (nsys 08:28:16-08:28:39, après 677 s d'attente du verrou d'poste6),
  8,5 min (ABBA 08:50:11-08:58:41)

## (1) Pas de décodage b=8 (nsys, 50 pas, conditions de la 173)
| | 173 (matin) | 188 | prédit (fourchette) |
|---|---|---|---|
| pas | 19,90 ms | **17,476 ms** (457,8 t/s) | 17,1 (16,8-17,4) → **FAUX de +0,08** |
| écart à NInfer 15,98 | +3,92 | **+1,50 ms (+9,4 %)** | +1,1 (+0,8 / +1,4) → FAUX |

| famille | 173 | 188 | noyaux principaux (188) |
|---|---|---|---|
| GEMM int8 étroit | 8,30 | **8,37** | `_etroit_reduit` 5,29 + `_etroit_segments` (176) 3,04 |
| Marlin NVFP4 | 5,75 | **5,80** | |
| GDN (récurrence + norme) | 1,06 | **1,18** | fused_recurrent 1,12 ; `_norme_gated` 0,06 |
| α/β bf16 | 2,99 | **0,70** | `_gemv_bf16_kernel` (triton, 48 × 14,6 µs ; prédit ≈ 0,45, alarme > 0,8 non atteinte) |
| attention | 0,60 | **0,44** | `paged_attn_partial_gqa` 0,345 (182) |
| normes, glue, copies | 0,86 | **0,68** | rmsnorm 0,29, conv 0,11, swiglu 0,06 |

Lecture de l'écart : α/β rend 2,29 ms (prédit 2,5), le reste rend ≈ 0,14 ms (GQA, z, 176). L'écart de +0,37 ms à ma
prédiction vient surtout d'α/β, qui coûte 14,6 µs par appel sous graphe contre 8,9 µs au micro-banc de la 175.

## (2) Les 3 prochains postes (par taille), avec leur gain max
1. **GEMM int8 étroit, 8,37 ms** : gain max ≈ 2,5 ms (idéal en octets 5,9 de la 173 ; le coût fixe ≈ 5,9 µs × 193 appels
   ≈ 1,1 ms est la plus grosse part évitable).
2. **Marlin NVFP4, 5,80 ms** : gain max ≤ 0,6 ms (180 : 106-110 % de son plancher).
3. **Récurrence GDN, 1,18 ms** : gain ≈ 0 (bornée par la lecture de l'état). Ma prédiction « normes + copies + glue en 3e »
   est FAUSSE (0,68 ms).
Par gain et non par taille, le 3e levier est **α/β : 0,70 → ≈ 0,19 ms** (concat 3,9 µs/couche au micro-banc, mais hors bit
à M = 8 : KL 3,4 × les témoins, 175) — ≈ 0,5 ms à prendre si un GEMM étroit au bit descend sous 5 µs. Suivent normes et glue
(≈ 0,3 ms, 3 fusions au bit, 182 à sec).
**Somme des gains max des trois leviers (int8, Marlin, α/β) ≈ 3,6 ms ; l'écart à NInfer est de 1,5 ms** : il se ferme par l'int8 étroit seul (≈ 60 % de son gain max).

## (3) Banc chat servi (ABBA, 10 passes ; A = défaut de main, B = + GDN_AB=auto)
| bras | t/s par passe | médiane | J/jeton net | W | prédit (fourchette) |
|---|---|---|---|---|---|
| A | 336,4 333,3 333,5 336,1 336,3 | **336,1** | 0,8403 | 358 | 330 (320-340) TENU |
| B | 373,7 373,4 373,3 373,3 373,3 | **373,3** | 0,8340 | 387 | 370 (355-385) TENU |

B/A **+11,1 %** (prédit +12, fourchette +8 / +16) : TENU. Face au matin (326) : **+14,5 %**. Face à NInfer (463,3, 139,
non remesuré) : **−19,4 %** (prédit ≈ −20 %). Toutes les passes marquées « bridage puissance », dans les deux bras.
Pas effectif du banc B : 8 / 373,3 = 21,4 ms, dont 17,5 de décodage, donc ≈ 3,9 ms/pas de préfill (4,8 ce matin, 179b).

Dumps (hors git) : trace nsys `p188.nsys-rep` et CSV dans le dossier de session ; journaux ignorés : `prise-nsys.log`,
`prise-abba.log`, `serveur-ab-*.log` ; cellule `cellule-ab-Qwen3.8-27B-unsloth-mixte-i8c-b8.jsonl`.
