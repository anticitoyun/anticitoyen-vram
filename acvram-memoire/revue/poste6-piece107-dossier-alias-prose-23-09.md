# Pièce 107 — dossier à sec : un alias moins abîmé à b=1 sur la prose (invite 11 : A 0,511 contre 0,045 en int8) — 23/09 (poste6)

Ordre (chef) : comparer (i) q/k nvfp4 + v/o int8 et (ii) une calibration AWQ des projections avec de la prose française ;
pour chacun octets/pas contre A et int8, pile qkv, coût de conversion ; puis un filtre rapide scellé avant (KL b=1 sur
l'invite 11 ≤ 0,25, sinon l'option est écartée). Aucune carte pour ce dossier (0 min).

## 1. Ce que la 100 B n'avait pas isolé (lu dans les manifestes)

A (`nvfp4-qkv-alphaqkv-23-09`) et le témoin i8c ne diffèrent **pas seulement par les projections** :

| | i8c (servi) | A |
|---|---|---|
| q/k/v/o | int8 par canal | nvfp4, alpha commun q/k/v |
| experts (18 432) | nvfp4 **sans AWQ** (`awq: False`, tables d'unité) | nvfp4 **AWQ calibré** (anglais, 16 k jetons) + alpha commun par expert |
| lm_head | **int8** g128 | **nvfp4** g128 |
| snr_floor / promotions | 25 dB (q/k/v/o promus) | 0 |

Donc « A 0,511 contre 0,045 sur l'invite 11 » peut venir des projections, des experts (calibration AWQ) ou de la tête —
la 100 B l'attribuait aux projections sans le prouver. Un instrument à sec l'isole : `outils/gpu/mesure/assembler-alias.py`
(BASE + tenseurs pris chez un DONNEUR, fragments liés, manifeste réécrit avec `assemblage`, 0 min de carte, ~1 min de
disque). Un tenseur converti ne dépend que de sa source, de son format et de ses options : l'assemblage rend ce qu'une
conversion avec ces formats aurait écrit. Assemblés et **chargés à sec (processeur, 2 s)** :

| alias assemblé | pris chez i8c | isole | pile qkv (chargement à sec) |
|---|---|---|---|
| S1 `assemble-S1-proj-i8` | q/k/v/o (192 tenseurs, 907 Mo) | **les projections** (experts et tête de A) | 48/48 (pile int8) |
| S2 `assemble-S2-tete-i8` | lm_head (318 Mo) | **la tête** | 48/48 |
| S3 `assemble-S3-vo-i8` | v/o (96 tenseurs, 453 Mo) | option (i) | **0/48** (« un des poids n est pas NVFP4 » ×48) |

## 2. Les options, chiffrées (octets par couche d'attention ; 48 couches ; 0,26 J/pas par Go/pas selon la 99)

| option | q | k | v | o | Mo/couche | Δ contre int8 (19,3) | Go/pas | J/pas (J/jeton) | pile qkv | conversion |
|---|---|---|---|---|---|---|---|---|---|---|
| int8 (i8c) | 8,4 | 1,05 | 1,05 | 8,6 | 19,3 | — | 0,93 | — | oui (int8) | — |
| **A** tout nvfp4 | 4,6 | 0,58 | 0,58 | 4,7 | 10,5 | **−8,7** | 0,50 | **−0,11 (−6,8 %)** | oui | faite (37 min) |
| **(i) S3** q/k nvfp4 + v/o int8 | 4,6 | 0,58 | 1,05 | 8,6 | 14,8 | −4,5 | 0,71 | −0,057 (−3,5 %) | **non** → 3 GEMM q/k/v (+0,9 ms/pas, pièce 42 = +0,24 J/pas : annule le gain 4 fois) sauf pile partielle q+k à écrire (`attention.py::fuse`, ~20 lignes + test au bit) | **0 min** (assemblé) ; réelle : `--promotion-classes v_proj,o_proj --snr-floor 30` ≈ 26 min |
| (i bis) v int8 seul | 4,6 | 0,58 | 1,05 | 4,7 | 10,9 | −8,4 (97 % de A) | 0,52 | −0,105 | non (même refus) | 0 min (assemblable) |
| B (100 B) o int8 seul | 4,6 | 0,58 | 0,58 | 8,6 | 14,4 | −4,9 | 0,69 | −0,06 | oui | faite (26 min) ; **invite 11 jamais mesurée** |
| **(ii)** A recalibré, prose FR | 4,6 | 0,58 | 0,58 | 4,7 | 10,5 | −8,7 | 0,50 | −0,11 | oui | **37 min** (service) + corpus à obtenir (demandé à poste4 : Gutenberg FR, 600-900 Ko, sha256) ; confondu : la calibration change aussi les 18 432 experts → assembler ses q/k/v/o seuls sur A (0 min) pour isoler |

Lecture : (i) ne vaut que si la pile q+k partielle est écrite, et ne garde que 52 % des octets ; (ii) garde tout, coûte
37 min et un corpus, et son effet sur l'invite 11 est incertain (AWQ des projections : les statistiques d'entrée
dépendent modestement de la langue). **Avant l'une ou l'autre, S1 et S2 disent si les projections sont seulement en
cause** — si S1 (projections int8, experts et tête de A) reste à 0,5 sur l'invite 11, le coupable est la calibration
AWQ des experts (ou la tête, S2) et les options (i)/(ii) visent à côté.

## 3. Filtre rapide — scellé AVANT la prise (≈ 7 min de carte, `scratchpad/poste6-p107-23-09/filtre.sh`)

Instrument `kl-lot-mele-p100-confirmation.py` (jeu F, rejeu ×2), dumps existants : invite 11 (le filtre), plus 1, 8, 9, 15
(les quatre autres pires de A) — six alias dans la même prise : i8c, A, S1, S2, S3, B.
Règle du chef : **option retenue si KL(b=1, invite 11) ≤ 0,25**, écartée sinon. Prédictions et ce qui les réfute :
* i8c 0,045 et A 0,511 rejoués à ± 0,01 (sinon l'instrument a bougé, prise invalide).
* **S1 ≤ 0,15** (les projections portent l'écart) — réfuté si S1 ≥ 0,40 : l'écart est ailleurs (experts AWQ), et la
  100 B est à relire ; S2 ≈ A (± 0,1) : la tête n'y est pour rien — réfuté si S2 ≤ 0,25.
* **S3 entre 0,25 et 0,45** (v/o int8 corrige une partie) → (i) écartée par le filtre ; retenue si ≤ 0,25.
* B (o int8) ≥ 0,35 (o seul ne suffit pas, comme sur les invites 0/4 de la 100 B).
* Si S1 ≤ 0,15 et S3 > 0,25 : le mal est dans q/k (ou dans la pile), et l'option qui reste est (ii) ou « q/k int8 +
  v/o nvfp4 » (S5, assemblable) — à proposer, pas à faire sans ordre.

Durée prévue : 6 chargements × ~60 s + KL ≈ 7 min.

## 4. Filtre joué (20 h 34-20 h 37, carte 3 min 30, six chargements de 28-60 s) — verdict

* **instrument** : `filtre.sh` → `kl-lot-mele-p100-confirmation.py` jeu F (KL b=1 + lots F1-F4, rejeu ×2), dumps existants ; commit 2fca6a0d ; compute-apps début = fin = llama-server 4627 ; rejeu 0 ulp pour les six alias
* **mesuré** (KL max sur 8 pas, b=1 ; seuil du filtre 0,25 sur l'invite 11) :

  | alias | pile qkv | **invite 11** | invite 1 | invite 8 | invite 9 | invite 15 | invite 11 sous lot (F1-F4) | filtre |
  |---|---|---|---|---|---|---|---|---|
  | i8c (servi) | 48/48 | 0,045 | 0,119 | 0,265 | 0,033 | 0,111 | 0,26-0,31 | témoin |
  | A tout nvfp4 | 48/48 | **0,511** | 0,210 | 0,214 | 0,228 | 0,176 | 0,60-0,78 | écarté |
  | **S1** A + q/k/v/o int8 | 48/48 | **0,044** | 0,089 | 0,065 | 0,017 | 0,123 | 0,06-0,18 | contrôle |
  | S2 A + tête int8 | 48/48 | 0,300 | 0,084 | 0,210 | 0,378 | 0,248 | 0,37-0,54 | écarté |
  | **S3** A + v/o int8 (option i) | **0/48** | **0,234** | 0,134 | 0,247 | 0,050 | 0,130 | 0,24-0,38 | **retenu (≤ 0,25)** |
  | B A + o int8 | 48/48 | 0,333 | 0,120 | 0,129 | 0,105 | 0,109 | 0,38-0,50 | écarté |

* **verdict** :
  1. **Les projections nvfp4 portent tout l'écart de l'invite 11** : S1 (experts et tête de A, projections int8) rend 0,044 = i8c (0,045). Prédiction tenue (≤ 0,15). Et S1 fait MIEUX que i8c sur les invites 1, 8, 9 (0,089 / 0,065 / 0,017 contre 0,119 / 0,265 / 0,033) : **les experts AWQ calibrés de A sont meilleurs que ceux de i8c** — l'alias « A à projections int8 » est le meilleur des six sur cinq invites (max 0,123) ; il ne gagne aucun octet sur les projections, mais il dit que la calibration AWQ n'est pas la cause.
  2. **La tête nvfp4 compte** : S2 (tête int8 seule) 0,511 → 0,300 — prédiction « S2 ≈ A » RÉFUTÉE ; pourtant S1 (tête nvfp4, projections int8) est à 0,044 : les erreurs des projections et de la tête se composent, aucune n'est linéaire (quasi-égalité au sommet). lm_head nvfp4 ne vaut que 78 Mo/pas (≈ 0,02 J/pas) : à rendre en int8 dans tout alias servi (S4 assemblé).
  3. **Option (i) S3 passe le filtre, de justesse (0,234)**, prédiction « 0,25-0,45 » réfutée à la marge ; B (o seul) 0,333 écarté (prédit ≥ 0,35, marge) ; v int8 vaut 0,10 nat de plus que o seul. Mais S3 n'a **pas de pile qkv** (0/48, formats mixtes) : servi tel quel, 3 GEMM par couche (+0,9 ms/pas, pièce 42) — le gain d'énergie (−0,057 J/pas) est mangé quatre fois. **Une pile PARTIELLE (q+k nvfp4 empilés, v int8 à part, 2 lancements) est la condition de toute option mixte** : `attention.py::fuse` empile aujourd'hui [q, k, v] tout ou rien ; grouper par format et empiler chaque groupe (~25 lignes + test au bit) — pièce moteur, pas la mienne.
  4. Option (ii) prose FR : non jouée (ordre) ; avec S1 à 0,044, la calibration n'est plus la première suspecte — la voie qui garde le plus d'octets est un mixte fin, pas une reconversion.

* **Prêts à sec (0 min de carte), non mesurés** : S4 = S3 + tête int8 (51 % des octets de A, prédit ≤ 0,15 sur l'invite 11) ; **S6 = k/v int8, q/o nvfp4 (91 % des octets)** ; **S7 = v int8 seul (96 %)**. Un second filtre de 3 minutes (S4, S6, S7 + i8c témoin) dit lequel garde le plus d'octets sous 0,25 ; tous exigent la pile partielle.
* **durée** : prévue ≈ 7 min ; tenue 3 min 30 (chargements 28-30 s, B 60 s)
