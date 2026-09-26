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

## 5. Second filtre + pièce 114 + 107 bis (prise 21 h 19-21 h 28, carte 9 min, compute-apps début = fin = llama-server 4627)

* **instrument** : `filtre2.sh` (commit d57710a7 ; `kl-lot-mele-p100-confirmation.py` jeu F pour le filtre, jeu G — prose Germinal, lots 7/3/9/12, 16 invites — pour la 107 bis ; rejeu ×2, 0 ulp partout) ; scellés `scelle-107bis.md` (§ 107 bis, § 114). **PPL non jouée : `ppl-decode-kv.py` plantait au chargement** (`KeyError: engine_regime`, un commentaire de fin de ligne du 23/09 — e5f08d7b — avait avalé la clé ; corrigé ici, 4 min de carte à reprendre : i8c, S1, S1b, S8).
* **second filtre** (KL b=1, invite 11, seuil 0,25) : **S4** (v/o + tête int8) **0,092 retenu** (prédit ≤ 0,15 ✓, pile qkv 0/48) · S6 (k/v int8) 0,317 écarté (prédit 0,30-0,45 ✓) · S7 (v int8) 0,369 écarté (prédit 0,40-0,50, à la marge) · i8c rejoué 0,045 ✓.
* **pièce 114 — S8 = A + projections nvfp4 en arrondi simple SANS AWQ (pile qkv 48/48, scalers identité) : invite 11 = 0,067 ≤ 0,15 → au scellé, L'AWQ EST COUPABLE, pas le format.** Invites 1/8/9/15 : 0,105 · 0,108 · 0,137 · 0,282 (i8c 0,119 · 0,265 · 0,033 · 0,111 ; A 0,210 · 0,214 · 0,228 · 0,176) ; sous le lot (invite 11) 0,11-0,14. Ma lecture « format » de la 107 § 4 était fausse : ce sont les échelles AWQ des projections (calibration anglaise, 16 k jetons, ou le report d'échelles) qui coûtent 0,44 nat sur la prose. **S8 est un candidat tout-nvfp4 à pile qkv : les octets de A (−0,42 Go/pas) avec la qualité de i8c sur 4 invites sur 5** ; l'invite 15 (0,282) dit qu'il n'est pas gratuit — qualification complète à faire (16 invites, lot mêlé, PPL).
* **107 bis, S1 contre i8c** (scellé : b=1 ≤ 0,74 sur 16/16 et ≤ i8c + 0,05 sur ≥ 13/16 ; lot mêlé littéral REGLES ≤ 3 échecs / 64 ; PPL ≤ 1,02) : b=1 **16/16 ≤ 0,74 (max 0,205), ≤ i8c + 0,05 : 13/16** (tenu de justesse : invites 0, 3, 14 au-dessus) ; **lot mêlé littéral 55/64 → 9 échecs → FAUX** (invite 0 : S1 0,09-0,15 contre i8c 0,01-0,05, l'écart est déjà à b=1 — 0,161 contre 0,008 — ; invite 14 : 0,28-0,38 contre 0,21-0,23 ; invite 15 G3 0,272) ; lecture |Δ| 55/64 aussi (invite 14 G2 +0,171, invite 15 G3 +0,149 : de vraies excursions) ; PPL absente. **Verdict 107 bis : S1 ne remplace pas i8c** (une porte tombée, sans seconde chance).
* **S1b en information** (experts de A + projections ET tête de i8c, octets de i8c) : b=1 16/16 (max 0,164), ≤ i8c + 0,05 : **15/16** ; lot mêlé littéral **61/64 → TENU** (3 échecs : invite 9 G3 0,100, invite 15 G3 0,426 et G4 0,299 — l'invite 15 est aussi la pire de i8c sous le lot) ; |Δ| 60/64 ; PPL absente. Meilleur que i8c sur 12 invites sur 16 à b=1. **C'est S1b, pas S1, le remplaçant plausible de l'alias servi — à octets et temps identiques ; sa PPL manque.**
* **durée** : prévue 15 min ; tenue 9 min (i8c 212 s au premier chargement — noyaux Triton recompilés —, puis 27-60 s par alias) ; PPL 3 × 13 s d'échec.

## 6. PPL (prise 22 h 27-22 h 29, 88 s de carte, outil réparé e972ab44 ; compute-apps début = fin = llama-server 4627)

`ppl-decode-kv`, 3 tranches Coder du corpus scellé (`scratchpad/tranches-coder/`), préfixe 8 192, 512 jetons notés par tranche,
cache de préfixe ON (défaut servi), KV int8 ; ratio géométrique contre i8c, 2 SE sur les 3 tranches ; seuil scellé ≤ 1,02.

| alias | t0 | t1 | t2 | géo / i8c | 2 SE | ≤ 1,02 |
|---|---|---|---|---|---|---|
| i8c | 3,4607 | 12,2212 | 6,8961 | 1 | — | témoin |
| S1 (experts A, proj int8, **tête nvfp4**) | 3,5428 | 13,6408 | 7,2982 | **1,065** | 0,050 | **FAUX** |
| **S1b** (idem, **tête int8**) | 3,4696 | 12,4572 | 6,9670 | **1,011** | 0,010 | **TENU** |
| S8 (proj nvfp4 RTN, tête nvfp4) | 3,4931 | 13,8879 | 7,2498 | 1,064 | 0,070 | FAUX |

* **La tête nvfp4 coûte +5 % de PPL** (S1 → S1b : seule la tête change ; 12,22 → 13,64 sur la tranche 1) — le gênant nommé
  au scellé (« la tête nvfp4 pèse plus en PPL qu'en KL ») s'est réalisé, et il pèse plus que prédit. **Les projections nvfp4
  RTN ne coûtent rien en PPL** (S8 1,064 = S1 1,065, même tête ; les projections diffèrent, le ratio non).
* **107 bis, verdict final : S1 ne remplace pas i8c** (lot mêlé 55/64 ET PPL 1,065). **S1b tient les trois portes**
  (KL b=1 16/16 et ≤ i8c + 0,05 sur 15/16 ; lot mêlé littéral 61/64 ; PPL 1,011 ± 0,010 ≤ 1,02) à octets et temps
  identiques à i8c — c'est le remplaçant qualifié de l'alias servi, sous réserve de la 118 (poste2) ; sa PPL est +1,1 %
  (les experts AWQ de A : meilleure KL, PPL un peu au-dessus — deux instruments, deux signes, dans la résolution).
* **S9 = S8 + tête int8** (assemblé à sec, 0 min : experts AWQ de A, projections nvfp4 RTN sans AWQ, tête int8) : prédit
  KL invite 11 ≈ 0,07 (comme S8) et PPL ≈ 1,01 (comme S1b) avec les octets de projections de A (−0,42 Go/pas contre i8c,
  tête +0 Mo) — le candidat tout-nvfp4-projections cohérent, à qualifier (16 invites, lot mêlé, PPL) **après la 118**
  (si le chemin nvfp4 étroit reste à 1 550-1 600 t/s, aucun alias à projections nvfp4 ne sert avant le noyau de la 100 A).
* durée : prévue 4 min ; tenue 88 s (4 chargements de 21-23 s, tranches ≈ 5 s chacune)
