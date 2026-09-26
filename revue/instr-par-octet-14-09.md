# Instructions par octet : acvram vs vLLM au décodage b=12 (14/09, poste4)

Commande de chef sur l'avis de poste7 (`acvram-memoire/revue/poste7-strategie-14-09.md`) :
l'anomalie n'est pas le temps mais la puissance — vLLM 242 W nets pour
1 198 t/s, nous 394 W nets pour ~717 t/s, mêmes octets par pas → ×2,9 J/jeton.
Hypothèse à réfuter : notre décodage exécute ≥ 3× plus d'instructions par
octet DRAM que vLLM (déquantification E2M1/int8 sur cœurs CUDA dans
`nvfp4_gemv_grouped*` / `int8_gemv`, contre la MMA block-scaled de CUTLASS
qui consomme l'E2M1 sans ALU).

## Protocole

Même modèle (Qwen3-Coder-30B-A3B NVFP4), même b=12, même régime court
(invites 128-256 jetons), carte exclusive (`outils/carte.sh`,
`ACVRAM_TYPE=mesure`), processus séparés.

M-ncu — `outils/ncu_instr_par_octet.sh {acvram|vllm} 12` : ncu sur les
noyaux d'une plage NVTX « mesure » qui enferme 4 pas de décodage après le
prefill (acvram : `banc_decodage_moe.py ncu`, rejeu de graphes ; vLLM :
`ncu_vllm_decode12.py`, EngineCore en processus, boucle `step()` à la
main, graphes actifs comme au duel). Métriques par noyau :
`gpu__time_duration`, `dram__bytes_read/write`, `sm__inst_executed`
(+ pipes tensor/fma/lsu), `sm__throughput`, `sm__cycles_elapsed.per_second`
(horloge réelle : `--clock-control none`). Agrégation par noyau puis par
poste : `outils/ncu_instr_par_octet_agrege.py`.

M-J — `outils/energie_par_poste.py 12 6` : chaque poste de NOTRE pas en
boucle 6 s (compteur d'énergie NVML, 5090 seule), repos 30 s avant,
horloge SM relevée ; deux témoins encadrent : copie DRAM 1 Gio (octets
sans instructions) et GEMM bf16 8192³ (instructions sans octets) ; puis le
pas complet en rejeu (`Engine.step`, régime réel). vLLM : pas de boucle
par noyau possible sans le démonter — son W par pas vient du duel (audit
A2), et son partage par poste se déduit de son profil ncu (temps × W).

## Prédictions scellées (avant toute mesure)

inst/octet = instructions warp exécutées par octet DRAM lu, par poste.

| poste | prédiction nous | prédiction vLLM | ratio prédit |
|---|---:|---:|---:|
| MoE (gate·up + down) | 0,5-1,0 | 0,05-0,10 | **≥ 5×** |
| projections d'attention | 0,6-1,2 (int8_gemv) | 0,05-0,15 (GEMM FP4 dense) | ≥ 5× |
| attention paginée | ≥ 2× le Triton de vLLM | | ≥ 2× |
| **pas entier** (Σ inst / Σ octets) | | | **≥ 3×** |

Réfutation (seuil de poste7) : ratio du pas entier ≤ 1,5× → la puissance
vient d'ailleurs (horloge, bulles, plafond 400 W) et il faut le dire
avant tout noyau de plus.

Énergie (W moyens en boucle, 5090 seule, net = brut − repos) :
témoin copie 250-300 W ; témoin GEMM ≈ 400 W (plafond) ; MoE GEMV
≥ copie + 60 W ; int8_gemv et lm_head ≥ copie + 50 W ; pas complet rejeu
380-400 W avec bridage « puissance » relevé. Réfutation de l'hypothèse
« instructions » côté énergie : noyaux du pas ≤ copie + 30 W.

Issues gênantes nommées d'avance : (a) si le pas complet sature le
plafond 400 W, l'horloge tombe et le temps monte — vLLM à ~310 W bruts
ne plafonne pas : une part de l'écart de temps serait alors un effet du
plafond, pas des noyaux ; (b) ncu sous `--clock-control none` : les
comptes sont exacts, les durées restent celles d'un noyau isolé et
sérialisé (leçon : ncu surestime les noyaux courts) — on ne compare
pas les ms de ncu à celles du profil ; (c) un ratio inst/octet élevé ne
prouve la puissance que si M-J le confirme sur nos noyaux.

## Résultats (14/09, 17h39-19h25, carte exclusive)

Écarts au protocole : (1) les compteurs ncu sont réservés à root
(`RmProfilingAdminOnly=1`) — sudoers `ncu` installé par l'utilisateur en
cours de route, `sudo -n ncu` avec environnement repassé par `env` (HOME de
travail, copie du cache de noyaux) ; (2) UN pas profilé par moteur au lieu
de 4 (ncu rejoue chaque noyau en sauvegardant les 15-24 Gio de mémoire du
processus : 5 min pour nos 1 506 lancements, 12 min pour les 1 412 de vLLM) ;
(3) métriques réduites à ce qui tient en peu de passes : durée, octets DRAM
(`dram__bytes_op_read/write`, les seuls noms qui existent sur Blackwell),
`sm__inst_executed`, pipe tensor, horloge — pas de `sm__throughput` ;
(4) vLLM sur le checkpoint ModelOpt du duel
(`models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4`, KV fp8), pas sur notre
converti (vLLM y lit des experts non quantifiés, 60 Gio, OOM) ; (5) le
premier passage a été perdu avec la session (relance identique).

### M-ncu — un pas de décodage b=12, noyaux de la plage NVTX

Les ms de ncu sont celles de noyaux isolés et sérialisés (surestimées,
surtout les courts) : on ne les compare pas aux ms du profil ; **les
comptes (octets, instructions) sont exacts.** inst/oct = instructions
warp par octet DRAM lu.

acvram (rejeu de graphes, godet 16 pour 12 séquences) :

| poste | lanc. | ms ncu | Go lus | G inst | inst/oct | W net en boucle (M-J) |
|---|---:|---:|---:|---:|---:|---:|
| MoE gate·up (`nvfp4_gemv_grouped_gateup`) | 48 | 4,17 | 2,556 | 3,47 | **1,36** | 353 |
| MoE down (`nvfp4_gemv_grouped_warp`) | 48 | 4,06 | 1,296 | 3,31 | **2,55** | 363 |
| projections int8 NV=12, 12 lignes (`int8_gemv<4,12>`) | 96 | 3,26 | 0,940 | 1,26 | 1,34 | 361 (q/k/v) / 358 (o) |
| **projections int8, 4 lignes FANTÔMES (`int8_gemv<4,4>`)** | 96 | 1,39 | **0,932** | 0,53 | 0,58 | — |
| lm_head int8, 12 lignes + 4 fantômes | 2 | 0,68 + 0,27 | 0,319 + **0,319** | 0,62 | 0,97 | 356 |
| attention paginée (`paged_attn_partial` + reduce) | 96 | 1,02 | 0,137 | 0,32 | 2,34 | — |
| routage + glue MoE | 293 | 0,92 | 0,080 | 0,02 | 0,20 | — |
| norm + rope + élémentaires | 731 | 2,09 | 0,059 | 0,04 | 0,67 | 89 (norme seule) |
| **total** | **1 506** | 18,2 | **6,667** | **9,57** | **1,44** | pas complet : **354 net / 392 brut** |

vLLM 0.29 (graphes CUDA, moteur en processus) :

| poste | lanc. | ms ncu | Go lus | G inst | inst/oct | tensor |
|---|---:|---:|---:|---:|---:|---:|
| GEMM FP4 CUTLASS (`cutlass::device_kernel` : MoE groupée gate·up, down + q/kv et o denses) | 192 | 5,54 | 4,926 | 0,91 | **0,18** | 15 % |
| lm_head + routeurs bf16 (`wmma_tensorop_bf16`) | 49 | 0,57 | 0,651 | 0,04 | 0,06 | 6 % |
| `cvt_fp16_to_fp4` (quantification des activations) | 192 | 0,80 | 0,044 | 0,06 | 1,3 | — |
| attention (`kernel_unified_attention`, Triton, KV fp8) | 48 | 0,36 | 0,155 | 0,10 | 0,66 | — |
| routage + glue (shuffle, reduce, topk, offsets, tris) | 535 | 1,93 | 0,108 | 0,07 | 0,69 | — |
| norm (Triton fusionnées) + élémentaires | 197 | 0,52 | 0,034 | 0,02 | 0,63 | — |
| **total** | **1 412** | 10,3 | **5,978** | **1,19** | **0,20** | 12 % |

**Ratio pas entier : 1,44 / 0,20 = ×7,2** (prédit ≥ 3× — tenu, seuil de
réfutation 1,5× loin). Sur les GEMM seules (MoE + projections) :
(3,47 + 3,31 + 1,26 + 0,53) / (2,556 + 1,296 + 0,940 + 0,932) = 1,50
inst/oct contre 0,18 → **×8,3**. Nos MoE GEMV seuls font 6,8 G des 9,6 G
instructions du pas (71 %) ; chez vLLM la MMA block-scaled consomme
l'E2M1 sans ALU (15 % d'instructions tensor, 0,18 inst/oct sur 4,9 Go).

Mêmes octets : 6,67 Go (nous) contre 5,98 Go (eux) ; **sans le trafic
fantôme ci-dessous nous serions à 5,42 Go, en dessous de vLLM**.

### Trouvaille, puis réfutation : le godet 16 « relit » les poids — sous ncu seulement

`int8_gemv_kernel<4, NV>` traite NV colonnes (lignes du lot) ; à 12
séquences en godet 16 (`graphs.py:45`, `bucket_batch`), chaque projection
et le lm_head sont lancés DEUX fois : `<4,12>` pour les 12 vraies lignes,
puis `<4,4>` pour les 4 fantômes, et ncu comptait pour le second
lancement 0,932 + 0,319 = 1,25 Go de DRAM par pas (19 % du pas).

Correctif : tranche de 16 (NV=13-16 instanciés, un seul lancement ;
`ACVRAM_INT8_TRANCHE=12` rend l'ancien découpage — témoin A/B dans le
même binaire ; `tests/test_int8_gemv_lot.py` N=1..16 bit-identique à
N=1, 29 passés). Prédiction scellée (chef) : −1,2 ms/pas, −19 % d'octets.

**Mesure (19h37-19h47, carte exclusive) : RÉFUTÉE.** Jetons identiques
A/B (12 empreintes, `test_graphes_vs_eager`). ms/pas en rejeu, ABAB :
A 14,79 / 14,80 ; B 14,81 / 14,85 (σ 0,05) → **±0**. Contrôle qui a
rendu « faux » : ncu relancé sur A avec `--cache-control none` (L2 non
purgé entre les rejeux) — les lancements `<4,4>` des projections lisent
**0,001 Go** de DRAM (contre 0,932 sous purge) : le second lancement était
servi par le L2 (poids d'une projection : 2-8 Mo, sous les 96 Mo de L2),
seul le lm_head `<4,4>` (319 Mo > L2) relit vraiment. **Par défaut ncu
purge les caches avant chaque rejeu (`--cache-control all`) : chaque noyau
est mesuré à froid, et une relecture que le L2 sert en vrai est comptée
en DRAM.** Troisième fois que la leçon « une relecture servie par le L2
n'est pas un levier » revient (tuile 128, INT8 NV=12, godet 16) — cette
fois par l'instrument. Total DRAM du pas à cache chaud : **5,52 Go**
(6,67 sous purge), inst/oct **1,73**. Le changement de tranche est gardé
(97 lancements de moins par pas, 319 Mo de moins pour le lm_head, ±0 ms,
jetons identiques) — sans gain à revendiquer. Données :
`revue/donnees-ncu-ipo-acvram-cache-chaud-14-09.txt`. Colonne vLLM refaite à cache chaud (20h11, noyaux lourds seuls :
CUTLASS 4,870 Go, wmma 0,648, attention 0,152 ; petits noyaux 0,158 dans
un passage séparé) : 5,83 Go, quasi inchangés (leurs noyaux ne relisent
rien). GEMM à GEMM : vLLM 4,87 Go (MoE + projections) contre nous
3,83 + 0,93 = 4,76 — mêmes octets à 2 %.
`revue/donnees-ncu-ipo-vllm-cache-chaud-14-09.txt`.

### M1 — noyaux sous rejeu (torch.profiler, `BANC_GRAPHES=1 profil 12`)

Σ noyaux sous rejeu = **13,71 ms** (41 noyaux) contre 12,25 eager : +1,46 ms.
Prédiction « Σ ≥ 14,0 » : **réfutée de justesse** (13,71). Le mécanisme
prédit (l'écart dans les postes étroits, pas dans le MoE) est celui que
ncu montre (1,25 Go de relecture int8 des fantômes, projections + lm_head),
mais le détail par noyau de ce passage a été perdu (filtre de sortie
faux) et la reprise a chargé un moteur DÉGRADÉ (experts exilés, 110 ms de
copies hôte→carte par pas : une autre mesure occupait la VRAM au
chargement) — résultat jeté, c'est le cas exact du garde-fou `regime.py`
d'poste1, à câbler ici avant toute reprise.

### M2 — octets DRAM réels et % de borne (ms du profil eager, octets ncu)

| poste | Go/pas | borne à 1 050 Go/s | ms eager | % de borne |
|---|---:|---:|---:|---:|
| MoE gate·up | 2,554 | 2,43 | 3,35 | 73 % |
| MoE down | 1,277 | 1,22 | 2,63 | **46 %** |
| MoE total | 3,831 | 3,65 | 5,98 | 61 % |
| projections int8 (12 lignes) | 0,930 | 0,89 | 2,62 | 34 % |
| lm_head int8 (12 lignes) | 0,319 | 0,30 | 0,88 | 35 % |
| attention paginée | 0,137 | 0,13 | 0,72 | 18 % |
| pas entier (rejeu, cache chaud) | 5,518 | 5,26 | 14,8 | 36 % |

(octets à cache chaud, `--cache-control none` ; sous purge le pas
comptait 6,67 Go, différence = relectures servies par le L2.)

Prédiction « MoE ≥ 70 % de borne » : **réfutée** (61 % ; gate·up 73 %,
down 47 %). Le MoE lit 3,85 Go, soit ≈ 30 experts distincts par couche
(3,85 / (48 × 2,65 Mo)) — pas 68 : le routage à b=12 est très concentré.
Conséquence pour poste7 (« MoE 82 % de borne, aucun levier ») : faux sur
les deux axes — 61 % en temps, et 71 % des instructions du pas.
Projections / lm_head / attention ≤ 40 % : tenu.

### M-J — puissance par poste (NVML, 5090 seule, repos 38,6 W, boucle 6 s)

| poste | W brut | W net | horloge SM | bridage |
|---|---:|---:|---:|---|
| témoin copie DRAM 1 Gio (1,5 To/s, ~0 instruction) | 322 | 283 | 2 992 | aucun |
| témoin GEMM bf16 8192³ | 394 | 355 | 2 115 | puissance |
| MoE gate·up GEMV | 391 | 353 | **1 642** | puissance |
| MoE down GEMV | 401 | 363 | 2 287 | puissance |
| MoE complet (gate·up + down + reduce) | 399 | 360 | 1 882 | puissance |
| projections q/k/v int8 | 399 | 361 | 2 850 | puissance |
| projection o int8 | 397 | 358 | 2 985 | puissance |
| lm_head int8 | 394 | 356 | 2 647 | puissance |
| norme RMS (bornée par le lancement) | 127 | 89 | 2 992 | — |
| **pas complet b=12, rejeu** | **392** | **354** | 2 632 | **puissance** (15,66 ms/pas, 0,512 J/jeton brut) |
| **vLLM, pas complet b=12, même instrument** (`ncu_vllm_decode12.py`, BANC_ENERGIE=200) | **306** | **265** | 2 947 | (6,30 ms/pas, 0,192 J/jeton brut, 0,167 net) |

Prédictions tenues : copie 250-300 (283 net), GEMM ≈ 400, nos GEMV
≥ copie + 60 W (+70 à +80 W), pas complet 380-400 bridé. Réfutation
(noyaux ≤ copie + 30 W) loin.

**Chacun de nos gros noyaux sature seul le plafond de 400 W**, et pour y
tenir la carte baisse l'horloge (1 642 MHz sous gate·up, 2 287 sous down)
— la demande réelle à pleine horloge est donc supérieure. Le témoin qui
déplace plus d'octets que n'importe lequel d'entre eux (1,5 To/s) n'en
demande que 322 W à 2 992 MHz : **les +70-80 W sont le prix des
instructions, pas des octets.** Les GEMM FP4 de vLLM tiennent à 306 W
et 2 947 MHz, sans bridage.

## Verdict : où partent les watts

Même instrument, même carte, même heure, moteurs en processus :

| | acvram | vLLM | ratio |
|---|---:|---:|---:|
| ms par pas b=12 | 15,66 | 6,30 | ×2,5 |
| W brut / net | 392 / 354 | 306 / 265 | ×1,28 / ×1,34 |
| J par jeton brut / net | 0,512 / 0,462 | 0,192 / 0,167 | **×2,7 / ×2,8** |
| Go DRAM lus par pas | 5,52 chaud (6,67 sous purge) | ≈ 5,83 chaud (CUTLASS 4,87 + wmma 0,65 + attention 0,15 + petits 0,16 ; 5,98 sous purge) | ≈ ×1 |
| G instructions par pas | 9,57 | 1,19 | **×8,0** |
| inst / octet (cache chaud) | 1,73 | 0,20 | **×8,5** |

Le ×2,8 en énergie se décompose en ×2,5 de temps et ×1,3 de puissance ;
les « 150 W » de l'énoncé sont ~90 W nets dans une mesure appariée (les
242 W nets de vLLM au duel venaient d'un repos plus élevé — poste3, même
jour). **Les deux facteurs ont la même cause, mesurée : 8× plus
d'instructions pour les mêmes octets.** Elles coûtent du temps (nos GEMV
à 47-73 % de leur borne octets, vLLM à 100 %) et des watts (plafond 400 W
atteint par chaque noyau, horloge bridée à 1,6-2,3 GHz), là où la MMA
block-scaled fait le même travail à 0,18 inst/oct, 306 W, horloge pleine.
Hypothèse de poste7 : **confirmée, au-delà du seuil (×7,2 pour ≥ 3
prédit).** Les noyaux SONT le bon chantier — mais pas ceux qu'on
croyait : par instructions, le MoE (71 %) avant les projections (19 %).

Ce que ça ordonne, en prédictions :

1. ~~**Godet 12 / NV=16** (trafic fantôme, 1,25 Go/pas) : −1,2 ms/pas,
   −19 % d'octets~~ — **RÉFUTÉ** (±0 ms, artefact de purge ncu, voir
   ci-dessus).
2. **Bead 1aj** (projections + lm_head en W4A4 MMA natif) : octets ÷ 1,8
   ET instructions ÷ 7 sur 19 % des instructions du pas → −1,5 ms/pas
   et −15 à −25 W en boucle sur ces postes. Réfutation : W en boucle
   inchangé (≥ 350 net) ou < −0,6 ms.
3. **MoE en MMA au décodage** — le poste qu'on avait fermé (`GEMV reste`,
   ×0,51 avec MMA2 forcée) : ce verdict mesurait la glue (quant_act +
   3 GEMM + activation par couche), pas le noyau ; vLLM fait 12 jetons ×
   30 experts en UNE GEMM groupée par projection à 0,18 inst/oct. Bead à
   ouvrir avec chef : gate·up et down groupés en un lancement chacun,
   activation quantifiée une fois par couche. Prédiction : instructions
   du MoE ÷ 4 (6,8 → ≤ 1,7 G), pas −2 à −3 ms, W du pas sous le plafond
   (< 380 brut). Réfutation : pas ≥ 14 ms ou W ≥ 390.

## Étape (iii) de poste7 — MoE en MMA groupée au décodage, b=12 eager (14/09, 19h51-19h58)

`ACVRAM_MOE_GROUPED_MAX=0`, `ACVRAM_MOE_MMA_BT=16` (une tuile m16 par
expert, M≈3 jetons par expert), eager, carte exclusive. Seuils de poste7
(`revue/poste7-moe-mma-decodage-14-09.md`) : ≥ 5,0 ms ou ≥ 380 W → split-K.

| mesure | GEMV (défaut) | MMA groupée BT=16 |
|---|---:|---:|
| noyaux MoE par pas, profil eager (`profilmma 12`) | gate·up 3,35 + down 2,63 = **5,98 ms** | 3 GEMM 3,62 + quant_act 0,17 + act/reduce ≈ 0,2 = **≈ 4,0 ms** |
| pas GPU (Σ noyaux, profil) | 12,25 ms | 12,34 ms (le MoE gagne 2 ms, la glue torch en rend 2 : 59 types de noyaux, 3 809 lancements/pas) |
| pas à l'horloge, eager | 19,54 ms | **39,47 ms** — borné par l'hôte (argsort, bincount, `int(ntiles.sum())`, 2 300 lancements torch de plus) : le blocage (ii) d'poste1 |
| octets DRAM MoE (ncu, cache chaud) | 3,83 Go | 4,33 Go (+13 %) — **c'est le routage, pas les tuiles** : bras `experts` au même pas, GEMV 30,7 experts distincts/couche × 2,654 Mo × 48 = 3,907 Go attendus (3,83 mesurés, 98 %) ; MMA 34,0/couche = 4,326 Go attendus (4,33 mesurés, 100 %) ; max 12 jetons/expert dans les deux (une seule tuile de 16, objection de poste7/chef fondée). Les jetons des deux chemins ont divergé après le prefill (W4A4), le pas échantillonné route plus large. **Aucune relecture : bt=32 sans objet.** |
| instructions MoE | 6,78 G (1,77 inst/oct) | **0,57 G (0,13 inst/oct, ÷ 12)** ; pas entier 9,57 → 2,60 G (÷ 3,7) |
| W en boucle 6 s, 3 GEMM seules (tuiles et activations pré-quantifiées) | gate·up 399 brut / down 399 (MoE complet 392-399, 1 785-1 792 MHz) | **398,6 brut / 340,5 net, 2 617 MHz**, bridage puissance |
| mJ par couche MoE en boucle | 78,4 (GEMV complet) | 54,6 (3 GEMM seules) : **−30 %** |
| quant_act ×2 en boucle | — | 150 W, 7,4 µs (borné par le lancement) |

Verdict (iii) contre les seuils scellés : **temps tenu** (≈ 4,0 ms
< 5,0 ; la prédiction 4,0-4,3 de poste7 est au point) ; **puissance
brute NON tenue** (398,6 ≥ 380 W ; en net 340,5 ≤ 345). L'horloge tient
(2 617 ≥ 2 400 contre 1 792 pour la GEMV) et l'énergie par couche baisse
de 30 % ; mais le noyau reste au plafond : à 1 156 Go/s (ncu) il est
maintenant borné par la DRAM (la copie témoin à 1,5 To/s fait 341 W à
elle seule) — le prix des instructions est parti, celui des octets
reste. Octets +13 % : **pas un défaut du noyau** — le dénominateur exact
(experts distincts × 2,654 Mo, bras `experts` de `banc_decodage_moe.py`)
reproduit les octets ncu à 98-100 % sur les deux chemins ; l'écart est
un routage plus large au pas échantillonné (34,0 contre 30,7 experts
par couche, jetons divergés après le prefill). Le noyau MMA lit
exactement ses poids, une fois.

Selon la règle de poste7 (≥ 380 W brut → split-K, +2 j, (3) derrière 1aj)
c'est split-K. Objection factuelle à lui soumettre : split-K vise un
noyau lent (latence non recouverte) ; ici le noyau est à 1 156 Go/s et
au plafond de puissance parce qu'il déplace les octets vite — split-K
n'enlève pas d'octets. Ce qui enlèverait des watts au pas : (a) les
2 ms de glue hôte (gratuits en W), (b) le +13 % d'octets des tuiles, (c)
rien côté noyau tant que la DRAM elle-même coûte ~300 W au débit de la
borne. Le pas complet sous graphes (poste1, (ii)) est la mesure qui
tranche `pas ≤ 380 W et ≤ 0,45 J/jeton`. Données :
`revue/donnees-ncu-ipo-acvram-mma-bt16-eager-14-09.txt`.

## Point 0 de reprise (poste7) — llama.cpp b=1 sous le même instrument (14/09, 10h10-10h34)

En-tête de mesure (REGLES §3) : 5090 seule (`CUDA_VISIBLE_DEVICES=0`),
plafond 400 W, horloge libre (3 135 MHz max ; sonde -lgc de poste3 à
10:11:55 hors verrou → la première fenêtre llama.cpp (9,8 s, 3 000 jetons)
est aussi invalide par sa durée, refaite), compteur NVML
`TotalEnergyConsumption` (energie.py), ncu sous sudoers avec
`--cache-control none`, llama-server = binaire LM Studio cuda12 2.22.0
(celui de poste8/poste3), Coder-30B Q4_K_M (18,6 Go), `-np 1 -c 16384 -ngl 999` ;
acvram = venv anticitoyen-vram, Coder-30B NVFP4 (experts NVFP4, attention et
lm_head int8), rejeu de graphes, `max_model_len=1024`. Outils :
`outils/ncu_llamacpp_b1.sh` (invite d'un jeton, tout profilé, ÷ pas),
`outils/energie_llamacpp_b1.py`, `outils/energie_par_poste.py 1 22`.

| b=1, par jeton | acvram | llama.cpp | rapport |
|---|---:|---:|---:|
| Go DRAM lus (ncu, cache chaud) | **2,299** | **1,946** | **+18 %** |
| dont MoE (8 experts × 48) | 1,020 | ≈ 1,22 (mul_mat_vec_q_moe 0,54 + mul_mat_vec_q<·,1> 0,68 mêlé) | |
| dont projections d'attention | 0,927 (int8) | (dans mul_mat_vec_q, Q4_K ≈ 0,5) | |
| dont lm_head | 0,318 (int8) | (Q6_K ≈ 0,24) | |
| G instructions warp | 0,927 | 1,078 | ×0,86 |
| **inst / octet** | **0,40** | **0,55** | llama.cpp ×1,4 **de plus** |
| ms par jeton (moteur en processus / serveur) | 4,18 (239 t/s) | 3,16 à 3 000 jetons de contexte, 3,50 à 8 000 (317 → 285 t/s) | ×0,76-0,84 |
| W brut / net (fenêtre ≥ 20 s) | 329,5 / 287,5 (2 962 MHz, pas de bridage) | 396,7 / 363,3 (**plafond 400 W atteint**) | llama.cpp ×1,26 net |
| repos chaud, modèle chargé | 42,0 (moteur en processus) | **33,3** (llama-server, 225 MHz) | |
| J par jeton brut / net | 1,377 / 1,200 | 1,390 / 1,273 (8 000 jetons) ; 1,250 / 1,140 (3 000, fenêtre 9,8 s invalide) | ≈ égaux |

Contre les seuils scellés de poste7 (`poste7-organisation-14-09.md`) :

* « llama.cpp ≤ 0,5 inst/octet » : **0,55** — au-dessus, sous le seuil de
  réfutation (≥ 1,0). Leurs `mul_mat_vec_q` K-quants font 0,5-0,7
  inst/oct ; **notre GEMV à b=1 fait 0,40** : à un jeton nos noyaux ne
  dépensent PAS plus d'instructions par octet qu'eux (c'est à b=12 que
  la réutilisation ×12 en FMA nous coûte 1,73).
* « octets ±10 % des nôtres » : **réfuté, +18 % chez nous** (2,30 vs
  1,95 Go) — l'écart est exactement celui de poste3 sur J/jeton (−18 %), et
  il est nommé : projections d'attention en int8 (0,93 Go) et lm_head int8
  (0,32) là où Q4_K_M lit du 4,5-6,5 bpw (≈ 0,5 + 0,24).
* « repos chaud 60-80 W comme nous » : **réfuté, 33,3 W** pour llama-server
  chargé (≤ 40 = « plancher » selon poste7) ; notre moteur en processus :
  42 W. Les 68-76 W de poste3 sont ceux du serveur HTTP acvram, pas du
  moteur.
* « la différence est dans les noyaux, W nets ≥ 1,2× les leurs » :
  **réfuté dans l'autre sens** — leurs W nets sont ×1,26 les nôtres
  (363 contre 288) ; ils vont plus vite parce qu'ils lisent moins d'octets
  (−15 %) et les lisent plus vite (555-615 Go/s contre 550), au prix du
  plafond de puissance. **En J/jeton, même instrument, même heure :
  1,38 contre 1,39 brut, 1,20 contre 1,27 net — pas de −18 %.** Le −18 %
  de poste3 (1,168 vs 1,430) compare deux serveurs HTTP à repos différents
  et sur un contexte différent : à refaire en processus avant d'en tirer
  un chantier.

Ce que ça ordonne à b=1 : le seul levier nommé est **les octets des
projections d'attention et du lm_head** (int8 → NVFP4 = −0,6 Go, −26 % des
octets du jeton) — c'est 1aj décodage, pas la MMA MoE (à M=1 notre GEMV
est déjà à 0,40 inst/oct et 1 017 Go/s sur gate·up). Contrôle qui peut
rendre faux : 1aj à b=1 doit rendre ≥ 3,6 ms/jeton (−14 %) ; s'il rend
≥ 4,0, les octets n'étaient pas le goulot à b=1.

Données : `revue/donnees-ncu-ipo-llamacpp-b1-14-09.txt`,
`revue/donnees-ncu-ipo-acvram-b1-14-09.txt`. Contention signalée par
poste2 (forward GPU de 2 couches hors verrou pendant le ncu acvram b=1,
10:20-10:25) : sans effet sur les COMPTES (octets, instructions), seules
les durées ncu — non utilisées — pouvaient bouger.

### Étape 2 (poste7) — bt=32 par noyau : sans objet, chiffré

À b=12, `bras experts` : max 12 jetons par expert sur 48 couches (GEMV et
MMA), 30,7-34,0 experts distincts par couche ; une tuile de 16 ne se
remplit jamais, aucune n'est doublée, les octets ncu = experts distincts
× 2,654 Mo à 98-100 %. bt=32 ne peut rien enlever (il ne changerait
qu'à ≥ 17 jetons par expert, soit b ≥ 35 en godet). Pas de manche.
