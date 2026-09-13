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

### Trouvaille : le godet 16 relit les poids des projections

`int8_gemv_kernel<4, NV>` traite NV colonnes (lignes du lot) ; à 12
séquences en godet 16 (`graphs.py:45`, `bucket_batch`), chaque projection
et le lm_head sont lancés DEUX fois : `<4,12>` pour les 12 vraies lignes,
puis `<4,4>` pour les 4 fantômes — et le second lancement **relit tous les
poids** : 0,932 + 0,319 = **1,25 Go par pas, 19 % des octets DRAM du pas**,
1,65 ms ncu. Le masquage des fantômes d'poste1 (routage/experts) ne
couvre pas ce chemin. Levier immédiat, sans nouveau noyau : godet 12 (ou
NV=16 en un lancement). Prédiction : −1,2 ms par pas de rejeu (14,5 →
13,3, −8 %) et −19 % d'octets ; réfutation < −0,6 ms.

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
| MoE gate·up | 2,556 | 2,43 | 3,35 | 73 % |
| MoE down | 1,296 | 1,23 | 2,63 | **47 %** |
| MoE total | 3,852 | 3,67 | 5,98 | 61 % |
| projections int8 (12 lignes) | 0,940 | 0,90 | 2,62 | 34 % |
| lm_head int8 (12 lignes) | 0,319 | 0,30 | 0,88 | 35 % |
| attention paginée | 0,137 | 0,13 | 0,72 | 18 % |
| pas entier (rejeu) | 6,667 | 6,35 | 14,5 | 44 % |

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
| Go DRAM lus par pas | 6,67 (5,42 sans fantômes) | 5,98 | ×1,1 |
| G instructions par pas | 9,57 | 1,19 | **×8,0** |
| inst / octet | 1,44 | 0,20 | ×7,2 |

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

1. **Godet 12 / NV=16** (trafic fantôme, 1,25 Go/pas) : −1,2 ms/pas,
   −19 % d'octets ; pas de nouveau noyau. Réfutation < −0,6 ms.
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
