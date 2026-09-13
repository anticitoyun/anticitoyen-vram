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

## Résultats

(à remplir)
