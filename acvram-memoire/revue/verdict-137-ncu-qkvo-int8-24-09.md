# Verdict — 137 : ncu du plateau QKVO int8 (Coder qkvo-i8c), b=1 et b=8 — 24/09 06 h 3x (poste1)

* **instrument** : `scratchpad/poste1-p137-24-09/harnais.py` (poids int8 aléatoires au format, groupe 128, 8 copies, L2 froid, `kernels.int8_matmul` tel que routé) sous `sudo ncu --clock-control none --target-processes all`, filtres NVTX par forme ; lecture `lire-ncu.py` (médiane de 4 lancements par forme) ; prise `prise-ncu.sh` (set -euo pipefail)
* **commit** : 90def1d8 + correctif d'appel de ncu (première prise du même commit : aucune mesure — `sudo -n env …` exige une authentification ; ncu lance désormais `env` lui-même)
* **régime** : RTX 5090, -lgc 2700 posé par la prise, horloge non touchée par ncu ; cpu-safe=off (100/100) ; compute-apps début = fin ; prise 06:32:07, < 1 min
* **scellé** : `scratchpad/poste1-p137-24-09/scelle.md` (commit 90def1d8, avant la carte). Prémisse corrigée : les QKVO int8 sont ceux de **Coder** (QKV fusionné 5 120 × 2 048, O 2 048 × 4 096), pas de Qwen3.8.
* **mesuré** (chemins confirmés par CHEMINS_INT8 : M = 1 → `gemv`, M = 8 → `etroit_triton`) :

| forme | noyau | grille × bloc | regs | µs | DRAM % | warps actifs % | SM % | secteurs/req | 1re attente (part) |
|---|---|---|---|---|---|---|---|---|---|
| QKV, M = 1 | int8_gemv<4,1> | 1 280 × 128 | 48 | 9,46 | **64,7** | 57,9 | 19,1 | 12,5 | long_scoreboard **77,8 %** |
| O, M = 1 | int8_gemv<4,1> | 512 × 256 | 48 | 8,48 | **57,8** | 47,3 | 16,9 | 12,5 | long_scoreboard **73,2 %** |
| QKV, M = 8 | _etroit_reduit (Triton) | 320 × 128 | **146** | 10,85 | 56,5 | **15,4** | 14,3 | 9,1 | long_scoreboard 40,3 %, barrier 12,4, mio 12,4 |
| O, M = 8 | _etroit_reduit (Triton) | 352 × 128 | **146** | 11,41 | 43,2 | **16,8** | 11,4 | 9,2 | long_scoreboard 46,4 %, barrier 11,3, mio 10,4 |

* **verdict** : **FAIT**.
  * Contrôle d'instrument **tenu** : ncu retrouve le plateau (QKV 64,7 % dans [60 ; 80], O 57,8 % dans [55 ; 75]) — il est DANS le noyau, pas entre les lancements.
  * **Cause 1 nommée (prédite)** : à b=1, **latence DRAM non masquée** — `long_scoreboard` 73-78 % des attentes (seuil 40 %), SM à 17-19 %, 47-58 % de warps actifs ; à K = 2 048, chaque fil fait UN tour de chargement puis attend. Cohérent avec la 116 quater (ROWS 2 à 16 sans effet : le total d'octets en vol sur la carte ne change pas avec ROWS). Un noyau de 8-9 µs pour 10,5 / 8,4 Mo paie sa montée et sa queue.
  * Cause 2 (barrière ≥ 25 %) : **non** (≤ 5 % à b=1, 11-12 % à b=8).
  * Cause 3 (secteurs/requête hors [14 ; 18]) : seuil **franchi** (12,5), mais **mon critère était mal posé** : la métrique agrège tous les chargements globaux (poids `uint4`, x par mots de 8 o, échelles half, zéros octet), et ne mesure donc pas la coalescence des seuls poids. **Non attribuable** ; aucun défaut d'accès démontré.
  * **À b=8, fait nouveau** : le GEMM étroit Triton tourne à **146 registres par fil → 15-17 % de warps actifs**, avec une DRAM à 43-57 %. L'occupation y est bridée par les registres : c'est un levier distinct de celui de b=1.
* **durée** : 06:32:07 → ~06:33 (prise), carte rendue.

## Leviers (non codés, au chef)
* b=1 : masquer la latence d'un noyau court — lancement dépendant programmatique (PDL, recouvre la montée du GEMV suivant avec la queue du précédent) ou plus d'octets en vol par SM via un GEMV persistant ; gain borné par la part fixe (~3 µs sur 9 : QKVO b=1 ≈ −0,1 à −0,2 ms/pas, estimation à sceller avant).
* b=8 : `gemm_etroit` à ≤ 96 registres (occupation ×2) — `ptxas`/Triton `maxnreg`, à sec d'abord.

**ERRATUM 24/09 07 h (poste1, pièce 140 arrêtée à sec)** : le levier « b=8 : gemm_etroit à ≤ 96 registres (occupation ×2) »
était FAUX, lu dans les compteurs de cette même mesure (ncu.csv, `_etroit_reduit_kernel`, grille 320) :
`launch__occupancy_limit_registers` = 3 blocs/SM, `launch__occupancy_limit_shared_mem` = 3 (12 warps, 25 % théoriques),
mais **`launch__waves_per_multiprocessor` = 0,63** — la grille (320-352 programmes) ne remplit pas une vague : ≈ 1,9 bloc
par SM, d'où les 15-17 % de warps actifs. Les registres ne sont pas la limite active ; à 96 registres, la mémoire partagée
bornerait encore à 3 blocs, et la grille n'en fournit que 1,9. Les réglages warps/étages/BLOCK de ce noyau sont en plus
déjà réfutés au bit (pièce 35, ffa952d8 ; pièce 57, poste5, 36 variantes : « plus d'octets en vol = plus lent »).
