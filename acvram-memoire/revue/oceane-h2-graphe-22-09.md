# H2 — le graphe de décodage b=12 par famille de noyaux : levier 3 chiffré, l objectif reste à portée (22/09, Océane, à sec)

* sources : budget nsys 19/09 (`laurine-scelle-b12-21-09` § a, Σ noyaux 6,873 ms, 1 169 lancements/pas, horloge libre), `verdict-m2-b12-21-09` (D = 42 experts/couche, Marlin à 94 % du plancher, `_etroit` servi 0,71 ms), JSON frontière 22/09 (graphe 6 725 µs à 2 700 sous sampler=graphe), `laurine-h2-frontiere-pas-21-09` (H2 « glue MoE » fermée : 13 lancements fixes par couche, la glue est de la frontière — désormais leviers 1-2)
* instrument livré : `outils/gpu/mesure/familles-noyaux.py` (une trace nsys → ms et lancements par pas et par famille, un pas = 48 `_route_fusee_kernel` ; test synthétique 2/2 ; validé sur les traces du 19/09 : Marlin 3,55-4,23, étroit 1,09, attention 0,39-0,45, routage 0,32, normes 0,22, rope/kv 0,21-0,23, tête 0,12 — mêmes chiffres que le budget de Laurine ; leur `glue_torch` 4,6 ms / 3 584 lancements est le DUMP d experts de cette prise, pas le service)
* état : levier 1 tenu (−54 µs), levier 2 codé (7f967671, prédit −160 µs) → pas ≈ 6 750 µs, écart à vLLM −0,75 % ± 0,3 (1 584 contre 1 596). **Il manque ≥ 0,8 % = 55 µs sur le graphe pour passer devant.**

## 1. Le graphe (6 725 µs à 2 700) par famille — ce qu on sait, ce qui manque
| famille | ms/pas (nsys 19/09, libre) | lancements/pas | plancher / statut | marge |
|---|---|---|---|---|
| experts Marlin gate·up + down | 3,73 | 96 | 3,50 à 1,55 To/s sur D = 42 (94 %) — H4 close | ≤ 0,2 |
| projections étroites int8 q/k/v/o | 1,07 (servi 0,71) | 97 | qkv 1,57 To/s ; o 1,03 To/s (K = 4 096) — H3 close | ≤ 0,13 (o) |
| attention paginée `_partiel` + `_reduce` | 0,41 (ctx ~300) | 96 | croît avec la longueur (T1 : 768 moy.) | à mesurer à 2 048 |
| routage `_route_fusee` | 0,33 | 48 | **7 µs = latence pure** ; « un nœud » déjà essayé sur GLM : 14,8 µs, plus lent (REGLES § 3) | 0,15 si fusion avec la norme, à bancher à sec d abord |
| normes rmsnorm | 0,23 | 97 | 2,4 µs : latence | dans une fusion épilogue |
| rope + kv_write | 0,22 | 96 | | ≤ 0,05 |
| tête wmma | 0,125 | 1 | + argmax capturé 4 µs | — |
| **glue torch** : reduce fp32 ×96, élémentaires ×144, casts ×96, Fill ×97, splitK, divers | **1,09** | **433** | **2,5 µs/nœud = latence de lancement dans le graphe ; aucun n est un calcul** | **0,5-0,8** |
| copies (preparer, D2D) | 0,08 | ~10 | | — |
Ce que le budget 19/09 ne dit pas : (i) l horloge (libre vs 2 700 : −4 à −7 % sur les noyaux, Laurine) ; (ii) la longueur (attention à 2 048) ; (iii) **ce que sont les 2 `reduce_kernel` fp32 et les 3 élémentaires par couche** — hypothèse la plus probable, à lire dans `moe.py`/`attention.py` avant de coder : la quantification d activation A4 (`chemin_moe=mma-a4`) = un `amax` par jeton (reduce fp32) + cast/pack (élémentaire) AVANT gate·up et AVANT down → 2 reduce + 2-3 élémentaires + casts par couche, soit ~7 des 9 nœuds de glue par couche.

## 2. Mesure d abord (Manon, ≤ 5 min de carte, arbre 0.6.35 candidat, 2 700, leviers 1+2 posés)
`nsys profile -t cuda --cuda-graph-trace=node -o graphe outils/carte.sh python outils/gpu/mesure/frontiere-pas.py f.json 12 60` puis `nsys stats --report cuda_gpu_trace --format csv -o graphe graphe.nsys-rep` et `familles-noyaux.py graphe_cuda_gpu_trace.csv --json familles.json` (60 pas, ~1 min de carte, le reste est le chargement).
**Prédit** (par pas, 2 700, ctx 256+) : experts 3,55-3,75 · étroites 0,70-0,80 · attention 0,40-0,50 · routage 0,33 · normes 0,23 · rope/kv 0,21 · tête 0,13 · **glue_torch 0,90-1,10 avec 420-440 lancements** · copies ≤ 0,10 · Σ ≈ 6,6-6,9 ; mur − Σ = trou ≤ 0,05 (levier 2). **Alarme** : glue_torch > 1,3 ou > 500 lancements → un nœud de plus par couche est apparu depuis le 19/09 (levier 1 n en ajoute qu un par pas) ; Σ > 7,0 → l horloge n était pas 2 700.

## 3. Levier 3 — la glue A4 dans l épilogue des experts (H2 « glue-first », Laurine 20/09, jamais codé)
Mécanisme : `amax` + quantification A4 + pack de x calculés **dans le noyau qui produit x** (rmsnorm/add_norm pour l entrée de gate·up, `moe_act` silu·up pour l entrée de down) au lieu de 3-4 nœuds torch après coup ; le reduce fp32 de sortie dans `moe_reduce`. Par couche : 24 → ~15 lancements.
* **Invariant au bit** : la quantification A4 est déterministe par jeton (amax puis arrondi E2M1) : fusionnée ou séparée, mêmes octets d activation → **ids au bit b=1/b=12** par construction ; test à sec : sortie du noyau fusionné == chemin torch actuel, au bit, sur entrées réelles (jouet CPU pour le pack, carte pour l amax en fp32 — même ordre de réduction requis : un amax par bloc de 16 est associatif sur les max, pas sur les sommes : au bit sans condition).
* **Plafond** : 433 → ~270 nœuds = 160 × 2,5 µs ≈ **0,40 ms** ; plus les Fill ×97 (0,08) si l accumulation MoE écrit au lieu d ajouter : **≤ 0,48 ms = 7 %**.
* **Prédit** : −0,30 à −0,45 ms/pas (−4,5 à −6,5 %), t/s **1 584 → 1 655-1 690**, soit **+3,7 à +5,9 % devant vLLM** ; J/jeton −4 à −6 %.
* **Réfuté** si `familles-noyaux` sous levier 3 ne retire pas ≥ 100 lancements/pas, ou si le gain < 0,15 ms (les nœuds retirés n étaient pas à la latence), ou ids ≠ au bit (défaut).
* Coût : 2 jours (Laurine l avait chiffré) ; prérequis : la mesure § 2 pour nommer les 9 nœuds/couche — **pas une ligne de noyau avant**.

## 4. Verdict sur l objectif b=12
**Pas hors de portée par la frontière** : leviers 1+2 la ramènent à ≤ 30 µs (2,2 % gagnés) et laissent −0,75 % ; le levier 3 (glue A4, 0,3-0,45 ms) est le seul poste du graphe à latence pure assez gros pour passer devant, avec 3-6 % de marge ; le routage (0,15) et `o` (0,13) viennent après, Marlin et l attention sont à leur plancher. Ordre : § 2 (Manon, 5 min) → nommer les 9 nœuds (Océane, à sec, moe.py/attention.py) → levier 3 sous scellé ids au bit + `familles-noyaux` A/B + ABBA.
