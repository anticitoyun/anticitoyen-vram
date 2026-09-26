# Pièce 42 — attribution des 591 lancements de glue (trace du contrôle 2 de poste5), à sec — 22/09 (poste1)

## Prédictions et issues, écrites AVANT de lire la trace

* instrument : `scratchpad/poste5-p42-22-09/03b-graphe_cuda_gpu_trace.csv` (prise
  poste5, arbre `41a69f47`, alias B `…-nvfp4-qkv-alpha2-22-09`), relu avec le
  découpage EXACT de `outils/gpu/mesure/familles-noyaux.py` (marqueur
  `_route_fusee_kernel`, 48 couches, exclusion (2, 1)) et le motif `glue_torch`
  tel quel ; 0 min de carte, `nice 19`.
* **contrôle d'abord : ma population doit être la sienne.** Si je ne retrouve pas
  `glue_torch` à **0,964 ms/pas ± 3 %** ET **591 lancements/pas ± 2**, je ne
  décompose rien et je le dis : je n'aurais pas lu le même objet.
* excédent à attribuer : 0,964 − 0,044 = **0,920 ms/pas**, 591 − 17 = **574
  lancements/pas** (officiel : verdict `poste5-piece42-controle2-22-09.md`).
* méthode du site d'appel (déclarée, faute de NVTX dans cette prise) : appariement
  temporel — un noyau de glue est dit *solidaire des projections séparées* s'il
  s'exécute dans les 5 µs qui précèdent un `_dense_etroit_kernel`, et si son
  compte par pas est un multiple de 48 compatible avec 3 ou 4 GEMM par couche
  (144 ou 192). Tout autre site est nommé par son voisin.

| issue | condition chiffrée | ce qu'elle rend |
|---|---|---|
| **I1 solidaire** | ≥ 80 % de l'excédent (**≥ 0,736 ms/pas**) apparié aux `_dense_etroit_kernel` d'attention, et le reste ≤ 0,15 ms au-dessus des 0,044 officiels | (a) se scelle : la glue part avec les trois GEMM |
| **I2 défaut propre** | ≥ 20 % de l'excédent (**≥ 0,184 ms/pas**) hors de cet appariement, sur un site nommé fichier:ligne | (b) se joue sur un arbre corrigé ; la voie n'est pas close |
| **I3 les deux** | les deux seuils atteints | les deux parts chiffrées, verdict au poids dominant, aucune des deux tue |
| **I4 population fausse** | le contrôle ci-dessus échoue | aucun verdict ; je rends l'écart et je m'arrête |

Ce qui rendrait « solidaire » faux : un noyau de glue au-delà des 17 officiels
qui ne précède aucun `_dense_etroit_kernel`. Ce qui rendrait « défaut propre »
faux : la totalité des 574 lancements excédentaires appariée aux GEMM séparés.

## Mesure (0 min de carte) — verdict : **défaut propre** (I3, les deux parts, experts majoritaires)

* population reproduite **exactement** : médiane par pas, découpage et motif de
  `familles-noyaux.py` → **0,964 ms/pas, 591 lancements/pas** (I4 écartée). La
  *moyenne* vaut 1,253 ms / 699 lancements : la glue est très dispersée d'un pas
  à l'autre — à retenir, le mur ne se lit pas sur la médiane.
* site = premier noyau **non-glue** qui suit, dans la fenêtre du pas (pas de
  NVTX dans cette prise) ; les 16 premières lignes couvrent 0,960 ms / 587 des
  591 lancements.

| site (noyau suivant) | ms/pas | lanc./pas | par couche | origine |
|---|---|---|---|---|
| `_dense_etroit_kernel` (projections d'attention) | **0,446** | 190 | ~4 | mise à l'échelle de x avant chaque GEMM séparé — cause 1 de poste5 |
| `nvfp4_gemv_marlin_kernel<__nv_bfloat16, 2>` (gate+up fusionnés) | **0,230** | 192 | 4 | `moe.py:1199-1202` : `x[tok64]`, cast, `tg[eid64]`, division |
| `nvfp4_gemv_marlin_kernel<float, 1>` (down) | **0,243** | 192 | 4 | `moe.py:1296` : cast bf16, `awq["down_proj"][eid64]`, division, `.contiguous()` (1301) |
| memset / memcpy / fin de pas | 0,041 | 13 | — | hors sujet |

* excédent sur l'officiel (0,044 ms / 17 lancements) = 0,920 ms : **0,446 ms
  (48 %) solidaire des trois GEMM séparés**, **0,473 ms (52 %) aux experts**,
  384 lancements = 8 par couche. I1 est donc **fausse** (le solidaire ne fait
  pas ≥ 0,736), I2 est **vraie** (0,473 ≫ 0,184) : les deux seuils portent, le
  poids est du côté du défaut propre.
* **ce défaut n'a rien à voir avec nvfp4.** Il est gardé par
  `awq.get("gate_proj") is not None or distinct` (`moe.py:1186`) et
  `awq.get("down_proj") is not None` (`moe.py:1294`) : l'alias officiel n'a
  **aucune** échelle d'expert (toutes à l'identité, écartées `moe.py:257`), donc
  il saute la branche entière. **Tout alias dont les experts ont des
  statistiques AWQ réelles le paie — y compris un alias q/k/v/o int8** ; c'est
  la conséquence directe du correctif de collecte `90c371f7` (22/09).
* **conséquence sur le contrôle 2** : B et A ne diffèrent pas d'un axe mais de
  **deux** (projections nvfp4 *et* experts calibrés). Les 0,473 ms de glue
  d'experts sont dans les 8,078 ms de B et absents de A : la comparaison B/A ne
  mesure pas le format des projections seul, et « exacte mais sans gain » ne
  peut pas se sceller sur elle.

## Remèdes, du moins cher au plus exact

1. **Repli à la conversion, côté `down` seulement** — 0 ligne de noyau. `act` est
   `gelu(g) · u` du **même** expert, donc `act / s_down[e]` s'obtient en divisant
   les lignes de sortie de `up_proj[e]` par `s_down[e]` avant quantification.
   Gain ≈ **0,243 ms/pas** (4 lancements/couche). **Pas au bit** : `up` est
   re-quantifié en nvfp4 sur une matrice autrement échelonnée — l'écart est
   celui d'une quantification ordinaire, pas une perte structurelle, à juger par
   KL (seuil 1,0, référence 0,519) et PPL à 2 SE. Ne s'applique pas à
   `gate`/`up` : leur échelle dépend de l'expert alors que `x` est partagé.
2. **Paramètre d'échelle optionnel dans le GEMV** — `nvfp4_gemv_marlin_kernel<XT,
   PX>` (`kernels/acvram_kernels.cu:2022`, PX=1 down, PX=2 gateup) reçoit déjà
   `eid` ; un pointeur `s` de plus et la division à la lecture de `x`. Gain
   **0,473 ms/pas** (8 lancements/couche), **au bit si** la division reste en
   bf16 avant accumulation, dans l'ordre de `moe.py:1201`/`1296`. Coût : deux
   noyaux, le binding, les tests d'équivalence dans le même commit.
3. Repli des seuls gathers d'index en un noyau Triton (gather+div+cast) : 8 → 3
   lancements/couche, au bit, mais c'est la moitié du gain pour autant de code
   que (2).

Le moins cher qui rende un chiffre honnête est **(2) restreint à `down`** : un
noyau, au bit, ≈ 0,243 ms/pas, et il ouvre (2) complet sans rien jeter.
