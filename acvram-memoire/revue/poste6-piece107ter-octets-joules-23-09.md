# Pièce 107 ter — octets et joules par pas, à sec : S1 contre i8c, et les options mixtes contre l'écart vLLM (−6,96 %) — 23/09 (poste6, écrit AVANT la prise du second filtre et de la 107 bis)

Méthode : celle de la 99. Un noyau borné mémoire coûte ses octets, mais le pJ/bit monte quand le noyau s'éloigne du
plancher : sur les deux points mesurés en isolé (qkv int8 1,10 To/s à 312 W nets ; o int8 0,78 To/s à 237 W),
**P_net ≈ 54 W + 234 W × BW(To/s)**, d'où **pJ/bit = (54/BW + 234)/8** : 35,4 (int8 qkv), 37,9 (int8 o), 37,6 (Marlin fp4
vLLM à 0,81), **44,2 pour NOTRE noyau nvfp4 étroit à 0,45 To/s** (pièce 42), 33,8 (tête int8 à 1,48). Experts : identiques
pour tous ces alias (nvfp4 4,5 bpw, 1,14 J/pas mesuré à la 99, 2,94 ms). b = 12, 48 couches ; référence i8c 1,617 J/pas
servi ; écart vLLM 0,104 J/pas. Octets par couche : q 8,39 / 4,72 Mo (int8 / nvfp4 + échelles), k = v 1,05 / 0,59,
o 8,39 / 4,72 ; tête 311 / 175 Mo.

| alias | proj Go/pas | proj J | proj ms | GEMM/couche | tête Mo | tête J | tête ms | **Σ J/pas** (experts + proj + tête) | **ΔJ contre i8c** | Δ J/jeton | Δ ms noyaux | part de l'écart vLLM fermée |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **i8c** (servi) | 0,906 | 0,265 | 0,974 | 2 | 311 | 0,084 | 0,210 | 1,489 | — | — | — | 0 % |
| **S1** proj int8, tête nvfp4 | 0,906 | 0,265 | 0,974 | 2 | 175 | 0,062 | **0,389** | 1,467 | **−0,022** | −1,4 % | **+0,18** | 21 % |
| A tout nvfp4, notre étroit | 0,510 | 0,180 | 1,133 | 2 | 175 | 0,062 | 0,389 | 1,382 | −0,106 | −6,6 % | +0,34 | 102 % |
| A tout nvfp4, Marlin dense (100 A) | 0,510 | 0,162 | 0,785 | 2 | 175 | 0,062 | 0,389 | 1,364 | **−0,125** | −7,7 % | −0,01 | **120 %** |
| S3 v/o int8 | 0,708 | 0,227 | 1,129 | 4 | 175 | 0,062 | 0,389 | 1,429 | −0,060 | −3,7 % | +0,33 | 58 % |
| S4 v/o + tête int8 | 0,708 | 0,227 | 1,129 | 4 | 311 | 0,084 | 0,210 | 1,451 | −0,038 | −2,4 % | +0,15 | 37 % |
| **S6** k/v int8 | 0,554 | 0,189 | 1,099 | 4 | 175 | 0,062 | 0,389 | 1,391 | **−0,098** | −6,1 % | +0,30 | **94 %** |
| **S7** v int8 | 0,532 | 0,185 | 1,116 | 4 | 175 | 0,062 | 0,389 | 1,387 | **−0,102** | −6,3 % | +0,32 | **98 %** |
| B o int8 | 0,686 | 0,222 | 1,146 | 2 | 175 | 0,062 | 0,389 | 1,424 | −0,065 | −4,0 % | +0,35 | 62 % |

Trois corrections que la table ne porte pas :
1. **Les GEMM séparées (4 par couche, options mixtes sans pile partielle) coûtent plus que leurs octets** : k et v seules
   tombent à 0,31 To/s (pièce 42 : « kv seule ») et la pièce 42 a mesuré **+0,9 ms/pas** pour q/k/v en trois GEMM. À
   ~120 W nets d'oisiveté partielle, +0,9 ms vaut **+0,11 J/pas** : sans pile partielle, S3/S4/S6/S7 **perdent** ce qu'ils
   gagnent. La pile partielle par format (`attention.py::fuse`, 2 lancements) est la condition, pas une option.
2. **Le temps compte dans J/jeton** : notre noyau nvfp4 étroit (0,45 To/s) allonge le pas (+0,3 ms sur A, +0,18 sur S1 par
   la seule tête) ; l'oisiveté hors noyaux à ~100-150 W ajoute +0,02-0,05 J/pas non comptés dans la colonne ΔJ. Avec le
   Marlin dense (100 A) ce terme disparaît (Δ ms −0,01).
3. La tête nvfp4 n'a jamais été chronométrée (0,45 To/s supposé, même noyau que les étroites) ; à 1,0 To/s elle serait à
   0,175 ms et S1 serait plus rapide que i8c.

## Réponses (prédictions, avant mesure)

* **S1 remplace-t-il i8c sans perdre d'énergie ?** Oui en joules de noyaux (**−0,022 J/pas, −1,4 %**, la tête lit 136 Mo
  de moins), mais **il perd du débit** si la tête nvfp4 tourne à 0,45 To/s : +0,18 ms/pas ≈ **−3 % de t/s**, et l'oisiveté
  ajoutée (+0,02 J) annule le gain : **bilan énergie ≈ 0 ± 0,02 J/pas, débit −3 %**. Prédit pour la cellule 101 sur S1 :
  t/s 1 935-1 975 (i8c : 1 995), J/jeton 0,133-0,136 (i8c 0,1349). Réfuté si S1 ≥ 1 990 t/s (la tête nvfp4 va au plancher).
  **S1b** (assemblé, 0 min : experts de A + projections ET tête de i8c) a les octets et le temps de i8c au bit près :
  **ΔJ = 0, Δt = 0**, seule la qualité change (KL) — c'est le remplacement sûr, s'il tient la KL comme S1 (S1 = 0,044 sur
  l'invite 11 avec la tête nvfp4 ; S1b est mesuré en information dans la prise, sans imposer la tête).
* **Quelle option ferme le plus l'écart vLLM ?** Par les octets : A avec le Marlin dense (120 %, ferme tout et rend
  0,01 ms), puis **S7 (v int8, 98 %) et S6 (k/v int8, 94 %)** — mais ces deux-là n'existent qu'avec la pile partielle, et
  leur qualité n'est pas connue (second filtre : prédit S6 0,30-0,45 et S7 0,40-0,50, donc **écartés** ; s'ils passent,
  ce sont les meilleurs candidats). S3 (58 %) et B (62 %) ferment la moitié. Sans le noyau de la 100 A, aucune option
  nvfp4 ne ferme l'écart à débit égal : elles échangent des joules de noyaux contre du temps.
* Ordre des paris : (1) 100 A (Marlin dense) + A ou S7, (2) pile partielle + S7/S6 si le filtre les garde, (3) S1b comme
  remplacement de i8c à énergie égale et qualité meilleure — indépendant des deux premiers.

Ce qui réfuterait le modèle : une mesure isolée de la tête nvfp4 ou des étroites nvfp4 à ≥ 55 pJ/bit (le floor de 54 W
serait faux), ou une cellule S1 à J/jeton < 0,131 (la tête nvfp4 pèserait moins que prévu).

## Erratum 21 h 4x (cellule 101 de poste2, fusionnée) : A sert 1 544-1 613 t/s à b=12 contre 1 995 pour i8c
Ma table prédisait +0,34 ms/pas pour A (−5 % de débit) ; la mesure dit −20 à −23 % (≈ +1,4 ms/pas). **Prédiction
réfutée** : le chemin nvfp4 étroit en service coûte 4 fois ce que son débit isolé (0,45 To/s) laissait prévoir — tête
nvfp4 à N = 151 936, lancements, ou un chemin non capturé ; l'attribution est la pièce 118 (poste2 : i8c, S1b, S8, S4).
Conséquence : la colonne « Δ ms » de la table ne vaut que pour les noyaux isolés ; aucun alias nvfp4-projections ne
remplace i8c avant la 118, et le bilan « S1 ≈ énergie égale » est suspendu (il dépend du temps de la tête nvfp4).

## Addendum 22 h 4x — S1b contre i8c (cellule 118 de poste2) : le coût AWQ des experts est du CALCUL, pas de l'octet

Fait (118) : S1b 1 645,6 t/s, 0,1918 J/jeton contre i8c 1 995,0 t/s, 0,1358 J/jeton ; seule différence : les experts de A
(AWQ calibré + alpha commun, tables d'activation non unité) contre ceux de i8c (sans AWQ, tables d'unité). Régime lu dans
mes journaux KL : i8c `chemin_moe=…(atteint=gemv_marlin+marlin_tensor)`, `echelle_awq=aucune` ; S1b
`(atteint=gemv_marlin)+tensor(b≥8, repli : tables AWQ d'activation par expert non unité …)`, `echelle_awq=gemv(48/48)`
(`moe.py:1911`) : **à b=12, S1b ne prend pas le chemin tensor-core, il retombe sur le GEMV** (le bras A de la pièce 62 :
1 596 t/s, 390 W bruts, horloge bridée ~2 490 MHz, contre le tensor 1 791 t/s, 320 W, 2 664 MHz).

Décomposition (b=12, par pas de 12 jetons) :

| | i8c (tensor) | S1b (GEMV + échelle AWQ) | Δ |
|---|---|---|---|
| ms/pas | 6,02 | 7,29 | **+1,28 (+21 %)** |
| P nette moyenne (J/jeton × t/s) | 271 W | 316 W | **+45 W (+17 %)** |
| J/pas | 1,63 | 2,30 | **+0,67 (+41 %)** |
| octets experts par pas | 4,29 Go (nvfp4 4,5 bpw) | 4,29 Go + tables d'activation ≤ 0,05 Go | **≤ +1 %** |
| débit mémoire effectif du MoE (4,29 Go / t_MoE) | 1,46 To/s (2,94 ms) | ≈ 1,0 To/s (≈ 4,2 ms) | −30 % |

* **Octets : ≤ 0,013 J/pas** (0,05 Go × 37 pJ/bit), soit ≤ 2 % de l'écart. Les poids d'experts sont les mêmes octets.
* **Calcul : ≈ 0,66 J/pas**, en deux parts qui se lisent avec la formule de la 99 (J = P_net × t) : (a) le pas dure
  1,28 ms de plus à 316 W → **+0,40 J** (le GEMV lit les mêmes octets 30 % moins vite : dequant et échelle x/s sur les
  cœurs CUDA au lieu des tensor cores) ; (b) sur les 6,0 ms d'origine, la carte tire 45 W de plus → **+0,27 J** (chemin
  scalaire au plafond de puissance : 390 W bruts, horloge bridée — la 62 l'avait nommé « bridage »). Le pJ/bit du MoE passe
  de ≈ 33 (tensor, 1,14 J pour 4,29 Go) à ≈ 52 (GEMV, ≈ 1,8 J) : c'est la signature d'un noyau qui n'est plus borné mémoire.
* **Ce que la 123 d'poste1 (échelle AWQ fusionnée dans le chemin tensor, au bit) doit rendre, écrit avant** : S1b
  **1 950-2 000 t/s et 0,134-0,140 J/jeton** (= i8c à 2 %) si l'échelle est portée par la quantification d'activation ou
  les poids sans lancement de plus ; **1 900-1 950 et ≤ 0,145** si elle coûte un noyau élémentaire x/s par groupe
  d'experts (+0,05-0,10 ms/pas, +0,02-0,03 J/pas). **Réfuté** (le coût AWQ ne serait pas le repli GEMV) si S1b-tensor
  reste ≤ 1 850 t/s : alors l'échelle coûte dans le GEMM lui-même (lecture des tables par expert, 34 × 8 Ko par couche,
  ou un chemin `alpha commun` gate/up qui refuse la pile w13).
* Corollaire pour la 99 : le lever « octets » ne vaut que sur le chemin tensor ; tout alias AWQ d'experts servi sans la
  123 rend +41 % de J/jeton, quelle que soit sa qualité.
