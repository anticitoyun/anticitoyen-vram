# Verdict — pièce 135 : Qwen3.8-27B b=8, unique + v2, balayage d'horloge SM (poste6, 24/09)

instrument : `scratchpad/poste6-p135-24-09/bloc.sh` (= bloc.sh de la 115 bis : `-i 0`, `ACVRAM_ECO=off`, ± 50 MHz lue/demandée pendant la fenêtre, 5 répétitions du palier par point ; serveur repéré par port + pgid), banc HTTP `scratchpad/banc-llamacpp-16-09.py decode` (BANC_SLOTS 8×5, 1 024 jetons, fenêtre ≥ 10 s), énergie nvml nette ; sorties `prise.txt`, `cellule-b8.tsv`, `banc-b8-<hz>.log`, `horloge-b8-<hz>.csv`
commit : bbfc42d3 (branche poste6 = main 3cd5b15a avec la 134 + pièce), asserté rc 65
régime : Qwen3.8-27B-nvfp4, b=8, `--max-model-len 2304 --speculative none`, bras B de la 129 (`ACVRAM_PROJ_MARLIN=1 ACVRAM_GEMV_MARLIN_V2=1 TPB=1 S=0 PROJ_MARLIN_DOUBLES=`) prouvé sur la ligne `[acvram] régime NOMINAL … dense=triton≥2|cuda+marlin(doubles=0,seuls=305)` aux 4 points ; `-lgc` {2 700, 1 800, 2 100, 2 400} dans cet ordre ; RTX 5090 seule ; cpu-safe=off (max_perf_pct 100 début et fin) ; compute-apps début = fin = llama-server 4627
scellé : `revue/poste6-piece135-scelle-24-09.md` (283965c5 + addenda 49e44d42, bbfc42d3 avant les chiffres) — borné mémoire si r_t(1 800) ≥ 0,85 ; calcul si ≤ 0,72 ; mixte entre ; prédit r_t(1 800) 0,80-0,88, minimum de J à 1 800 avec r_J 0,70-0,80, 440-490 t/s à 2 700
mesuré : 4/4 points tenus (horloge lue 2 656 · 2 379 · 2 076 · 1 782, écart ≤ 44 MHz ; NOMINAL ; W ≤ 365 à 2 700, pas de bridage). Médianes de 5 : **2 700 → 477,8 t/s, 0,617 J/jeton, 365 W · 2 400 → 445,6, 0,568, 312 W · 2 100 → 398,9, 0,548, 274 W · 1 800 → 346,0, 0,536, 240 W**. r_t = 1 · 0,933 · 0,835 · **0,724** (r_horloge 1 · 0,896 · 0,782 · 0,671) ; r_J = 1 · 0,919 · 0,887 · **0,868**. Dispersion ≤ 2,1 % en t/s, ≤ 1,3 % en J par point
verdict : **FAUX** — r_t(1 800) = 0,724, à la frontière « calcul » (≤ 0,72) : le débit suit l'horloge à 92 % de la proportionnelle (0,724 contre 0,671) ; **pas borné par la mémoire**, mixte à peine. Minimum de J à 1 800 MHz (prédit) mais **−13,2 %** seulement (prédit −20 à −30 %) ; débit 2 700 dans la bande (477,8)
durée : prévu ≤ 20 min ; tenu 1 375 s (22 min 55, quatre chargements de 12 s, banc 5 × 17-24 s par point) ; trois prises nulles avant (04:51, 05:08, 05:28 : `--regime` quitte, `setsid` forke — addenda)

## Chiffres
| horloge demandée | lue (MHz) | t/s (méd. 5) | J/jeton net | W | r_t | r_horloge | r_J |
|---|---|---|---|---|---|---|---|
| 2 700 | 2 656 | 477,8 (468,0-478,0) | 0,617 (0,616-0,620) | 365 | 1 | 1 | 1 |
| 2 400 | 2 379 | 445,6 (436,8-448,0) | 0,568 (0,567-0,574) | 312 | 0,933 | 0,896 | 0,919 |
| 2 100 | 2 076 | 398,9 (395,3-400,9) | 0,548 (0,542-0,549) | 274 | 0,835 | 0,782 | 0,887 |
| 1 800 | 1 782 | 346,0 (342,0-347,6) | 0,536 (0,530-0,537) | 240 | 0,724 | 0,671 | 0,868 |

## Lecture
* **La thèse « devenu borné par la mémoire » tombe** : de 2 656 à 1 782 MHz (−33 %) le débit perd 27,6 %. Un pas borné par la
  HBM aurait perdu ≤ 15 %. Le pas de Qwen3.8 à b=8 reste du calcul à ~90 % — comme le Coder i8c de la 115 bis (proportionnel
  strict). Le « ~1,5 To/s » n'est donc pas la borne : à 16,7 ms/pas (477,8 t/s ÷ 8) et ≈ 14-15 Go lus, l'HBM tourne à 0,85-0,9
  To/s, sous la moitié du plancher 1,79 — les GEMV v2 sont proches du plancher **quand ils tournent**, mais le pas contient
  encore ~1/3 de temps qui ne lit pas de poids (attention linéaire GDN/fla, glue, godets) et qui suit l'horloge.
* **Le levier d'énergie existe mais il est plus petit que prédit** : −13,2 % de J/jeton à 1 800 pour −27,6 % de débit ;
  −11,3 % à 2 100 pour −16,5 % ; −8,1 % à 2 400 pour −6,7 %. Le point 2 400 est le seul où l'énergie gagnée dépasse le débit
  perdu (ratio 1,2) ; à 2 100 et 1 800 on perd plus de débit que de joules. Même forme que la 115 bis (minimum à 1 800,
  −11,1 % sur i8c b=12) : rien de spécifique à la disposition unique + v2.
* Ma prédiction s'est trompée en surestimant la part mémoire : les 129/130 mesuraient le bras B à 367 W, non bridé, et j'ai lu
  « proche du plancher HBM » comme « borné par la HBM ». Un noyau au plancher pendant 60 % du pas laisse 40 % de calcul pur ;
  c'est ce 40 % qu'un balayage d'horloge mesure. La question « où passent les 1/3 du pas hors GEMV » est une pièce nsys, pas
  une pièce d'horloge.
* Contrôles tenus : horloge lue ≤ 44 MHz de la demandée aux 4 points (csv pendant la fenêtre) ; 365 W à 2 700 = pas de plafond ;
  serveur NOMINAL, `dense=…+marlin(seuls=305)` aux 4 points ; compute-apps début = fin.

## Pour chef
Rien à changer au défaut éco 2 700 sur cette base : à b=8 sur Qwen3.8, 2 400 MHz gagne 8 % de J pour 7 % de débit, 1 800
gagne 13 % pour 28 %. Si un profil « énergie » est voulu un jour, c'est 2 400 (pas 1 800) qu'il faut nommer, et il se juge
en service réel (b variable), pas au palier fixe.
