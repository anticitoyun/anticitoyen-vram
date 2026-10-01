# 61w — état KDA en [K, V] partout : lot AU BIT, noyau b=1 AU BIT à D=64 ; à D=128 « sortie identique, état fp32 différent d'un arrondi sur une ligne » — accepté par chef, tolérance nommée (poste5, 01/10)
* instrument : `tests/test_kda_etat_kv.py` (+ test_hybrides_fla_lot, test_gdn_coherence) ; prises `scratchpad/poste5-kda-01-10/prise.sh <commit>`, sorties `prise-0844.txt`, `prise2.txt` ; SASS `kda_noyaux.cu` → `kda_rapide.sass` (nvcc --use_fast_math, sm_120f : les drapeaux de l'extension)
* commit : b151bba52 (poste5-kda, NON poussé : le test au bit n'est pas vert partout) ; extension = .so précompilé sans carte (`precompiles/c9ad713763dafe14`, empreinte = le .cu), chargé par la prise (PRECOMPILE True)
* régime : carte 0 (RTX 5090, 01:00.0) sous `carte.sh` mesure ; avant/après : seul le llama-server 4436 sur la 3080 Ti (02:00.0) ; arbre importé contrôlé (ARBRE et acvram.__file__ = poste5-kda) ; aucune mesure de temps (elle ira à poste2 ou à poste1)
* scellé : prédiction d'poste1 (revue/poste1-p81-cake-kda-01-10.md, « Levier ») — non jugée ici ; au bit exigé, test qui casse si une transposition revient
* mesuré : 11/12 verts ; rouge : test_noyau_b1_au_bit_du_temoin[128] — « D=128 pas 0 : sortie bf16 0 éléments ≠ ; état fp32 1 183 éléments ≠, max 82,9 ulp »
* verdict : lot (b = 1, 3, 12) AU BIT ; b=1 AU BIT à D=64 (64 pas) ; à D=128 sortie identique, état fp32 différent d'un arrondi sur une ligne — DÉCISION chef 01/10 : arithmétique homogène acceptée, tolérance nommée (une seule ligne j, ≤ 2⁻²² (|S nouveau| + |S ancien|), sortie bf16 identique) dans `test_noyau_b1_d128_une_ligne_un_arrondi`, écrite après la décision et pas encore jouée sur carte
* durée : prises 08:44:41-08:44:48 (8 s, 2 rouges, cause ci-dessous) et 09:11:02-09:11:09 (7 s)
## Ce qui est livré
`kda.py` : S en [K, V] (disposition de fla) dans les statiques, `forward` (préfill par blocs sans transposition), `forward_batch`
et `decode_static_batch` — ce dernier appelle `fused_recurrent_kda_fwd` EN PLACE (h0 = ht = la tranche statique : chaque
programme lit sa tuile avant la boucle et l'écrit après), donc ni transposition ni copie (les 2,17 ms/pas de la nsys p81).
Les deux replis torch transposent à l'entrée et à la sortie (arithmétique d'avant, au bit). Noyau b=1 (acvram_kernels.cu) :
le fil i lit la COLONNE i (coalescé), deux passes j croissant sans `row[D]` : **44 registres, 0 déversement** (témoin :
255 registres, 48 o déversés). L'ancien noyau reste en témoin `kda_decode_vk` (jamais servi).
## Le premier rouge (08:44) et sa cause
Écrit en « r = s·e ; r += d·sk », le noyau était contracté par nvcc en fma(s, e, d·sk) — le témoin fait fma(d, sk, s·e) :
mêmes comptes d'instructions, autre arrondi (SASS relu ; ma première comparaison, compilée SANS --use_fast_math, ne
pouvait pas le voir). Corrigé par intrinsèques (`__fmul_rn`, `__fmaf_rn`) copiés du SASS du témoin ; le calcul de d (le
témoin fusionne la SiLU de v : FFMA acc·rcp − pred, puis FMUL β) laissé en C, relu identique.
## L'écart restant à D=128 : il vient du témoin
Dans le SASS du témoin à D=128, d sert à **127 FFMA** fma(d, sk, s·e) et à **1 FMUL** `FMUL R32, R32, R204` (R204 = d) :
pour UNE valeur de j, nvcc y a contracté fma(s, e, d·sk) — la FMUL de trop vue à sec hors boucle (23 contre 22). À D=64
le témoin est homogène (64 FFMA, 0 FMUL avec d), et le nouveau noyau y est au bit. Le nouveau noyau fait la même forme
pour les 128 j. Les 1 183 éléments ≠ (sur 524 288) tiennent dans une ligne j des 32 têtes (4 096 éléments) : cohérent
avec cette seule itération, pas prouvé élément par élément. La sortie bf16 est identique au premier pas.
Lecture proposée : le chemin servi d'avant 61w n'était pas « régulier » — reproduire au bit son irrégularité de
compilation (une itération sur 128) n'a pas de sens ; le nouveau noyau est l'arithmétique homogène. **Décision de chef : homogène accepté**
(REGLES § 1 : la règle « au bit » porte sur le défaut servi ; ici le défaut servi changerait de ≤ 1 arrondi fp32 sur une
ligne de l'état, sortie identique au premier pas). Si refusé : reproduire en dur cette itération (repérée dans le SASS du témoin), ou passer en opt-in jugé par KL.
Précompilé hors dépôt (REGLES : > 1 Mo non commité) : `travail/poste5-kda/scratchpad/poste5-kda-01-10/precompiles/d81e44210bd7866b/acvram_kernels.so`, sha256 e9aea4855cb18d74…, empreinte de source d81e44210bd7866b = le .cu du HEAD ; `_precompile_utilisable` à sec pour 12.0 → « précompilé » (aucun JIT) ; `prise.sh` refuse (rc 66) sinon.
## Reste
Mesure (poste2 ou poste1) : b=12 −14 à −16 % du temps GPU, b=1 −9 à −11 % du mur (prédiction scellée d'poste1).
