# 61w — état KDA en [K, V] partout, AU BIT partout attendu : le noyau b=1 REPRODUIT l'itération j=0 que le témoin contracte autrement à D=128 (décision chef 10:40, après le rouge d'poste1 10:35) (poste5, 01/10)
* instrument : `tests/test_kda_etat_kv.py` (+ test_hybrides_fla_lot, test_gdn_coherence) ; prises `scratchpad/poste5-kda-01-10/prise.sh <commit>`, sorties `prise-0844.txt`, `prise2.txt` ; SASS `kda_noyaux.cu` → `kda_rapide.sass` (nvcc --use_fast_math, sm_120f : les drapeaux de l'extension)
* commit : poste5-kda, HEAD donné à poste1 (voir git log) ; extension = .so précompilé sans carte `precompiles/98af5877cb43721c` (empreinte = le .cu, sha256 093921d9e795c5dd…, hors dépôt : REGLES:162), `_precompile_utilisable` → « précompilé » (aucun JIT), prise.sh refuse sinon (rc 66)
* régime : carte 0 (RTX 5090, 01:00.0) sous `carte.sh` mesure ; avant/après : seul le llama-server 4436 sur la 3080 Ti (02:00.0) ; arbre importé contrôlé (ARBRE et acvram.__file__ = poste5-kda) ; aucune mesure de temps (elle ira à poste2 ou à poste1)
* scellé : prédiction d'poste1 (revue/poste1-p81-cake-kda-01-10.md, « Levier ») — non jugée ici ; au bit exigé, test qui casse si une transposition revient
* mesuré : 11/12 verts ; rouge : test_noyau_b1_au_bit_du_temoin[128] — « D=128 pas 0 : sortie bf16 0 éléments ≠ ; état fp32 1 183 éléments ≠, max 82,9 ulp »
* verdict : lot (b = 1, 3, 12) AU BIT ; b=1 AU BIT à D=64 ; à D=128 la version « homogène » changeait la sortie (poste1, carte 10:35 : bf16 au pas 41 malgré la resynchronisation) → j=0 reproduit, test au bit STRICT en pas chaînés (D=64 : 64 ; D=128 : 64 et 512) — à jouer par poste1 en étape 0
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
## D=128 : le témoin contracte une itération autrement (prise 09:11)
SASS du témoin à D=128 : d sert à **127 FFMA** fma(d, sk, s·e) et à **1 FMUL** `FMUL R32, R32, R204` (R204 = d) ; à D=64 il
est homogène. La version homogène donnait, à D=128, 1 183 éléments d'état ≠ au pas 0 (sur 524 288 : une ligne j).
Historique : l'arithmétique homogène (128 × fma(d, sk, s·e)) acceptée le matin a été REFUSÉE à 10:40 sur mesure — à
D=128 la sortie bf16 bougeait au pas 41 même en repartant du même état à chaque pas (effet d'un seul pas, ≈ 0,6 bascule
attendue sur 64 × 4 096 sorties), et en service l'écart se propagerait. **Reproduction déterministe** : SASS du témoin
D=128 (lignes 956-966) `FMUL R32, R32, R204` (d·sk[0]), `FFMA R176, R176, R160, R32` (srow[0]·se[0] + d·sk[0]),
`FFMA R160, R176, R160, RZ` (o = r·sq[0] + 0) ; le nouveau noyau détache j=0 et, à D=128, calcule
`__fmaf_rn(s0, se[0], __fmul_rn(d, sk[0]))`. **SASS relu du nouveau noyau** (mêmes drapeaux) : D=128 `FMUL R12, R2, R12`
(d·sk[0]) consommé par `FFMA R19, R7, R8, R12` — la forme du témoin ; D=64 aucune FMUL d·sk (homogène comme le témoin) ;
44 registres, 0 déversement. Sans carte et sous TRITON_INTERPRET : 42 passés, 6 sautés (carte).
## Reste
Étape 0 d'poste1 (tests au bit sur carte, même pause e50.2), puis mesure : b=12 −14 à −16 % du temps GPU, b=1 −9 à −11 % du mur (prédiction scellée d'poste1).
