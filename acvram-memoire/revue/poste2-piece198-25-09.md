# Verdict — 198 (poste2, 25/09, ordre chef) : relecture à sec de la pièce 195 d'poste6 (étroit int8 par canal, K entier, opt-in)

instrument : lecture de code (branche poste6-195 fusionnée dans poste2-p198) + `tests/test_gemm_etroit.py -k 195`
sous `outils/carte.sh`, deux fautes injectées puis annulées à la main dans `acvram/kernels/gemm_etroit.py`
commit : poste6-195 fusionné sur origin/main (be837ca1 + 196), branche poste2-p198
régime : sans carte pour la lecture, ACVRAM_NOM=poste2-p198-tests* pour les prises pytest (mesure, ≤ 1 s chacune)
scellé : aucun scellé propre à 198 — questions posées par chef, réponses ci-dessous
mesuré : voir chaque point
verdict : voir chaque point
durée : ≤ 5 min de lecture, 4 prises pytest (0,6-1,3 s chacune, hors file d'attente)

## (1) le défaut reste-t-il strictement inchangé quand ACVRAM_ETROIT_CANAL=0 ?

Par lecture (`acvram/kernels/__init__.py`, bloc pièce 195 dans `int8_matmul`) : le nouveau chemin est gardé par
`if geo is not None and gemm_etroit.disponible() and gemm_etroit.canal_eligible(t):` où `geo = geometrie_canal(...) if
gemm_etroit.canal_actif() else None`. `canal_actif()` lit `os.environ.get("ACVRAM_ETROIT_CANAL", "0") not in ("", "0")`
→ **False par défaut**, donc `geo = None`, la condition est fausse, aucun `return` n'est pris, le code tombe dans le
chemin C11 (vue g128) existant, INCHANGÉ. **TENU par construction** — la pièce est additive, jamais un `if` existant
n'est modifié.

**Lacune signalée** : `test_195_l_opt_in_est_ferme_au_defaut_et_le_regime_l_imprime` vérifie `canal_actif()` et
`etroites_texte()` mais n'appelle jamais `int8_matmul()` de bout en bout pour comparer sa sortie, au bit, contre le
même appel sur `origin/main`. La preuve du (1) est donc STATIQUE (lecture du code), pas un test qui pourrait rendre
faux une régression future dans ce garde. Recommandation à poste6 : un test `test_195_defaut_identique_a_main` qui
appelle `int8_matmul` sur un tenseur i8c réel sans la variable posée et vérifie `torch.equal` contre une référence figée.

## (2) ses tests peuvent-ils rendre FAUX ?

Deux fautes injectées à la main dans `gemm_etroit.py` (worktree poste2-p198), testées sous carte, puis annulées
(`git checkout`) :

* **Faute A** (`_etroit_canal_kernel`, ligne 352) : zéro-point `128.0` → `127.0` (biais systématique de 1 sur le
  déquant). **Résultat : les 5 tests passent quand même** (`/tmp/p198-faute1.log`) — la tolérance relative de
  `test_195_le_canal_k_entier_suit_la_reference` (`TOL=2⁻⁸` fois la borne L1 `|x|·|w|`) absorbe ce biais : l'erreur
  introduite (`s[n]·Σx[m,k]`) est petite devant la borne quand `x` a des signes mêlés. **Ce test ne détecte pas un
  off-by-one de zéro-point.**
* **Faute B** (même fonction) : `for k0 in range(0, K, BK)` → `for k0 in range(0, K // 2, BK)` (moitié des K sommée).
  **Résultat : 2/3 cas paramétrés cassent** (`m=8,n=5120,k=6144` et `m=16,n=1000,k=640`, écarts 31 182 et 14 629
  éléments hors tolérance, sur GPU réel `device='cuda:0'`), **mais le petit cas `m=2,n=100,k=200` reste vert**
  (`/tmp/p198-faute2.log`).

**Verdict (2) : PARTIEL.** Le test peut rendre faux pour une faute structurelle sur K assez grand, mais PAS pour un
biais de zéro-point même grossier (±1/256), et PAS pour la même faute K-tronqué sur la plus petite forme testée
(`n=100,k=200`). `test_195_le_canal_refuse_ce_qui_n_est_pas_par_canal` (éligibilité) n'a pas été refauté ici — sa
logique est une comparaison de shapes/valeurs directe, pas une tolérance, donc a priori plus sûre, non retestée par
manque de temps de carte.

## (3) la variable est-elle dans VARIABLES_LUES et le régime ?

Confirmé par lecture (`acvram/regime.py` : `Variable("ETROIT_CANAL", "0", ...)` ajoutée à `VARIABLES`) et par le test
vert `test_195_l_opt_in_est_ferme_au_defaut_et_le_regime_l_imprime` : `etroites_texte()` imprime
`serie+canal(table)` ou `serie+canal(BNxBKxWxS)` quand l'opt-in est posé, rien de plus quand il ne l'est pas. **TENU.**

## (4) scellé qualité : T1/T2 même prise, seuils avant ?

`scratchpad/poste6-p195-25-09/scelle-qualite.md`, écrit « AVANT la prise » (titre) : **T1** = séquence 0 seule
(b=1, GEMV int8 servi) comparée à sa propre ligne dans le lot sous A, **T2** = rejeu de A — les deux dans le MÊME
processus/prise que la mesure B (assertion sur `CHEMINS_INT8["etroit_canal"]` pour prouver que la config a pris).
Seuils numériques écrits avant mesure : `kl_AB_max ≤ 1,1e-3`, `argmax_AB ≥ 0,964`, `rejeu_A = rejeu_B = 0` exigé, avec
falsificateurs nommés (KL au-delà du seuil, argmax sous le seuil, non-déterminisme). **TENU structurellement** — je
n'ai pas rejoué la mesure elle-même (aucun `.json` de résultat trouvé sous `scratchpad/poste6-p195-25-09/`, la
mesure n'a apparemment pas encore été prise ou son résultat n'est pas committé) ; ceci porte sur le SCELLÉ, pas sur
un résultat.

## Synthèse pour poste6

Le garde par défaut est solide (1, TENU par construction). Les tests protègent contre une faute structurelle
grossière sur K assez grand (2, PARTIEL) mais pas contre un biais fin de zéro-point ni sur les petites formes —
resserrer `TOL` ou ajouter un test dédié à ±1 sur le zéro-point serait utile avant de lever l'opt-in au-delà du
banc. Le reste (variables/régime, scellé qualité) est en ordre.
