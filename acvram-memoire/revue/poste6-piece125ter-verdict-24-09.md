# Verdict — pièce 125 ter : (a) prix du W8A8 sur invites longues ; (b) bisection de l'excédent du préfill (poste6, 24/09)

instrument : `outils/gpu/mesure/kl-chemins-p125.py` (modes `invites-longues`, `prefill`, `paire`, `compare`), `scratchpad/poste6-p125-24-09/prise-ter.sh` → `prise-ter.txt`, `coder-long/paire-gemv-cublas-<alias>.json`, `coder/compare-{D,B1,B2,B3}-<alias>.json`, preuves `*-preuve.json` (`chemins_int8`, `chemins_moe`, env)
commit : 2f30f244 (branche poste6), asserté rc 65 ; venv de mesure, PYTHONPATH de l'arbre
régime : b=1, Coder i8c, six bras de préfill dans la même prise ; (a) invites 526-547 jetons (480 jetons de prose + question, réponse de HF de l'invite courte) ; (b) invites courtes, dumps HF du 02 h 03, forcé du 02 h 04 en repère ; cpu-safe=off (max_perf_pct 100 début et fin) ; compute-apps début = fin = llama-server 4627
scellé : `revue/poste6-piece125ter-scelle-24-09.md` (36bc18b6, avant) — (a) prédit KL_moy 0,01-0,04, alarme > 0,2 ; (b) B1/B2 isolent si invites 1 et 2 ≤ 1,25 × forcé, B3 au bit
mesuré : (a) KL(témoin W8A16 ‖ cublas) moy **0,019** (0,008 · 0,008 · 0,010 · 0,021 · 0,048), max 0,55, argmax égaux 136/141, L2 d'invite 0,011 (c0) → **0,055** (fin) ; `chemins_int8` cublas 240 + cublas_partage 720 contre gemv 965. (b) **B1, B2 identiques à D au bit** (KL, max, L2 égaux à 10⁻⁵) et B3 au bit (L2 ± 2e-4) ; `chemins_moe` = `marlin` 240/240 dans les QUATRE bras — `ACVRAM_MOE_MMA=0` et `ACVRAM_PREFILL_DEQUANT=1` sont **inertes** sur i8c (disposition unique : moe.py:920 `_PREFILL_GROUPED == "marlin" or unique` prime, moe.py:788 `mma` exige `not unique`)
verdict : (a) **TENU** — prix du W8A8 relatif au témoin W8A16 : ≈ 0,02 nat, ≤ 0,05 par invite, 5 % de L2 en fin de pile, argmax 96 % ; (b) **FAUX** — les bras experts n'ont rien coupé (preuve dans le processus), la cause de l'excédent des invites 1-2 n'est pas isolée ; la glue C15 est au bit (prédit)
durée : prévu ≤ 5 min ; tenu 90 s (`tenue=` au journal, six chargements de 14 s) ; une prise nulle avant (02:23, ids en chaîne, 11 s de carte)

## Lecture
* **(a) pour le README** (étiquette obligatoire : « relatif au témoin W8A16, pas à HF ») : sur des invites de 530 jetons le
  W8A8 des projections coûte 0,02 nat de KL moyenne (max 0,55 sur un pas), 4 argmax changés sur 141, 5,5 % de L2 sur le
  flux résiduel en fin de pile — pour × 1,29-1,34 de préfill (C15). Le chemin fusionné q/k/v (`cublas_partage`, A8 quantifiée
  une fois) porte 720 des 960 appels : c'est bien le W8A8 servi qui est mesuré.
* **(b) ce que la prise a réellement testé** : rien côté experts. Sur le Coder i8c (disposition unique Marlin), le préfill des
  experts est **Marlin groupé W4A16** (pas le MMA W4A4 : `mma` exige `not unique`) — le commentaire moe.py:775 « coupé par
  défaut » est vrai ici pour cette raison, et mon scellé qui prédisait « B1/B2 isolent le W4A4 » reposait sur une lecture
  fausse du dispatch : `MOE_MMA=0` et `PREFILL_DEQUANT=1` n'ont aucune voie sur `unique`. Les preuves `chemins_moe` l'ont
  rendu visible avant tout chiffre — sans elles, quatre bras identiques auraient passé pour « cause hors des trois ».
* Reste donc, pour l'excédent des invites 1-2 (préfill ×2,1 / ×3,35 le forcé, projections et experts SANS variable témoin) :
  le Marlin groupé de préfill contre le GEMV Marlin (mêmes poids, autre ordre de sommation, autre découpe par expert) et
  l'attention SDPA bf16 contre la paginée. Aucun des deux n'a de témoin par variable sur cet alias. Le profil L2 préfill‖forcé
  de la 125 (1 % à la couche 0 → 4-9 % en fin, croissance régulière) ressemble à un bruit d'ordre de sommation accumulé,
  pas à une couche fautive — c'est une lecture, pas une mesure.
* Ce qui peut encore trancher, hors de cette pièce : un bras « experts par le GEMV au préfill » n'existe pas ; un bras
  « attention paginée au préfill » non plus. Les créer est du code moteur (poste1), pas un réglage.

## Chiffres (a) — KL(gemv ‖ cublas) par invite longue
| invite | n_invite | KL moy | KL max | argmax égaux | L2 c0 | L2 fin |
|---|---|---|---|---|---|---|
| 0 | 539 | 0,021 | 0,17 | 32/32 | 0,011 | 0,056 |
| 1 | 526 | 0,008 | 0,15 | 31/32 | 0,012 | 0,054 |
| 2 | 527 | 0,008 | 0,12 | 20/20 | 0,011 | 0,055 |
| 3 | 547 | 0,010 | 0,22 | 24/25 | 0,012 | 0,055 |
| 4 | 530 | 0,048 | 0,55 | 29/32 | 0,012 | 0,056 |
