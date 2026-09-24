# Scellé — pièce 145 : TTFT et énergie du préfill en service, défaut contre PROJ_MARLIN (poste6, 24/09, AVANT la mesure)

Ordre de chef : la décision « PROJ_MARLIN par défaut » manque du chiffre du préfill SERVI. poste1 : temps d'eval +69 % sur
gemma4 31B (142 : dépaquetage Triton de la 134 au préfill, tête 262 144 × 5 376) et +4,6 % sur Qwen3.8-27B (134).

## Instrument
`outils/gpu/mesure/ttft-service-p145.py` (suivi) contre `acvram serve --max-batch 1 --max-model-len 4608 --no-prefix-cache
--speculative none` ; par longueur L ∈ {512, 2 048, 4 096} : invites DISTINCTES de L ids (`invite(k, L)` du banc, mêmes ids
dans les deux bras), une requête à la fois, `max_tokens=1`, TTFT = premier fragment SSE portant du texte ; fenêtre ≥ 10 s et
≥ 5 requêtes sous `Energie`, ligne de base `repos(8 s)` après ; J par préfill net. Prise `scratchpad/poste6-p145-24-09/prise.sh
<gemma|qwen38> <1|2>` : ABBA par relance du serveur (prise 1 : A B B A A ; prise 2 : B B A A B → 5 A + 5 B par modèle),
`-lgc 2700` posé par prise, `set -euo pipefail`, HEAD asserté, `acvram.__file__` imprimé (import du worktree), preuve du bras
dans le journal du serveur (`dense=…+marlin(doubles=0,…)` en B, absent en A) et `/metrics` (`ACVRAM_PROJ_MARLIN=1`).
* A = défaut (`ACVRAM_PROJ_MARLIN=0`) ; B = `ACVRAM_PROJ_MARLIN=1 ACVRAM_PROJ_MARLIN_DOUBLES= ACVRAM_GEMV_MARLIN_V2=1
  ACVRAM_GEMV_MARLIN_TPB=1 ACVRAM_GEMV_MARLIN_S=0` (le bras B des 129/142 : disposition unique + v2 + dépaquetage 134).
* Modèles : `gemma-4-31B-it-nvfp4-vision` (celui de la 142) et `Qwen3.8-27B-nvfp4` (129/134).
* Mesures : TTFT médiane par (modèle, L, bras) sur 5 lots (médiane des médianes), ratio B/A, ΔTTFT ms ; J par préfill net et
  J par jeton de préfill ; jetons/s de préfill. Point NUL : fenêtre < 10 s, n < 5, bras non prouvé, horloge hors 2 700 ± 50.

## Prédictions (avant), par case — ratio TTFT B/A (J/préfill suit le même ratio à ± 5 %, même puissance)
| modèle | L | TTFT A prédit | B/A prédit | FAUX si |
|---|---|---|---|---|
| gemma4 31B | 512 | 120-250 ms | **1,4-1,8** | < 1,25 ou > 2,2 |
| gemma4 31B | 2 048 | 450-900 ms | **1,5-1,8** | < 1,3 ou > 2,2 |
| gemma4 31B | 4 096 | 1,0-2,0 s | **1,5-1,8** | < 1,3 ou > 2,2 |
| Qwen3.8-27B | 512 | 100-220 ms | **1,02-1,10** | > 1,20 ou < 0,97 |
| Qwen3.8-27B | 2 048 | 400-800 ms | **1,03-1,08** | > 1,15 ou < 0,97 |
| Qwen3.8-27B | 4 096 | 0,9-1,8 s | **1,03-1,08** | > 1,15 ou < 0,97 |
Pourquoi : à l'eval le préfill est la quasi-totalité du temps → +69 % (gemma) et +4,6 % (Qwen3.8) sont des majorants du ratio
en service, où le TTFT contient aussi le tour HTTP, l'échantillonnage et le décodage du premier jeton (≈ 15-30 ms fixes) ; le
dépaquetage est proportionnel aux poids, donc son surcoût relatif ne dépend guère de L (chunks de 256). Réponse à la question
de chef : **« +69 % tient-il en service ? » — prédit OUI à 2 048 et 4 096 (1,5-1,8), un peu moins à 512 (part fixe).**
* Ce qui me gênerait : gemma B/A ≤ 1,25 — le +69 % de l'eval viendrait d'ailleurs (logits complets de l'eval, tête 262 144
  dépaquetée à chaque fenêtre) et ne pèserait pas sur le service ; je le dirais, c'est la réponse utile à la décision.
* Ce qui invalide : A et B rendent le même TTFT au ms près sur les deux modèles → la variable n'a pas pris (preuve du journal).

## Durée
Par lot : chargement (gemma 31B nvfp4 ≈ 60-120 s, Qwen3.8 ≈ 15-90 s) + 3 × (≥ 10 s + 8 s repos + chauffe) ≈ 2-3 min ;
5 lots par prise ≈ 12-15 min ; 4 prises (gemma 1, gemma 2, qwen 1, qwen 2), chacune sous carte.sh ≤ 1 500 s, en file derrière
la chaîne 32B d'poste1. cpu-safe=off, relevés au début et à la fin.

## Addendum 10 h 3x — avant la prise qwen38 2 (3 prises faites : gemma 1-2, qwen38 1)
* Le plafond de puissance (400 W) bride le préfill dans les DEUX bras : horloge SM lue 2 380-2 440 MHz sous `-lgc 2700` (A ≈ 2 435,
  B ≈ 2 400, 394-398 W). La clause « point NUL si horloge hors 2 700 ± 50 » a été écrite pour un décodage sous le plafond ; elle
  se déclenche ici sur les 15 lots. Je ne la rouvre pas : les chiffres sont publiés comme **régime servi, bridé en puissance
  (plafond 400 W), horloge lue par lot**, pas comme une cellule « à 2 700 » — c'est le régime réel d'un préfill sur cette carte.
* À L = 512, les lots A atteignent N_MAX = 40 requêtes avant 10 s (9,2 s) → `fenetre_valide` faux, écartés à tort par l'agrégat.
  Pour la prise qwen38 2 : N_MAX = 60 ; l'agrégat accepte n ≥ 20 (les TTFT par requête ne changent pas). Les lots gemma A à 512
  (n = 39, 10,2 s) étaient valides.
* Équité de carte (chef) : la prise qwen38 2 ne repart qu'après « rendue » de poste4-* et chef-tests-139 (attendre-equite.sh).
