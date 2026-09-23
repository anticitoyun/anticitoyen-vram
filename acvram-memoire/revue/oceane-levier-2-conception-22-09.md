# Levier 2 — rapatriement épinglé, non bloquant, à double tampon par parité (conception à sec, 22/09, Océane)

* instrument : `outils/gpu/mesure/frontiere-pas.py` (juge principal) ; `certifie-b12` ABBA (secondaire)
* commit de référence (témoin) : b0c25910 sous `ACVRAM_SAMPLER_GRAPHE=1` (levier 1 tenu : `verdict-frontiere-2-graphe-22-09`)
* régime : b=12 Coder nvfp4, ctx 2048, invite 256, éco 2700, graphes + pipeline + sampler=graphe
* écart à vLLM : −2,9 % (1 550 contre 1 596 t/s)

## 1. Ce que les JSON de Manon disent (A0/A1 lent, B0/B1 graphe ; médianes µs, 300 pas pleins)
| | A (lent) | B (graphe) |
|---|---|---|
| pas_gpu | 6 905 / 6 922 | 6 904 / 6 909 |
| graphe (tête incluse) | 6 665 / 6 687 | 6 725 / 6 726 |
| echantillon (carte) | 41,2 / 42,2 | 3,7 / 3,7 |
| **trou_gpu** | 188,9 / 188,9 | **172,3 / 173,6** |
| suite_prep (hôte : _grow + _build_batch_device + preparer) | 165 | **157** |
| lancement (rejouer_suivant, hôte) | 14 | **16** |
| echant_hote | 76 | 14 |
| attente_evt (`event.synchronize`) | 6,3 | **5,8** |
| **consommer** (`.tolist()` + `_emit`) | **6 610 / 6 640** | **6 694 / 6 696** |
| reste_step + hors_step | 15 | **13** |
Lecture : `attente_evt` = 6 µs — le pas n est FINI quand on le consomme ; pourtant `consommer` dure 6,7 ms : **le `.tolist()` est une copie D2H enfilée sur le flux courant, DERRIÈRE le rejeu n+1 qu on vient de lancer**. L hôte reste bloqué toute la durée du graphe n+1, puis seulement prépare n+2 : **trou_gpu 173 = suite_prep 157 + lancement 16** (+ reste 13 − recouvrement), à 5 µs près sur les deux paires. Le pipeline ne recouvre aujourd hui que le LANCEMENT de n+1, pas la préparation de n+2. La part hôte qui remplit le trou : suite_prep 91 %, lancement 9 % ; `consommer` n y est pour rien (il attend, il ne travaille pas).

## 2. Mécanisme
Aucune arithmétique ne change : c est un déplacement de données.
* `_sample_only(depuis_graphe=True)` : après le clone `[2, n]` (levier 1), enfiler sur le même flux `epingle[p].copy_(paquet, non_blocking=True)` vers un tampon **hôte épinglé** `[2, b_max·ql]` int64, `p` = parité du pas ; PUIS enregistrer l événement (il couvre le clone ET la copie). Le pendiente porte `(tokens_dev, epingle[p], event)`.
* `_consommer` : `event.synchronize()` (déjà là) puis lecture du tampon épinglé `[:, :n].tolist()` — **aucune copie sur le flux**, aucune attente derrière n+1 ; `_emit` python inchangé.
* **Double tampon par parité** : la copie de n+1 est enfilée AVANT que l hôte lise n (même `step()` : lancer n+1 puis consommer n) ; avec un seul tampon, ce serait une course gagnée par 6,7 ms de marge — refusé. Deux tampons : n+1 écrit `epingle[(n+1) % 2]` pendant que l hôte lit `epingle[n % 2]`. L invariant qui porte la parité : **au plus un pas en vol** (`_pipeline_pendiente` est un seul objet). Rendu impossible à sauter : chaque tampon porte `lu = True/False` ; en cibler un non lu lève `RuntimeError("tampon épinglé de parité p réutilisé avant lecture")` — pas un avertissement.
* Chemins sans changement : lot inéligible (température, pénalité, finie) → `sample` sur les logits, tenseurs neufs, `.tolist()` comme aujourd hui ; pipeline=0, eager, spéculatif : rien. Opt-in `ACVRAM_RAPATRIEMENT_EPINGLE=1` (déclaré dans `regime.VARIABLES`), ligne `rapatriement=epingle|flux` ; défaut après verdict, comme le levier 1.
* Ce qui reste hôte après : suite_prep 157 µs et lancement 16 µs se font **pendant** le graphe n+1 (6,7 ms de marge : 40 ×) ; `_emit` avec tokenizer (`_decode_delta` × 12, absent de certifie) aussi.

## 3. Invariants au bit et tests (même commit)
1. **ids et logprobs au bit** : la copie transporte les mêmes int64 que le clone ; test à sec : sur 100 pas simulés (faux graphes, tampon `sortie` réécrit à chaque pas), la sortie de `_consommer` == `_sample_lent` sur les mêmes logits, ids ET `cumulative_logprob` ; test carte (Manon) : ids/logprobs au bit b=1/b=12 contre b0c25910 sous les deux opt-in.
2. **Parité** : test à sec où le tampon de parité p est réécrit (copie n+2 simulée) après la lecture de n → lecture intacte ; et où on cible p AVANT la lecture → `RuntimeError` (doit rendre faux si la garde disparaît).
3. **Aucune copie sur le flux à la consommation** : `monkeypatch` de `Tensor.tolist`/`.cpu` : aucun appel sur un tenseur device pendant `_consommer` du chemin épinglé ; exactement un sur le tampon hôte.
4. **Événement après la copie** : l ordre `clone → copie épinglée → event.record()` est testé par un faux flux qui journalise les enfilages (l événement doit être le dernier) — sinon `synchronize` rendrait avant la copie.
5. Ligne de régime : `rapatriement=epingle` seulement si pipeline ET graphes ET sampler=graphe ET opt-in ; défaut `flux`.

## 4. Plafond calculé, prédiction, réfutation (écrits avant)
Plafond = le trou lui-même : 173 µs − latence de lancement incompressible (~10-15 µs : le rejeu n+2 est déjà en file quand n+1 finit, il ne reste que le passage graphe → graphe) ≈ **160 µs sur 6 905 = 2,3 %** ; rien d autre ne bouge (graphe 6 725, échantillon 4).
* **Prédit** (frontiere-pas, opt-in) : `trou_gpu` 173 → **10-30 µs** ; `consommer` 6 694 → **≤ 60 µs** (lecture épinglée + emit sans tokenizer) ; `attente_evt` 6 → **≈ 6 500 µs** (c est là que l hôte attend désormais, pendant que n+1 tourne — signature attendue, pas un défaut) ; `pas_gpu` 6 905 → **6 740-6 760** (−2,1 à −2,4 %).
* **t/s** (ABBA certifie, ≥ 8 paires) : **+2,0 à +2,4 %** (1 550 → 1 581-1 587), écart à vLLM −0,5 à −0,9 % ; J/jeton −1,5 à −2 % (même énergie de calcul, moins de temps oisif à 400 W).
* **Réfuté** si `trou_gpu` reste > 100 µs sous opt-in (le mécanisme n a pas supprimé l attente : la copie n est pas là où je crois, à relire `_pipeline_pendiente`) ; ou gain t/s < +1,0 % ; ou ids/logprobs ≠ au bit (défaut). **Alarme** : gain > 2,6 % ou `graphe` qui change de plus de 10 µs — autre chose bouge (horloge, ordre), on le dit avant de publier.
* Après ce levier, le pas est borné par le graphe (6 725 µs) : la suite est dans le graphe (H2 glue 0,59 ms, MoE), plus dans la frontière.

## 5. Protocole
1. Équivalence carte (§ 3.1) avant toute cellule.
2. `frontiere-pas.py` A B B A (A = sampler=graphe seul, B = + rapatriement épinglé), 300 pas pleins, ≤ 1 min la passe.
3. `certifie-b12` ABBA ≥ 8 paires × 20 s, horloge SM médiane par fenêtre (rejet > 3 %), J/jeton contre le témoin de la même fenêtre.

## 8. Amendement après `verdict-levier2-2-frontiere-22-09` (Manon) — RÉFUTÉ sur mon critère, cause lue, corrigée
Mesuré (A flux / B épinglé, médianes) : consommer 6 702 → **15 µs** et attente_evt 5 → 5 (l hôte est libéré) MAIS **suite_prep 151 → 6 860 µs** et trou_gpu 165 → 182 : l attente a seulement changé de place — elle est maintenant DANS `_pipeline_suite`, donc dans la copie épinglée elle-même (`_apres_echantillon`, appelée depuis la suite ; lancement 15 et echant_hote 13 inchangés). Cause (pipeline.py:83-88 de 7f967671) : le tampon épinglé était alloué `[2, max(n, 16)]` et la copie faite dans la vue `[:, :n]` — **non contiguë** pour n = 12 ; une copie carte → hôte non contiguë passe par un tampon paginable intermédiaire, et une copie vers de la mémoire paginable est SYNCHRONE : l hôte attendait derrière le rejeu n+1 exactement comme avec `.tolist()`. Ma prédiction « attente_evt ≈ 6 500 » supposait la copie asynchrone ; elle ne l était pas, et rien ne le contrôlait.
Correctif (`oceane-levier-2-contigu`) : un tampon par (parité, n), **exactement [2, n]**, contigu ; garde qui LÈVE si source/cible ne sont pas contiguës de même forme ou si la cible n est pas épinglée (une copie synchrone ne peut plus passer en silence) ; test : forme (2, n) contiguë après recomposition n = 12 → 5, et une cible `[:, :12]` d un [2, 16] refusée. Prédiction inchangée pour le rejeu de frontiere-pas (A = flux, B = épinglé corrigé) : **suite_prep 6 860 → ≤ 200 µs, attente_evt 5 → ≈ 6 500 (la signature), trou_gpu 182 → 10-30, pas_gpu 6 925 → 6 740-6 760** ; réfuté si trou_gpu > 100 µs (autre attente cachée : à chercher alors par `nsys --trace=cuda,osrt` sur 50 pas, les appels cudaMemcpy synchrones y sont visibles). L ABBA de Manon sur 7f967671 ne juge pas ce levier : B y est A avec 15 µs de consommer en moins et 6 860 d attente ailleurs — prédit B/A = 1,000 ± 0,005, sans signification.
Leçon : « non_blocking=True » n est une promesse que pour une copie contiguë vers de l épinglé ; l invariant s écrit dans le code (garde), pas dans la doc.
