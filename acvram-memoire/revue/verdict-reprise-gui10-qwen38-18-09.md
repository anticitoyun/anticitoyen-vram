# Verdict — reprise : (1) `energie.j_par_jeton_10s` **0,3119 contre energie.py 0,3101 (+0,59 %)**, deux lecteurs identiques → **TENU** ; (2) Qwen3.8 b=12 aux défauts du jour **408,6 t/s · 0,978 J bridé, nu 434** contre la cellule 412 · 0,97 · 441 → **écarts < 3 % : un profil dense suffit**

instrument : `scratchpad/controle-gui10-18-09/{chaine.sh,controle.py,controle.json}` (serveur Coder 8094 v0.6.12, b=12, charge continue 12 fils, fenêtre energie.py 10,11 s ouverte à t = 20 s, deux `GET /metrics` à la fermeture) ; `scratchpad/qwen38-defaut-18-09/{cert,profil}.json` (`certifie-b12`, `profil-gdn b12`)
commit : dfc0bdb (poste3 = main 79ac659), 09:47-09:51, régime classé, défauts RPW=4 + XREG=down (rien de posé), graphes on
régime : (1) 168 requêtes sur 45 s, 0 échec, 13 476 jetons décodés dans la fenêtre de 10 s (1 333 t/s), 413 W deux cartes ; (2) Qwen3.8-27B-nvfp4-calibA b=12 bridé 399 W
scellé : (1) |GUI / energie.py − 1| ≤ 0,10, lecteurs identiques ; (2) écart bridé < 3 % ⇒ profil dense suffit
mesuré : (1) GUI **0,3119** J/jeton (deux lecteurs : 0,3119 / 0,3119) · energie.py 4 178,5 J / 13 476 = **0,3101** · écart **+0,0059** ; (2) bridé **29,37 ms, 408,6 t/s, 0,9776 J/jeton** (cellule 412 / 0,97 : **−0,8 % / +0,8 %**) · nu **27,63 ms, 434,3 t/s** (cellule 441 : −1,5 %)
verdict : (1) **TENU** — l'anneau rend le J/jeton des 10 dernières secondes à 0,6 % près et la valeur ne dépend plus du lecteur (les deux appels consécutifs sont égaux, ce que l'ancienne version ne pouvait pas faire) ; (2) **< 3 % sur les trois chiffres** — Qwen3.8 b=12 est insensible aux réglages GEMV experts (rpw=4, xreg=down) : son pas est porté par la GEMM dense (75 %), un profil dense suffit pour ce modèle ; la cellule publiée reste valable au défaut du jour

## Notes
- `/metrics.energie` ne contient que `j_par_jeton_10s` et `cartes` : les champs `fenetre_s` et `jetons_fenetre` annoncés dans la décision (poste7-metrics-energie-fenetre-18-09) ne sont pas dans app.py:836-841 — à chef de dire s'ils sont attendus.
- L'écart +0,59 % (contre −0,18 % pour l'ancienne fenêtre à bornes communes) est celui de deux fenêtres de 10 s décalées d'au plus un tic de 1 s sous charge stable : conforme à la prédiction < 5 %.
- Qwen3.8 : −0,8 % bridé / −1,5 % nu sont dans la dérive entre fenêtres (2 %) ; aucune régression des défauts du jour sur le modèle dense-dominé.
