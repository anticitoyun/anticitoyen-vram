# P3 (4) duel Qwen3-VL-30B reconverti (correctif experts groupés, carte visible) — 21/09 (poste2)

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-4.sh` (worktree poste2-w-21-09)
* commit : 02a68005 (moteur ; convertisseur de l'alias = même commit, `jq '.convertisseur'`)
* régime : `ctx_tenu=non-chauffe deepstack=3 eco=2700(2692) graphes=on(hybrides≤4) kv=int8 masque_images=causal mrope=[24,20,20](interleaved) vision=bf16(eager,transformers=5.17.0)`, `-lgc 2700` posé/relâché proprement
* scellé : seuils H1 (poste1, ordre du jour) TTFT ≤ 0,3 s, J ≤ 60 — annoncé d'avance : « le scellé 0,094 s peut rester réfuté par la tour eager transformers, à dire » ; seuil historique § P3(4) du 20/09 (TTFT ≤ 0,094 s, J ≤ 40,0) gardé en référence
* mesuré : 20/20 images décodées — TTFT médian **0,1138 s** (témoin 0,0938 s), J net médian **41,1** (témoin 40,2), top1 identique au témoin sur **10/20** (5/20 le 21/09 matin avant reconversion : 17/20)
* verdict : **TENU sur les seuils H1** (0,1138 ≤ 0,3 ; 41,1 ≤ 60 — TTFT ×45→×1,21, J ×4,4→×1,02, amélioration massive vs l'alias int4_awq cassé) ; **réfuté sur le seuil historique 20/09** (TTFT +21 %, J +2,75 % — attendu, annoncé d'avance, tour eager) ; **ALARME qualité non couverte par ce scellé** : top1 chute à 10/20 (50 %) contre 17/20 avant reconversion — à mettre en regard de `experts_sans_stats=18432/18432` (100 % des tenseurs d'experts en repli échelle identité, AUCUNE statistique AWQ collectée à la calibration : `cli.py:654` « statistiques relevees pour 0 tenseurs », toutes couches en échec `setStorage`) — la latence/énergie sont bonnes, la justesse ne l'est pas, cohérent avec une conversion sans AWQ.
* durée : 114 s (18:27:17–18:29:11), prévu ≤ 30 min ; `nvidia-smi` propre après (seul PID 4453 étranger permanent)

## Suite
Cause de l'échec de calibration (`setStorage: ... storage size of 8 are out of bounds for storage of size 0`) hors mon domaine (collecte de stats, `acvram/quant/collect.py`) — à nommer par poste1 avant de juger la qualité P3 (3) qui suit. Enchaîne sur P3 (3).
