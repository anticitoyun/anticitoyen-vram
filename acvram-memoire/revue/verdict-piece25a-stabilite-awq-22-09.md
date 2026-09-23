# Pièce 25 (a) awq-stabilite-experts — ÉCHEC, 0 expert mesuré — 22/09 (Manon)

* instrument : `outils/awq-stabilite-experts.py /mnt/AI_GENERATOR/models_acvram/Qwen3-VL-30B-A3B-awq-dequant-bf16 --jetons 100,1000,10000 --json`, sous carte.sh, worktree manon-w-21-09
* commit : main à jour (Océane, pièce 25, instrument neuf du 22/09)
* régime : 3 collectes de stats successives (100/1000/10000 jetons), échantillonnage stratifié par classe de jetons routés (1-7, 8-31, 32-127, 128-511, ≥512)
* scellé : seuil de stabilité mesuré attendu classe 32-127 ; réfuté si stable dès 1-7 ou instable même ≥512
* mesuré : les 3 collectes réussissent (`10791/15807/17427 tenseurs`, 94/38/50 s, 17235 experts de référence au total) — mais **`0 experts sur 17235` échantillonnés dans les classes stratifiées**, `courbe: {}`, `seuil_stabilite: null`
* verdict : **ÉCHEC — INVALIDE (aucune classe mesurée)**, pas un résultat réfuté ni tenu. Aucun expert n'a été retenu par la stratification (bug probable dans le script neuf : critère de sélection trop strict, ou décalage entre les experts routés à 100 jetons vs 10000 jetons — les tenseurs de référence existent (17235) mais aucun n'a été classé/échantillonné). Hors mon domaine de corriger un instrument écrit ce matin, à nommer par Océane.
* durée : ~3 min de carte (94+38+50 s de collecte), verrou rendu propre

## Suite
(a) à refaire après correctif du script. Continue sur c9-m-uva.py puis (c) sur l'alias actuel avec la référence identite en attendant.
