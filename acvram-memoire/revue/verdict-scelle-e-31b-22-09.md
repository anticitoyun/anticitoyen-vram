# Scellé E qualité nvfp4 31B texte — A(max6)/B(4sur6)/T(historique) — MARGINAL, B pire que A — 22/09 (Manon)

* instrument : `scratchpad/scelle-e-31b-21-09/chaine.sh` (worktree manon-w-21-09, main à jour) — decode-pas.py TEXTE_SEUL, un passage HF offload + un passage acvram par alias ; 1er essai du soir ÉCHOUÉ (`TypeError depuis_graphe`, même défaut d'instrument que `mesure-c.py` — `decode-pas.py:76` `force()` corrigé, poussé) ; script final `chaine.sh` resté bloqué après B sur son calcul de verdict (tuée proprement, verrou déjà libre, aucun coût carte), verdict calculé à la main à partir des deux json témoins déjà écrits
* commit : d7e57da2
* régime : `sans_exil`, invite texte ≈ 217 jetons (corpus Manon), source HF `gemma-4-31B-it-bf16`
* scellé (Maîtresse, Q1) : KL max B ≤ 1,2 nat (tenu) ; réfuté si > 1,5 ; témoin historique 20/09 (ancien alias avant Four Over Six) = 2,12
* mesuré : **A (max6) `kl_max=0,8673`** (verdict interne decode-pas.py : « NON RÉSOLU », entre ses propres seuils 0,5/1,0 — une 2e invite trancherait A définitivement, non jouée ici, hors du scellé E) ; **B (4sur6) `kl_max=1,3935`** (verdict interne : « 31B nvfp4 en général ») ; `part_amax4_B = 0,324`
* verdict : **MARGINAL** pour B (1,3935 entre 1,2 et 1,5) — **B est PIRE que A sur ce prompt** (1,3935 contre 0,8673), contraire à l'attente que Four Over Six (choix adaptatif amax/4 vs amax/6 par bloc) améliore la qualité par rapport au max6 fixe. Les deux restent nettement sous le témoin historique cassé (2,12). Résultat à trancher par Sage/Maîtresse : Four Over Six n'aide pas ici, ou un seul prompt/8 pas est insuffisant pour juger (cf. P3(3) du même 30B-VL, résidu de 2487 experts sans stats — hypothèse similaire de couverture de calibration à vérifier pour le 31B).
* durée : essai 1 (échoué) 8 min ; essai 2 (valide, A+B) 17 min 30 s ; verrou rendu propre les deux fois

## Suite
Ne pas conclure sur Four Over Six pour le 31B à partir d'un seul prompt texte — Sage/Maîtresse tranche si un 2e prompt ou la PPL relative (pièce laissée à part par ce scellé) est nécessaire avant décision. Pointeur transmis.
