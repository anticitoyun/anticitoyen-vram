# 11e — échelles de bloc nvfp4 (max6 / balayage-w / 4sur6 / balayage), PPL appariée sur Coder-7B : verdict (poste1, mesureuse, 01/10)

* instrument : `outils/gpu/mesure/ppl-balayage-11e.sh` (convertir ×4, evaluer, analyser) + `ppl-appariee-bootstrap.py` (20 000 tirages) ; analyse rejouée avec la lecture corrigée (e492c72a6, voir § Instrument) ; données `scratchpad/poste1-11e-01-10/`
* commit : 0f7a54062 (arbre figé `travail/poste1-11e-fige`, COMMIT_ATTENDU asserté, PYTHONPATH sur l'arbre mesuré vérifié avant chaque prise)
* régime : Qwen2.5-Coder-7B-Instruct, AWQ calibré (32 × 512 jetons Gutenberg #1342), grille AWQ en max6 pour les quatre bras (option A), seule la règle finale change ; wiki-gptq sha256 e52922746ad0, fenêtres disjointes de 2 048, min_context 0 ; 5090 seule (8081 sur la 3080 Ti)
* scellé : `poste6-11e-scelle-ppl-30-09.md` (S1-S7, issues a-f) + addendum 1 (mesureur, résolution ≤ 0,25 %, témoin) + addendum 2 (S8/S9, B1/B2, option A) — tous écrits avant la prise
* mesuré : témoin max6 = max6-t **au bit** (deux passes) ; act_scale **égaux au bit** entre les 4 bras (196 tenseurs, c27eab2c9f6de0c6). À 32 fenêtres : 3 comparaisons sur 5 au-dessus de la demi-largeur 0,25 % → NON RÉSOLU → rejeu fixé d'avance à 128 fenêtres (262 016 positions), mêmes convertis :
  | comparaison (128 fenêtres) | Δ PPL | IC95 | z | lecture du scellé |
  |---|---|---|---|---|
  | S8 balayage-w − max6 | **−0,171 %** | [−0,322 ; −0,021] | −2,21 | conclusif, mais **pas « prometteur »** (seuil Δ ≤ −0,5 %) ; prédit −1,5 à −0,2 : juste au-dessus de la bande |
  | S2 4sur6 − max6 | **−0,157 %** | [−0,275 ; −0,041] | −2,65 | conclusif ; prédit −1,0 à +2,0 : tenu |
  | S3 balayage − max6 | −0,094 % | [−0,238 ; +0,051] | −1,27 | non conclusif → défaut inchangé |
  | S9 balayage-w − balayage | −0,077 % | [−0,164 ; +0,009] | −1,75 | non FAUX (balayage pas meilleur) ; non conclusif |
  | S4 balayage − 4sur6 | +0,064 % | [−0,058 ; +0,184] | +1,03 | balayage ne bat pas 4sur6 → « Q1 suffit » |
  S1 PPL max6 9,742 (5-12 : tenu). S5 part_balayes 0,598 (balayage) / 0,601 (balayage-w) contre 0,60-0,75 prédit : 0,002 sous la bande pour balayage, loin du FAUX de code ; 0 sous-normale ; balayage-w : 1 tenseur replié (tête, sans statistiques, en balayage MSE). S6 max6 neuf ≠ alias du 05/09 (converti SANS calibration, code antérieur) : dit, n'invalide rien. S7 conversions 1,6-6,2 min contre 10-25 prédit : faux par le bas (plus rapides), sans effet
* verdict : **défaut max6 inchangé (règle du scellé). 4sur6 et balayage-w gagnent chacun ≈ 0,16-0,17 % de PPL sur max6 (conclusif à 128 fenêtres), sous le seuil de 0,5 % ; balayage (MSE pur) ne se distingue pas de max6.** La pondération (balayage-w contre balayage) va dans le sens de H-Scale (−0,077 %) sans être conclusive. Rien ne justifie de payer balayage-w (×10 en quantification finale) quand 4sur6 donne le même gain pour ×2 ; si un gain de 0,16 % vaut un changement de défaut, c'est 4sur6 qui se présente, sur pièce séparée (alias, KL b=1 ≤ 0,74 sous gabarit), décision du chef
* durée : prévue ≤ 30 min par prise ; tenues : conversions 370 + 120 + 99 + 106 s, evals 5 × ~15 s (65 k) puis 5 × ~40 s (262 k) ; ≈ 15 min de carte en tout

## Bilan des hypothèses scellées d'poste6 (bd8, `poste6-bd8-arxiv-01-10.md` l.73-81 ; scellées pour le 14B sur le corpus privé, jugées ici sur le 7B/wiki-gptq, 128 fenêtres, PPL ponctuelles)
* H1 PPL(balayage-w)/PPL(max6) = 9,7251 / 9,7417 = **0,99830** ; prédit 0,985-0,998 → **FAUX de justesse, par le haut** (0,0003 au-dessus de la bande ; le falsificateur explicite « > 1,000 » n'est pas atteint : le proxy act_scale² vaut quelque chose, mais moins que prédit).
* H2 PPL(balayage)/PPL(max6) = 9,7326 / 9,7417 = **0,99907** ; prédit 0,990-1,005, FAUX si < 0,985 → **tenu** (balayage MSE pas mieux que max6, comme la figure 1 de H-Scale l'annonçait).
* H3 PPL(balayage-w) ≤ PPL(balayage) : 9,7251 ≤ 9,7326 (écart −0,0075) → **tenu** (FAUX si balayage meilleur de > 0,003).
* H4 PPL(4sur6)/PPL(max6) = 9,7264 / 9,7417 = **0,99843** ; prédit 0,992-1,000, FAUX si > 1,002 → **tenu**.
* Décision du chef (01/10) : max6 reste le défaut ; 4sur6 n'est pas ouvert (0,16 % < seuil 0,5 %).

## Instrument : un défaut d'analyse, pas de mesure
La première `analyser` (09:02) a rendu rc 6, que le script imprime comme « témoin différent ». En réalité, `acvram eval --json > fichier` écrit
son en-tête (variables ignorées, cadrage, budget, lignes `[acvram]` et `[régime]`) AVANT le JSON, et `json.load` sur le fichier entier
cassait. Le contrôle du témoin ne s'était exécuté que sur du JSON propre. Les données sont intactes : le témoin relu à la main est au bit,
cadrage et corpus identiques sur les 5 évaluations. Correctif sur la branche poste1-11e-analyse (e492c72a6) : la lecture commence à la
première ligne en `[`/`{` qui se décode, avec un test sur la forme réelle (rouge sur l'ancienne lecture). Le critère n'a pas changé.
Deux autres pièges, trouvés à sec AVANT la prise, sont consignés : venv éditable (poste1-11e-pythonpath) et B1/B2 (addendum 2).

## Suite proposée (au chef)
* H5 (importance diag(XᵀX) réelle, bd8 d'poste6) : à 0,17 % de gain total pour balayage-w et 0,08 % pour la pondération, le reste
  à gagner par une meilleure importance est probablement sous la résolution de 128 fenêtres (± 0,15 %). À ne pas mesurer sur ce couple
  modèle/corpus ; si l'on y tient, sur le 14B et le corpus privé de bd8.
* 4sur6 en défaut : pièce séparée, alias Coder-30B (le servi), KL b=1 sous gabarit, débit inchangé attendu (même format, mêmes noyaux).
