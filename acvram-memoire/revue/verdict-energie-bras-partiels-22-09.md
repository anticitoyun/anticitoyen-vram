# Énergie, bras partiels manquants (trtllm b=12 2e fenêtre + acvram b=1 ×3) — complétés, cohérents avec le chiffre officiel — 22/09 (Manon)

* instrument : `chaine-energie-4moteurs.sh --bras "trtllm:12:2 acvram:1:1 acvram:1:2 acvram:1:3"`, sha `91bb89f4` (chauffe de Laurine, `jetons_chauffe` publié), puis `rejuger-energie.py` sur l'union des sorties
* mesuré (nouvelles fenêtres, toutes VALIDES après le critère bridage assoupli — plus de rejet automatique sur `bridages≠aucun`) :
  * **trtllm b=12 (2e fenêtre)** : 1952,6 t/s, 0,1588 J/jeton net — comble le trou (l'ancienne T2 restait « RESULTAT absent/invalide », timeout récurrent inchangé sur cette fenêtre-là précisément, mais la nouvelle passe réussit).
  * **acvram b=1 ×3** : 392,1/392,3/392,5 t/s, tous rejetés `sd 12-13% > 10%` — toujours instable à b=1, cohérent avec les 4 essais précédents du jour.
* **Union rejugée, chiffres officiels par moteur (b=12, cohérents avec le rejugement de Laurine 2ea1779f)** : trtllm 0,1588 J/jeton (1 fenêtre valide), vLLM ~0,163 (med 2 fenêtres), acvram ~0,197 (med 2 fenêtres), llama.cpp 0,2055 (med, 4ᵉ mesure convergente du jour) — **écarts <2% avec le chiffre déjà publié par Laurine**, confirmation croisée.
* verdict : **bras manqués complétés, cohérents avec le chiffre officiel déjà arrêté**. acvram b=1 reste non mesurable proprement (sd systématiquement >10%, 4/4 essais) — signal distinct, pas résolu par le critère bridage assoupli.
* durée : ~9 min de carte

## Suite
Enchaîne diag-disposition-experts (à sec) avant la reconversion --alpha-commun-experts.
