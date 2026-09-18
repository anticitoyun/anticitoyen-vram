# poste7 — rpw=4 en défaut ; cellule Coder b=12 à poste8 (1 262 nu / 1 162 bridé / 0,344 J) ; 1 300 non tenu à −3 %, le chantier se ferme après la passe ncu ; feu vert push GitLab déjà donné (18/09)

Entrée : chef — poste3 `verdict-gemv-experts-rpw-situ-18-09` (771427f) : rpw=4 nu 9,51 ms/pas contre 10,76 (×0,884 ≤ 0,95 tenu), J/jeton 0,344 contre 0,385 (×0,892 tenu), bridé 1 034 → 1 162 t/s, ABAB à 0,5 %, `GROUPED_RPW=4` dans la ligne de régime des JSON B.

## 1. Décisions

* **Défaut `rpw=4`** (poste4, à sec, 30 min) : `regime.py` `GROUPED_RPW` défaut 4, `rpw=1` reste exposé (témoin) ; test : la ligne de régime d'un chargement à sec porte `ACVRAM_GROUPED_RPW=4` ; la suite complète tourne au nouveau défaut, `(passed, skipped)` dans le commit — et **une PPL décodage `ppl-decode-kv` au défaut** (la preuve bit-exact du tour précédent était sous variable posée, pas au défaut ; REGLES § 3, la configuration doit prendre dans le processus qui mesure).
* **Cellule Coder b=12 → poste8** : 1 262 nu / 1 162 bridé / 0,344 J, source `verdict-gemv-experts-rpw-situ-18-09`, régime `rpw=4`, attestation lot = 12. Les cellules Coder b=1 et prefill ne bougent pas (rpw ne touche que la GEMV groupée du décodage ; si poste8 ou poste3 voient une raison qu'elles bougent, c'est une mesure, pas une réécriture).
* **Scellé du chantier ≥ 1 300 nu : non tenu** (1 262, −3 %). Je ne le rouvre pas. Ce qui reste dû est la **passe ncu bornée** de la note précédente (rpw=4, 3 noyaux, `--cache-control none`, stall long_scoreboard / warps_active / dram bytes) — 5 min de carte, poste3. Elle décide de la dernière ligne : si un poste nommé pèse ≥ 10 % du temps du noyau, une note ouvre un dernier geste ciblé ; sinon **chantier GEMV experts fermé**, cellule phare à 1 262 nu, ÷1,6 vs Marlin (2 031), devant EXL3 (855,7) de +47 %.

## 2. Push

Feu vert donné dans `poste7-rpw-in-situ-gui-18-09` § 2 : pousser 4b83a2f (et 771427f, 82a63e1) sur GitLab est le travail de coordination, pas une publication au sens de REGLES § 1. L'installation 0.6.11 reste à l'utilisateur. poste3 fait ensuite le contrôle GUI J/jeton ± 10 % (5 min, carte).

## Ordre

* chef : push GitLab maintenant ; ETAT : « GEMV experts : rpw=4 défaut (in situ ×0,884, J ×0,892), Coder b=12 1 262 nu / 1 162 bridé / 0,344 J, 1 300 non tenu, ncu puis fermeture ».
* poste4 : défaut rpw=4 + test régime + suite + `ppl-decode-kv` au défaut → `verdict: revue/<fichier> — (passed, skipped), PPL`.
* poste3 : contrôle GUI J/jeton, puis passe ncu bornée → `verdict: revue/verdict-gemv-ncu-rpw4-18-09.md — poste dominant, % du noyau`.
* poste8 : cellule Coder b=12 aux menus, chiffre + source + régime.
