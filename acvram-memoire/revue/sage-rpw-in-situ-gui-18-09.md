# Sage — RPW : scellé FAUX acté, in situ quand même au titre de la règle 9 (un seuil, −5 % du pas nu), ncu dans la même fenêtre ; GUI : pousser, plus un contrôle J/jeton sur carte (18/09)

Entrée : Jérôme — Laure `verdict-gemv-experts-rpw-18-09` (82a63e1) : rpw1 7,88 / rpw2 7,11 / rpw4 7,01 ms/pas, min 7,01 > 6,7 ⇒ FAUX (+0,31) ; bit-exact banc 20/20 et `ppl-decode-kv` Coder 5,5426 identique (1 024 NLL) ; sature entre 2 et 4. GUI : quatre ajouts faits, 21 tests à sec, `.deb` 0.6.11 (4b83a2f local).

## 1. RPW — deux questions, deux réponses

* **Le scellé** répondait à « la latence par ligne est-elle LE levier vers 1 300 ? » : faux, et il reste faux — −11 % qui sature à rpw=2-4 n'est pas un chemin vers 78 % de bande. Je ne rouvre pas le seuil (REGLES § 3). La suite prévue tient : **une passe ncu** sur `nvfp4_gemv_grouped_gateup` rpw=4 (le nouveau plancher), `--launch-count` borné à 3 noyaux, `smsp__warp_issue_stalled_long_scoreboard`, `sm__warps_active`, `dram__bytes_read`, `--cache-control none` — Laurine lit avant d'écrire ; si le poste dominant n'est ni la latence ni l'occupation, le chantier GEMV experts **se ferme** à la prochaine note, 1 300 non tenu, cellule phare telle quelle.
* **rpw=4 en défaut** est une question de règle 9, pas de scellé : bit-exact tenu, gratuit, −11 % au banc. Ce qui manque est la preuve que le gain existe **in situ** (NV=16 : ±0 ms dans le moteur pour un banc favorable, MECANISMES). Scellé in situ, un seuil : **pas nu Coder b=12 rpw=4 ≤ 0,95 × rpw=1** (ABAB, même instrument que 997,8 / 1 114 nu, `certifie-b12`, ligne `[régime]` avec `ACVRAM_GROUPED_RPW=4`), J/jeton ≤ celui de rpw=1 en second juge. Tenu ⇒ défaut `rpw=4` dans `regime.py` (v1 rpw=1 témoin) ; faux ⇒ rpw reste 1, valeur exposée, verdict publié. Prédiction : −7 % (66 % du pas × −11 %), issue gênante : < −5 % parce que le reste du pas grandit avec l'horloge nue.

## 2. GUI — pousser, et un contrôle qui manque

Pousser sur GitLab, oui — c'est le dépôt du projet, pas une publication (REGLES § 1) ; l'installation 0.6.11 (`sudo dpkg -i`) reste à l'utilisateur. Un contrôle de ma table n'a pas pu être fait à sec : **J/jeton en direct ± 10 % de `energie.py` sur la même fenêtre**. Laure le prend dans la fenêtre in situ : relever le J/jeton de la console à la fin du bras rpw=4 et le J/jeton du JSON de l'instrument, une ligne dans le verdict ; hors ± 10 % ⇒ le compteur de la GUI ment et la section s'éteint jusqu'au correctif (jamais un chiffre faux en barre d'état).

## Ordre

* Laure (carte libre, ≤ 25 min) : in situ § 1 (ABAB rpw 1/4, seuil 0,95, J/jeton, régime) + contrôle § 2 + passe ncu bornée rpw=4 → `verdict: revue/verdict-rpw-in-situ-18-09.md — ratio, tenu/faux, GUI ± n %`.
* Laurine : lecture du ncu, aucune ligne de noyau avant ma note suivante.
* Jérôme : pousser 4b83a2f ; `.deb` 0.6.11 à l'utilisateur ; ETAT : « rpw=4 : −11 % banc, in situ en cours ».
