# Sage — Pause du 18/09 soir : P1 adopté (à basculer en défaut), point de reprise en cinq lignes

Entrée : utilisateur, extinction. Rien de Sage sur la carte.

## État au moment de la pause

* **P1 Marlin ADOPTÉ** (oui utilisateur) : bascule en défaut PAS ENCORE FAITE — `sage-p1-situ-verdict-18-09` § 6 (Laurine regime.py ; Laure contrôle au défaut sans variable, 20 min ; Jérôme comparatif contre la plus haute passe llama.cpp : b=12 +16 % t/s, J 0,303 vs 0,278 derrière 9 %, prefill parité +1,7 %, b=1 +2 %).
* **P2** : ouvert sur Coder seul (i8c 0,9947, `verdict-p2-hors-moteur-18-09`), fermé sur GLM (MLA, `verdict-p2-glm-hors-moteur-18-09`) ; chantier Laurine après la bascule : chemin `_int_mm`, équivalence décodage sur le converti i8c, cible ≈ 20 000 j/s.
* **Poste b=1** : 350,9 (0,959 × 366) ; profil (b) vs v1 sous graphes (Laure 20 min) avant tout geste.
* **Bead** : écart graphes/eager jusqu'à +0,206 sur une fenêtre (tr2-p256, `verdict-juge-stat-b-18-09`) — Océane après la bascule, deux exemplaires puis godet b=1.
* **Décisions utilisateur en suspens** : téléchargement source Devstral (Manon prête : mapping, yarn, FP8, llama4 scaling porté et validé à sec) ; Mistral-Small-4-119B (3-5 j après P1/P2, ≤ 10 t/s, `sage-devstral-mistral4-18-09`).

## Reprise

Lire ETAT, puis `sage-p1-situ-verdict-18-09` § 5-6 et `sage-devstral-mistral4-18-09`. Premier geste : la bascule (§ 6, 1-2-3) — rien d'autre avant.

## Ordre

* Jérôme : ETAT ≤ 40 lignes avec ces cinq lignes ; pause.
