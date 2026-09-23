# Sage — transfert des 48 alias restants (643 Gio) et 100 chemins en dur : rang et ordre (20/09, 10 h 40, horloge machine)

Question de Jérôme (10 h 38) : rang du transfert USB → nvme3 (60-80 min, écrit sur le disque des modèles mesurés, sha256 sur un cœur) ; Océane sur les chemins en dur maintenant ?

État lu 10 h 39 : 16/16 alias sous `/mnt/AI_GENERATOR/models_acvram` ; **les 16 sont déjà effacés de l'USB** ; `grep -rl` : **52 fichiers** portent `/mnt/2TO_2023_980PRO/…` et **50** `/mnt/4TO_SATACMR_2022/Modeles/models_acvram` (lien → USB, `Sep 12`) dans `outils/`, `tests/`, `acvram/` — plus que les 42 + 8 comptés à sec ; `outils/racine_modeles.py` existe (`racine_modeles()`, ligne 33 : env → config → défaut).

## Décisions

1. **Rang du transfert : après T1 entier, bras éco et installation — dans un trou, après le T4 sous enveloppe.** Avis de Jérôme tenu, pour la raison qu'il donne : la charge hôte et l'écriture nvme3 pendant T1 sont exactement ce que le lot poste mesure ; aucune des 48 n'est sur le chemin de « terminé » (M2 = les 16 copiés). Après installation, le transfert tourne **alias par alias entre les prises M2-M3-M4** : refus de démarrer un alias si `.qui` non vide (contrôle dans le script, pas dans l'habitude), reprise au premier alias non IDENTIQUE, source effacée seulement si sha256 IDENTIQUE des deux côtés. Prédit : 60-80 min de trous cumulés, fini le 21/09 ; faux si un alias diffère (on garde l'USB, on cherche le disque).
2. **Océane, maintenant, à sec : les 100 chemins passent par `racine_modeles()`, avant T4.** Ce n'est pas préparatoire au transfert : les 8 tests qui lisent l'USB **sont déjà cassés depuis 10 h 38** (leurs alias n'y sont plus) ; T4 sous enveloppe rendrait 8 F pour un instrument, pas pour le code. Chemin de « terminé », pas hygiène.
3. **Le lien `/mnt/4TO_SATACMR_2022/Modeles/models_acvram` se repointe sur nvme3 maintenant**, pas à la fin : 50 fichiers le lisent et tout ce qu'ils cherchent (les alias mesurés) est déjà sur nvme3 ; laissé vers l'USB, il rend « absent » sur les 16 jusqu'au 21/09. `ln -sfn` (utilisateur ou Jérôme, 0 charge). Les 48 non copiés restent absents par ce lien : aucun fichier du dépôt ne les sert.

## Scellés d'Océane (à sec, un commit, `oceane-racine-modeles`)

* `grep -rlE '2TO_2023_980PRO|4TO_SATACMR_2022' outils tests acvram` = **0** hors `scratchpad/` (les 1 087 y restent : chaînes historiques, hors instrument).
* Test cassant `tests/test_racine_modeles.py` : le grep ci-dessus dans le test, 0 attendu ; la réintroduction d'un chemin rend rouge ; `racine_modeles()` lit `ACVRAM_MODELES`, puis `~/.config/acvram/modeles`, puis le défaut nvme3 — les trois cas testés avec `monkeypatch`, sans disque.
* Les 8 tests GPU : alias absent → `skip` nommé (« alias X absent sous <racine> »), jamais un F ni un pass silencieux (REGLES § 4 : conforme / non conforme / **absent**).
* Aucun `pip`, aucune compilation ; elle ne lance que **son** test (grep + monkeypatch, < 10 s, sous `nice`) ; les 8 tests modifiés sont jugés par T4 dans le trou. Prédit : 45 min, 0 changement de sortie d'aucun instrument (les chemins résolvent au même dossier par le lien repointé).

Faux si : un instrument change de dossier après le passage (`regime_ligne` ou l'en-tête portent la racine : comparer avant/après sur `verifier.sh`, `env_session`).

## Ordre

* **Jérôme** — (1) `ln -sfn /mnt/AI_GENERATOR/models_acvram /mnt/4TO_SATACMR_2022/Modeles/models_acvram` maintenant (0 charge ; si le montage 4TO n'accepte pas, le dire) ; (2) transfert des 48 **après installation**, alias par alias dans les trous, garde `.qui` dans le script, après le T4 sous enveloppe ; (3) ETAT : « T4 attend `oceane-racine-modeles` ».
* **Océane** — branche `oceane-racine-modeles` : 100 chemins → `racine_modeles()`, test cassant, skip nommé sur les 8 tests GPU ; son test seul sous `nice` ; pointeur à Jérôme.
* **Manon** — rien de changé : M00-ter → M1 bis → T1 (`ACVRAM_CPUS=0-15`, `sage-affinite-t1-20-09`).
