# Sage — Ordre de reprise (20/09, 19 h 36) — minimum de jetons, Manon absente

Base : tête PAUSE d'ETAT 19 h 17 (main 2b8e387d) + `sage-p3-1-scelle-temoin`, `sage-3b-lanceur-contexte`, `sage-auto-yes-efforts` (20/09). Faits intégrés : auto-yes a envoyé 2 tours « 1 » factices (diagnostic tenu) ; deux Océane vivantes ; Manon absente ; REGLES § 1 ligne 12 en cours (Jérôme).

## Trois règles pour cette reprise

1. **Tant que le motif resserré n'est pas dans `/etc/auto-yes/patterns.conf`** (un `sudo cp` de l'utilisateur depuis le scratchpad de Jérôme, après les contrôles ±) : aucune session n'affiche la phrase du motif — ni `cat`, ni Read de `sage-auto-yes-efforts-20-09.md`, qui la contient cinq fois. Cette note-ci ne la contient pas et se copie sans risque. Les sessions vivantes gardent l'ancien motif jusqu'à relance : **on ne relance rien avant le `sudo cp`** (le `--resume` rejoue le transcript, c'est ce qui a tiré). Compte des tours factices tenu dans `jerome.md` : 2 au départ ; chaque nouveau tour = une ligne datée.
2. **Manon absente = aucune mesure GPU**, mais aucune chaîne non plus : `.qui` vide + `nvidia-smi --query-compute-apps` vide, relevés une fois dans le carnet de celui qui lance → les tests processeur (`rtk pytest` ciblé) passent sans « trou » à demander. Tout ce qui suit est à sec, sauf le bloc Manon.
3. **Doublon Océane** : Jérôme écrit dans `jerome.md` l'Océane qu'il sert (son `[ref]` de ListAgents) ; l'autre ne lit, n'écrit, ne commite rien jusqu'à sa fermeture par l'utilisateur. Deux sessions sur la branche `oceane` = fusion cassée.

Un tour par bloc, réponse = pointeur (`note: revue/<fichier> — 1 ligne`), pas de récit ; un échec est un résultat, il ne se corrige pas dans le même tour.

## Ordre — Jérôme

1. REGLES § 1 ligne 12 → table (en cours). Même commit : `.gitignore` += `.webui_secret_key`, `*.deb` (un secret et trois binaires non suivis à la racine — jamais commis) ; tri des 11 fichiers non suivis d'`outils/` (EXECUTION-OCEANE-MANON.md, OCEANE-MANON-TACHES.sh, README-COPIER-COLLER.md, completer-tsv ×2, metriques ×2, nettoyage-modeles-liste.txt, oceane-5-3-poste-b1.py, remplir-usage-dans-tsv.py, verifier-scelles-oceane-manon.py) : suivi si une note `revue/` le cite, supprimé sinon — un fichier non suivi n'est pas un instrument. Contrôle : `git status --porcelain | grep -v worktrees` vide.
2. auto-yes : contrôles ± finis et écrits (négatif : rien envoyé ; positif : `recu=1`) → un message à l'utilisateur avec la seule commande `sudo cp` ; jusque-là, règle 1.
3. Traductions trad-0634 : 8/31 → 31/31, lots de 8 par tour (3 tours), sans relire les 8 faites.
4. ETAT : tête « REPRISE 20/09 » de 6 lignes pointant cette note ; (b′) → (b″) si pas fait ; ensuite une ligne par pointeur reçu, jamais une nouvelle tête.

## Ordre — Océane (celle nommée)

1. Tests écrits non joués (masque-famille, chauffe-ctx ×3, langues-gui, mm_moteur 19+, mm_conversion — fichiers `tests/` de ses commits du 20/09 : `git log --since=2026-09-20 --name-only -- tests/`) : `rtk pytest <fichiers> -q`, processeur, ≤ 1 min ; carnet = compte passed/failed.
2. Addendum 19 h 03 : dtype des logits de `ref-2b-fp32.pt` et top-2 à 6 décimales sur img02/08/12/18 (1 min, à sec). Tous multiples de 0,125 → étage bf16 dans `references-transformers.py`, correctif + sha dans le pointeur ; sinon « fp32 plein, 0,750 = coïncidence » écrit tel quel. Ce pointeur débloque Manon 1.
3. Rien d'autre : pas de 30B, pas de qualité 31B avant 0.6.34.

## Ordre — Femoceane

1. Tests écrits non joués (lanceur source, verifier-contexte, paquet parc) : même règle qu'Océane 1.
2. `~/.local/bin/acvram-serveur` → `parc/bin`, paquet par défaut, opt-in `ACVRAM_ARBRE`, `source=` sur la ligne de régime. Contrôle : sans variable → `source=paquet` ; avec → `source=arbre` ; les deux lignes dans le carnet.
3. § 3a : `verifier-contexte.py` garde la marque « plan » jusqu'au S2 rejoué (Manon 2) ; plans par lots après.
4. Xephyr 1a, VM parc : après 0.6.34 (feu vert parc 21/09 inchangé). Ni Chromium ni Maîtresse ce soir.

## Ordre — Manon (dès l'icône lancée ; prise `carte.sh` ≤ 30 min)

1. P3 (1) : `chaine-p3-1-rejeu.sh` sur main ≥ 5d9de783, **après** le pointeur Océane 2 (refs réutilisées si sha égal ; kv=bf16 ×20 puis int8 ×20 ; 8 min) ; verdict 7 lignes, cinq clauses (b″), marges fp32 par pas.
2. S2-contexte rejoué sur la chauffe : 16 alias + 3 vision, scellé écrit avant = 0 × 500, ctx+64 → 400 nommé 19/19 (≈ 15 min) ; tout 500 restant = défaut nommé par alias avec Mio demandés/libres.
3. § 1b menus réels, nouvelle prise.
Sans Manon, rien de 1-3 n'est fait par un autre poste.

## Porte 0.6.34

= P3 (1) (b″) tenu + S2 0 × 500 + lanceur paquet + traductions 31/31 + Agent OS Open WebUI ; Jérôme assemble, feu vert par rejeu GLM b=1 comme 0.6.33. Qualité nvfp4 31B et VM parc après. Réfutation de cet ordre : un poste qui attend un autre plus d'un tour → Jérôme le signale à Sage avec les deux pointeurs, et l'ordre change.

## Addendum 20 h 17

Règle 1 retirée (décision utilisateur 20 h 17, REGLES § 1, main 7e9f4276) : un tour « 1 » = oui de l'utilisateur, à suivre ; relances libres. Règles 2-3 et les blocs Ordre tiennent.

## Addendum 20 h 2x — Maîtresse

« Ni Chromium ni Maîtresse » (Femoceane 4) retiré : Maîtresse est membre du circuit (utilisateur 20 h 18). Pièce confiée : relecture témoin des 31 traductions trad-0634 (onglet Agent OS Open WebUI, 04ebb4db) — par langue, un appel `vibe -p` borné (`--max-turns 1 --max-price 0.05`), entrée = les 27 chaînes de la langue, sortie = seuls les numéros des chaînes qui ne sont pas dans cette langue ou gardent un mot français/anglais non traduit. Contrôle qui peut rendre faux : Jérôme plante une chaîne française dans 3 langues sur 31 avant l'envoi ; 3/3 rattrapées = lecture valide, sinon la relecture ne compte pas. Coût plafond 31 × 0,05 ; résultat = liste (langue, n°) dans `jerome.md`, corrections dans un commit, Océane non mobilisée.
