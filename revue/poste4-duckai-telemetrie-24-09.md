# duck.ai 24/09 — telemetrie Claude Code vs AGENTS.md (d'apres korben.info)

## Fait de depart, verifie chez nous (lecture seule)
`env | grep -i "TELEMETRY\|DISABLE_NONESSENTIAL\|DO_NOT_TRACK"` et `grep` sur `~/.claude/settings.json`
et `anticitoyen-vram/.claude/settings*.json` : **rien nulle part**. La telemetrie n'est coupee dans
aucun lanceur ni settings de ce circuit. Version Claude Code : 2.1.281 (`CLAUDE_CODE_EXECPATH`).

## Verifie dans la doc officielle (code.claude.com/docs/en/env-vars, section
"Features that need feature-flag fetching") — source primaire, pas duck.ai
`DISABLE_GROWTHBOOK`, `DISABLE_TELEMETRY`, `DO_NOT_TRACK`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`
font tous sauter la **recuperation des feature flags distants** (pas seulement l'envoi de metriques —
DISABLE_TELEMETRY coupe donc plus que son nom ne le laisse penser). Fetching coupe, la doc dit
explicitement qu'on perd : demarrage en mode auto par defaut (Pro/Max/Team), lecture des settings de
permission mode par l'extension VS Code, `/auto-mode-setup`, **Remote Control**, **messages entre
sessions au-dela de cette machine** (la doc precise noir sur blanc que **la messagerie entre sessions
sur la MEME machine continue de fonctionner flags coupes**), `claude import`/`/import`. La page ne
mentionne PAS AGENTS.md — c'est un point mort de la doc officielle, pas une omission de ma recherche.

## Verifie dans l'article (korben.info, 24/09/2026, citant blog.szypowi.cz — source secondaire,
non recoupee par une troisieme source independante, a traiter comme telle)
Depuis 2.1.277, la lecture native d'AGENTS.md passe par un plugin interne active par un feature flag
distant nomme `tengu_agents_md_mod`. Sans recuperation de flags (memes variables que ci-dessus), le
plugin reste eteint et AGENTS.md n'est jamais lu, **sans aucun avertissement**. Parade rapportee (Boris
Cherny, employe Claude Code, conseil donne sur un ticket GitHub avant meme la lecture native) : poser un
`CLAUDE.md` a cote contenant la seule ligne `@AGENTS.md` — cet import ne depend d'aucun flag distant et
fonctionne telemetrie coupee ou pas.

## Les trois modeles duck.ai (raisonnement pur sur ces faits, pas de recherche web demandee)

### GPT-5.6 Luna
Le plus prudent des trois. Reprend fidelement chaque fait, nomme explicitement "point non determine"
pour les mises a jour et "moins certain" pour AGENTS.md (source non officielle). Ne confond jamais
"cassee" et "incertaine". Repond a la Q2 : aucun reglage plus fin documente dans les faits fournis ;
refuse d'inventer un nom de variable. Repond a la Q3 : CLAUDE.md lu normalement, AGENTS.md natif
dependant du flag, `@AGENTS.md` dans CLAUDE.md recommande comme parade robuste, mais dit ne pas savoir
si un risque de double lecture existe. Reponse la plus fiable des trois.

### Gemma 4 31B
Reponse correcte et fidele aux faits, format tableau, dit "incertain" pour les mises a jour
(contrairement a gpt-oss). Confirme Remote Control casse, AGENTS.md casse, messagerie meme machine
intacte. Recommande `@AGENTS.md` dans CLAUDE.md "par precaution", coherent avec l'article.

### gpt-oss 120B
**Invente des noms de feature flags absents des faits fournis** : `remote_control`, `auto_mode`,
`auto_update` — aucun de ces identifiants n'etait dans le prompt ni dans les sources. Affirme sans
reserve que les mises a jour ET les sessions cloud sont cassees, alors que Luna et Gemma4 signalent ce
point comme non tranche par les faits donnes — contradiction directe malgre une consigne explicite de
dire "incertain" plutot que d'inventer. A ecarter comme source fiable sur ce sujet, deuxieme fois de
suite (meme defaut releve sur la question claude-mem plus tot dans la session).

## Contradictions notees entre les trois
Seul point de desaccord reel : gpt-oss affirme "mises a jour cassees" et "sessions cloud cassees" avec
certitude ; Luna et Gemma4 (2/3, independamment) disent ce point non tranchable par les faits fournis.
Les trois s'accordent sur : Remote Control casse, messagerie inter-session meme machine intacte,
AGENTS.md natif casse (avec la reserve de source que Luna et Gemma4 rappellent, gpt-oss l'affirme sans
reserve), aucun reglage plus fin connu que les 4 variables tout-ou-rien, `@AGENTS.md` dans CLAUDE.md
recommande dans tous les cas.

## Reponses aux trois questions de chef

**1) Que coupe DISABLE_TELEMETRY, et en plus CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC/DO_NOT_TRACK ?**
Verifie doc officielle : les trois (+ DISABLE_GROWTHBOOK) coupent la MEME chose — la recuperation des
feature flags distants (Growthbook), pas seulement l'envoi de telemetrie. Aucune des trois n'est plus
large que les autres sur ce point precis ; elles sont equivalentes pour cet effet.
Fonctions qui en dependent (verifie doc officielle) : demarrage auto-mode par defaut, lecture VS Code
des settings de permission, `/auto-mode-setup`, **Remote Control**, **messagerie entre sessions AU-DELA
de cette machine** (celle sur la MEME machine continue de marcher), `claude import`/`/import`.
Lecture native d'AGENTS.md : casse aussi, mais ce fait vient d'une source non officielle (korben.info/
blog.szypowi.cz), absente de la doc Anthropic elle-meme — a garder marque comme tel.

**2) Comment couper la telemetrie sans casser le circuit a 8 sessions (messagerie + Remote Control) ?**
Aucun des trois modeles, ni la doc officielle lue, ne documente de reglage plus fin que ces 4 variables
tout-ou-rien. Remote Control depend de la recuperation de flags : toute variable qui la coupe casse
Remote Control, sans exception connue. La messagerie inter-session SUR CETTE MACHINE n'est elle pas
affectee (confirme par la doc officielle). Donc : si le but est de reduire le trafic reseau tout en
gardant Remote Control fonctionnel, aucune des variables citees ne le permet — c'est un compromis
tout-ou-rien tel que documente aujourd'hui, pas une simple question de reglage a affiner.

**3) CLAUDE.md ET AGENTS.md : lequel est lu, faut-il @AGENTS.md dans CLAUDE.md ?**
CLAUDE.md est lu independamment des flags distants. AGENTS.md nativement (sans CLAUDE.md) ne l'est que
si le flag `tengu_agents_md_mod` repond oui — source non officielle mais coherente entre article et les
trois modeles. Reponse pratique : oui, mettre `@AGENTS.md` dans CLAUDE.md par precaution, meme
telemetrie non coupee — cet import ne depend d'aucun flag et fonctionne dans tous les cas (source :
conseil de Boris Cherny, employe Claude Code, rapporte par korben.info).
