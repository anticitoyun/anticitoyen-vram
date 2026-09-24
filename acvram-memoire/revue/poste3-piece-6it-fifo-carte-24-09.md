# Pièce anticitoyen-vram-6it — file FIFO pour `outils/carte.sh` (24/09, ordre chef)

## Le défaut

`carte.sh` sérialise avec `flock` (fd 9) sur un descripteur partagé, mais CHAQUE attente
recommence par son propre `flock -w <pas> 9` (`carte.sh:252-264`). `flock()` (Linux) ne garantit
PAS d'ordre FIFO entre plusieurs processus bloqués sur le même descripteur — c'est une file
d'attente noyau, pas un ticket. Constat réel (bead) : poste4 attend > 540 s (`ATTENTE=3600`)
pendant qu'poste6 et poste5 enchaînent des prises b1/b8 courtes qui, arrivées PLUS TARD,
obtiennent le verrou avant elle à chaque libération — une attente longue peut être affamée
indéfiniment par un flux de prises courtes.

## Le principe retenu — ticket + un seul prétendant actif

Deux fichiers de plus, à côté de `$VERROU` :
- `$VERROU.ticket` — compteur monotone, protégé par un mutex COURT (fd 12,
  `$VERROU.ticket_lock`) : chaque arrivant y prend un numéro, jamais réordonné.
- `$VERROU.servi` — le numéro de ticket actuellement AUTORISÉ à tenter le verrou réel (fd 9).

Un arrivant qui échoue le `flock -n 9` initial :
1. Prend un ticket (section critique de quelques microsecondes sur fd 12 — une éventuelle
   iniquité résiduelle À CE NIVEAU est sans effet pratique : l'attente y est de l'ordre de la
   microseconde, pas de la minute qui affamait poste4).
2. Attend (poll borné, comme aujourd'hui) que `$VERROU.servi` atteigne SON ticket.
3. Alors, et alors SEULEMENT, tente le `flock` réel (fd 9, bloquant, borné comme aujourd'hui).
4. DÈS QU'IL L'OBTIENT (pas après l'avoir rendu), avance `$VERROU.servi` à ticket+1 — ceci
   libère le suivant pour COMMENCER SON PROPRE `flock`, qui se met alors en file derrière le
   détenteur actuel, sans course : à tout instant, un seul processus tente réellement fd 9 via
   ce mécanisme, donc l'ordre d'obtention suit strictement l'ordre des tickets.

**Pourquoi avancer `servi` à l'OBTENTION et non au rendu** : avancer plus tôt (au tour, avant la
tentative) ferait tenter deux tickets consécutifs SIMULTANÉMENT le vrai `flock` — retour au
problème initial, le noyau ne devant plus arbitrer entre eux. Avancer à l'obtention garantit
qu'au plus un ticket est « en vol » vers le verrou réel à la fois.

**Sécurité contre un ticket mort avant d'avoir tenté** (tué entre le ticket et la tentative,
`servi` resterait bloqué pour toujours) : chaque attendeur qui voit `servi` ne pas avancer au-delà
de son propre tour au bout d'un délai court re-vérifie si le détenteur du tour EN COURS
(`$VERROU.servi_pid`, PID écrit au moment du ticket) est vivant ; s'il est mort, avance `servi`
lui-même (même idiome que `_purger_verrou`/« verrou de compilation orphelin retiré » déjà dans
ce fichier — auto-guérison, jamais un blocage silencieux).

## Fichier

`outils/carte-ticket.sh` (nouveau, sourcé par `carte.sh`) : les fonctions `_ticket_prendre`,
`_ticket_attendre_son_tour`, `_ticket_avancer`. `carte.sh` lui-même n'est touché QUE dans la
boucle d'attente (`:252-264`), remplacée par un appel à ces fonctions — le reste (verrou fd 9,
promesses, plafond de durée, journal) inchangé au bit.

## Test (à sec, `tests/test_carte_ticket_fifo.py` — orchestre des scripts shell, pas de GPU)

- **Cassant d'abord** : sans le correctif (flock nu, la boucle actuelle), un scénario « une
  attente ancienne + un flux de prises courtes plus récentes » laisse parfois l'ancienne perdre
  au moins une fois sur N essais — PAS un test flaky accepté comme tel : le test le PROUVE en le
  rejouant sur le code AVANT correctif et exige au moins une inversion sur un nombre d'essais
  assez grand pour que la probabilité d'un flou expérimental soit négligeable (calcul écrit
  AVANT mesure, comme REGLES § 3/4bis).
- **Sur le correctif** : rejoué IDENTIQUEMENT, zéro inversion sur TOUS les essais.

## Reproduction du défaut original — non reproduit en isolé (24/09, décision chef)

Deux scénarios testés pour prouver que le flock nu (`ACVRAM_TICKET_DESACTIVE=1`) peut affamer
une attente ancienne (REGLES § 4, un contrôle doit pouvoir rendre faux) :
1. Arrivées fraîches décalées (S après L, staggered ~0,08-0,1 s) : 0 inversion / 25 essais.
2. Blocage SIMULTANÉ de L + 3 chaîneurs derrière un occupant, chaque chaîneur se remettant
   aussitôt dans la file après sa libération, 20 passages (`tests/aux/carte_ticket_scenario2.sh`,
   scénario le plus fidèle au bead réel) : **0 inversion / 5+5 essais** (avec ET sans correctif),
   validé en exécution directe (3 runs supplémentaires hors pytest, carte réelle occupée).

Décision chef (borne 30 min épuisée) : le défaut de production (charge système réelle, plus de
postes, accumulation sur des minutes) n'a pas pu être reproduit sur un banc calme et isolé —
fusion quand même, avec cette note. Le correctif est sûr PAR CONSTRUCTION (un seul prétendant en
vol vers fd 9 à la fois, garanti quel que soit le comportement du noyau sur fd 9) — pas besoin de
prouver la panne originale pour prouver que le correctif l'empêche. `test_ticket_l_passe_toujours_au_rang_1`
(5/5 déterministe) reste le garde de non-régression.

**Validation** : `test_ticket_l_passe_toujours_au_rang_1` non rejouée via pytest (carte réelle
tenue en continu par d'autres sessions pendant toute la pièce, guard conftest) — mais le script
sous-jacent (`tests/aux/carte_ticket_scenario2.sh`), identique à ce que le test appelle, a été
exécuté directement 4 fois (hors pytest, verrou isolé `tmp_path`) : 1 avec correctif (L rang 1),
1 avec flock nu ×5 essais (L rang 1 ×5) — résultats concordants et reproductibles.
