# L'effaceur du .qui (jxm) — écartés : tmpfiles, eco.py, tout writer du dépôt

Ordre chef (23/09, après la 116) : trouver ce qui efface `.qui` alors que le flock reste tenu.
Observé en direct sur la prise de poste2 (115 bis, pid carte.sh 2884083, `env`, prise 22:45:40),
sans y toucher : guetteur en lecture seule (`scratchpad/poste3-effaceur-23-09/guet.sh`, sondage
0,2 s) + lecture du journal `/tmp/acvram-carte-0.lock.journal`.

## Fait

- `.qui` déjà ABSENT en continu de 22:59:32 (premier sondage) jusqu'à la fin de la prise —
  aucune réapparition entre-temps, alors que le point 1500→1800 (kill+relance serveur,
  22:57:04) s'est déroulé PENDANT cette fenêtre sans que rien ne change côté `.qui`. La
  disparition n'est donc PAS synchrone d'une relance HTTP `acvram serve` (hypothèse de
  chef affaiblie, pas confirmée) : elle a eu lieu une seule fois, tôt (entre 22:45:40 et
  22:59:32, avant que je regarde), puis plus rien ne l'a réécrit jusqu'à la sortie du process.
- À la sortie de 2884083 (`timeout 900` de la chaîne de poste2, expire pile à 23:00:40 =
  22:45:40 + 900 s), le propre trap de `carte.sh` (garde `jxm` déjà en place) a tenté de lire
  `.qui` : fichier introuvable, `p` vide → ligne `ANOMALIE … .qui deja repris par pid ? (jxm),
  non efface`. **`p` vide, pas un autre pid** : ce n'est PAS une reprise par un détenteur
  concurrent (le flock exclusif l'aurait empêché, et confirmé par `fuser` tout du long) — c'est
  une suppression pure du fichier, par quelque chose qui n'a jamais tenu le verrou.
- Recherche exhaustive de tout `rm`/`os.remove` touchant ce chemin dans le dépôt
  (`grep -rln "rm -f.*\.qui\|acvram-carte.*qui" outils/ scratchpad/`) : **aucun écrivain/
  effaceur** — seulement le trap gardé de `carte.sh` lui-même (qui refuse d'effacer un `.qui`
  qui n'est pas le sien) et des LECTEURS purs (`app.py:581`, `attn-isole.py:42`,
  `graphe-cout-par-noyau.py:41`, `nettoyage-modeles.sh` qui ne fait que tester `[ -e ]`).
  `eco.py:67` efface un AUTRE fichier (`acvram-eco-<idx>.json`), pas `.qui`. tmpfiles-clean
  écarté (dernier passage 18:30, 10 j de rétention, sans rapport).
- Immédiatement après la sortie de 2884083, deux autres sessions (poste5, poste4) ont pris
  et rendu le verrou en quelques secondes — comportement normal de la file, pas une preuve
  d'usurpation.

## Hypothèse retenue, non prouvée

`.qui` absent → `qui_tient()` répond « detenteur inconnu » à QUICONQUE attend (vécu moi-même
sur la 116, 22:47-22:57 : dix lignes « toujours detenteur inconnu »). Aucun script du dépôt ne
supprime ce fichier : la piste la plus probable est un geste MANUEL hors dépôt — une session en
attente, lisant « detenteur inconnu » et croyant le verrou orphelin, a pu faire un `rm -f
/tmp/acvram-carte-0.lock.qui` à la main pour « débloquer » (le flock, lui, n'est jamais touché
par un tel geste : `fuser` l'a montré tenu sans interruption). Non vérifiable après coup — aucun
journal ne trace une commande shell tapée hors carte.sh. Pas de piste rouverte côté code.

## Reste, à la reprise

- Si le phénomène se reproduit : `strace -f -e trace=unlink,unlinkat -p <pid de carte.sh, dès la
  ligne 'prise' du journal>` armé DÈS le début de la prochaine prise longue (≥ 6 points), pas
  après coup — c'est la seule preuve qui manque ici.
- Corollaire indépendant de la cause : `qui_tient()` qui répond « detenteur inconnu » alors que
  `fuser` sur le VERROU montre un pid vivant est en soi trompeur pour tout opérateur humain —
  pourrait valoir une pièce (afficher aussi le pid `fuser` brut en repli quand `.qui` manque),
  à proposer, pas décidée ici.

## Guetteur v3 (`guet3.sh`) — fausses alertes consignées, pas signalées (ordre chef 23/09 23h2x)

Deux transitions présent→absent détectées et vérifiées FAUSSES avant tout envoi (pid mort
au moment du contrôle, jamais une vraie anomalie) — classe d'erreur du détecteur lui-même,
pas du bug traqué :

1. `photo-230737.txt` (23:07:37, guet2.sh) : deux transitions collées à un relais de file
   normal (rendue `env` 2933260 tenue=371s → prise/rendue poste4 0s → prise poste5) — le
   détenteur précédent était déjà mort à chaque fois. Cause : v2 ne distinguait pas relais
   normal et anomalie. Corrigé en v3 : n'alerte que si `kill -0` réussit encore sur le
   dernier pid vu au moment de la disparition.
2. `anomalie-232310.txt` (23:23:10, guet3.sh v1) : pid 2985747 (`poste5-rust-a-vidage`),
   journal confirme `rendue … tenue=25s` PILE à cet instant — rendue normale. Cause : course
   dans le guetteur lui-même, pas dans carte.sh — le trap EXIT fait `rm -f .qui` AVANT que le
   process finisse réellement de sortir (le reaper c7w y ajoute un peu de délai), donc
   `kill -0` réussit encore pendant ce court intervalle après suppression. Corrigé (v3 courant) :
   re-vérifie `kill -0` 2 s plus tard avant de photographier ; confirmé mort dans les deux cas
   au contrôle a posteriori.

Guetteur courant (pid variable selon relance, voir `guet3.sh`) armé en continu, observateur
seul. N'écrire à chef que pour une capture qui survit à la re-vérification à 2 s (flock
toujours tenu par un pid vivant, `.qui` absent).
