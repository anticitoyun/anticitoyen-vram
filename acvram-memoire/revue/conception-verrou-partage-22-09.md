# Conception — classe PARTAGÉ du verrou carte.sh (pièce 21, poste3, 22/09)

Document de CONCEPTION, avant tout code (ordre chef). Il pose le modèle, la
matrice de compatibilité, le mapping flock, la promesse et son contrôle, le test
cassant. Le code (`outils/carte.sh`) ne s'écrit qu'après validation de cette page.

## Le problème

Le verrou actuel est binaire : `flock` EXCLUSIF sur un seul fichier (fd 9). Tout
ce qui prend la carte — mesure chronométrée, service permanent, **conversion
longue** — prend le même verrou exclusif. Or une conversion `acvram convert
--quant-device cpu` calcule sur le PROCESSEUR : elle lit le modèle depuis le
disque, quantifie en RAM, et ne touche presque pas le GPU (≤ quelques centaines
de Mio, aucun noyau lourd). Pendant qu'elle tient le verrou exclusif (parfois
> 30 min), aucune autre prise n'est possible, alors que le GPU est libre. C'est
le cas vécu aujourd'hui : la reconversion 31B a bloqué la carte des heures pour
un travail qui ne l'utilisait pas.

## Trois classes de prise

| classe   | ce qu'elle fait du GPU                     | exemple |
|----------|--------------------------------------------|---------|
| PARTAGÉ  | rien de lourd : ≤ 512 Mio, aucun noyau     | `acvram convert … --quant-device cpu`, tri/copie |
| SERVICE  | sert des requêtes (GPU réel), permanent    | acvram-serveur, llamacpp-serveur, vllm-serveur |
| MESURE   | calcul GPU chronométré, veut le silence    | cellules de perf, bancs, ≥ 20 s au compteur |

## Matrice de compatibilité (qui peut coexister)

|              | PARTAGÉ | SERVICE | MESURE |
|--------------|:-------:|:-------:|:------:|
| **PARTAGÉ**  |   oui   |   oui   |  non   |
| **SERVICE**  |   oui   |   non¹  |  non²  |
| **MESURE**   |   non   |   non²  |  non   |

¹ deux services sur une même carte = collision de VRAM (refus nommé par le `.qui`,
déjà en place côté mode service). ² refus mutuel MESURE↔SERVICE : déjà codé
(carte.sh, bloc `flock -n`, code 4) — une mesure n'attend pas un service qui ne
se libère jamais, et l'inverse. **Le seul ajout de cette pièce : la ligne/colonne
PARTAGÉ.** PARTAGÉ coexiste avec PARTAGÉ et SERVICE, s'exclut avec MESURE.

## Mapping flock — deux fichiers

Un readers-writers à deux descripteurs. Fichier P = `…/acvram-carte-N.lock`
(existant), fichier S = `…/acvram-carte-N.share.lock` (nouveau).

- **PARTAGÉ** : `flock LOCK_SH` sur S seulement. Plusieurs SH coexistent (autres
  partagés + services). Ne touche jamais P.
- **SERVICE** : `flock LOCK_SH` sur S (coexiste avec les partagés), **+** `flock
  LOCK_EX` sur P (un seul service : exclut les autres services). Le fd P hérité
  par le serveur détaché le tient vivant (mécanisme actuel du mode service).
- **MESURE** : `flock LOCK_EX` sur S **et** `LOCK_EX` sur P. `LOCK_EX` sur S
  attend/refuse tous les SH (partagés ET services) → la mesure est seule ;
  `LOCK_EX` sur P refuse les autres mesures et services.

Pourquoi deux fichiers et pas un seul SH/EX : S porte la coexistence
(partagé+service), P porte l'unicité (un service OU une mesure, jamais deux).
Un seul fichier ne distinguerait pas « service seul » (autorisé avec partagés)
de « mesure seule » (exclut tout).

Le refus reste NON BLOQUANT et NOMMÉ : `flock -n`, et sur échec on lit les `.qui`
pour dire qui tient quoi (déjà le motif actuel), code 4 pour les incompatibles
(mesure vs service, mesure vs partagé), attente bornée seulement entre pairs
compatibles qui se sérialisent.

## `.qui` par job

Aujourd'hui un seul `…​.lock.qui`. Avec des partagés multiples, un fichier par
job : `…/acvram-carte-N.share.<pid>.qui` (format 4 champs `<pid> <epoch> <nom>
partage`). `qui_tient()` agrège les `.share.*.qui` vivants + le `.qui` exclusif.
Un partagé mort laisse un `.qui` périmé, nettoyé comme aujourd'hui (kill -0).

## La promesse et son contrôle (jamais de kill)

Un PARTAGÉ **promet** : aucun processus de calcul GPU à lui, ≤ 512 Mio de VRAM.
La promesse est VÉRIFIÉE, pas supposée : à chaque nouvelle prise (de n'importe
quelle classe), carte.sh relève `nvidia-smi --query-compute-apps=pid,used_memory`
et, pour chaque partagé déclaré (`.share.*.qui`), vérifie qu'aucun de ses PID (ni
descendant) n'a de contexte GPU > 512 Mio. **Si la promesse est violée** : la
prise en cours est REFUSÉE (le partagé fautif est nommé, code dédié) et une ligne
`PROMESSE-VIOLEE` est écrite au journal. **Jamais de kill** — on refuse et on
journalise, l'humain tranche (REGLES : ne jamais tuer un processus GPU inconnu).

## Test cassant (à écrire avec le code)

`tests/test_verrou_partage.py`, à sec, verrou isolé par `ACVRAM_VERROU` (jamais
les vrais `/tmp/acvram-carte-*`) :
1. **deux PARTAGÉS coexistent** : deux `sleep` en LOCK_SH démarrent tous deux,
   leurs deux `.share.*.qui` présents.
2. **un PARTAGÉ ne bloque pas un SERVICE** : partagé en cours → un service
   démarre quand même (le test CASSE si le service est refusé/attend).
3. **une MESURE exclut les PARTAGÉS** : partagé en cours → une mesure est refusée
   nommée (code 4), sans attendre 30 min.
4. **deux MESURES / mesure+service ne coexistent jamais** (régression du refus
   mutuel existant).
5. **promesse violée** : un faux partagé qui alloue > 512 Mio (témoin) → la prise
   suivante est refusée + `PROMESSE-VIOLEE` au journal, et AUCUN kill (le faux
   partagé vit encore après).

## Points d'implémentation (carte.sh)

- `ACVRAM_TYPE` accepte `partage` en plus de `mesure|etat|service`.
- ouvrir S (`exec 8>"$SHARE"`) en plus de P (fd 9) ; `flock -s 8` pour partagé,
  `flock 8 && flock 9` pour mesure, `flock -s 8 && flock 9` pour service.
- pas de plafond 30 min pour `partage` (comme `service`) : une conversion longue
  est légitime ; `DUREE_MAX=0`.
- le contrôle de promesse est une fonction `_verifier_promesses()` appelée avant
  d'écrire le `.qui`.

## Questions ouvertes (à trancher avant le code)

1. SERVICE prend-il `LOCK_SH` sur S (ma proposition) ou faut-il un 3e état pour
   qu'une mesure puisse distinguer « attendre un service » de « refuser » ? Ma
   proposition garde le refus mutuel mesure↔service par le `.qui` (type lu), pas
   par la seule sémantique flock — à confirmer.
2. Seuil 512 Mio : fixe, ou `ACVRAM_PROMESSE_MIO` ? (je propose la variable, défaut 512).
3. Le contrôle de promesse tourne-t-il aussi périodiquement (guet.sh) ou seulement
   à chaque prise ? (je propose : à chaque prise, guet.sh le relaie déjà pour les intrus.)
