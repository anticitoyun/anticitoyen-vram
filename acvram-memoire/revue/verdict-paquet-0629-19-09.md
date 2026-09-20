# Verdict — 0.6.29 (main 53ecbce4, mesuré sur e30e1d49 ⊇ 53ecbce4 ; = 0.6.28 + `empty_cache()` après chaque pile, défaut) : **trois bras éco conformes** — `eco=2700(2685)` posé et libre après SIGTERM ; `ECO=off` sous `-lgc` à la main → `eco=off(2677: verrou 2700 posé hors processus)` ; régime `glue=compact(8)` ; **feu vert** (le paquet est déjà installé par l'utilisateur à 07 h 12 : rien à défaire)

instrument : `scratchpad/verif-eco-19-09/chaine.sh`, 07:16:13-07:16:54, prises `carte.sh`, carte vide avant (225 MHz) ; **une première prise à 07:14 est écartée** : elle chevauchait la chaîne β (une autre prise acvram tenait le verrou éco : `eco=2700(2977: non pris)`, pid 279440 = la β en cours) — c'est le comportement attendu quand un autre processus possède le verrou, pas un défaut, mais ce n'est pas la mesure demandée ; journaux `scratchpad/cellule-0629-19-09/` (non suivis : .gitignore 38f67588)
scellé (chef 07 h 14 / REGLES § 3, avant) : régime `eco=2700(...)` juste, `-rgc` à l'arrêt, pas de charge résiduelle
mesuré : (1) réel : `eco=2700(2685)`, `etat {"mode": "2700", "pid": 288266}`, requête à 2 677 MHz, après SIGTERM `horloge=libre` (615-2 842, stable) ; (2) `-lgc` main + `ECO=off` : `eco=off(2677: verrou 2700 posé hors processus)`, requête à 2 670, après SIGTERM `lgc2700?` (posé hors acvram, non rendu : juste) ; `-rgc` rendu par moi à la fin
verdict : **TENU** ; charge utile non rejouée (le `.deb` 0.6.29 n'est pas dans l'arbre poste2 — chef l'a construit sur main ; même charge utile que 0.6.27 par construction)
durée : 41 s (07:16:13-07:16:54) + prise écartée 1 min 06 s, rédaction 2 min
suite : chef : feu vert confirmé ; ma file : C15-prefill d17a719d partie 1 → partie 2 → sélection par rang → main/d01eb2cb tranches-9 ; verdict β à écrire d'abord (journal `journal-beta-9tranches.log`)

## Rejouable
`bash scratchpad/verif-eco-19-09/chaine.sh` (1 min), carte libre.
