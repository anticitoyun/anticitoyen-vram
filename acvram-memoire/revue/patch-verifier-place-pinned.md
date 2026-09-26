# Correctif soumis : `verifier_place` est aveugle à la mémoire DMA-pinned

Soumis par poste2 le 8 septembre 2026, **non poussé**, en attente de revue.
Ce correctif corrige un défaut de mon propre code, celui-là même que j'avais
écrit pour attraper le motif « un compteur qui mesure la propriété voisine ».

## Le défaut

`outils/banc-4moteurs.py`, second test de `verifier_place`, borne le
verrouillage total avec `ram_verrouillee_mio()` = `Unevictable` de
`/proc/meminfo`.

**Or la mémoire épinglée par le pilote CUDA n'apparaît PAS en `Unevictable`.**
Preuve, machine au repos, sans rien charger :

    nr_foll_pin_acquired  137 375 647
    nr_foll_pin_released   137 110 943
    → 264 704 pages = 1,01 Go actuellement DMA-pinned
    nr_unevictable         65 227 pages = 255 Mio  (ne contient pas ce Go)

Le noyau laisse une page DMA-pinned sur sa LRU d'origine (souvent shmem) et la
saute à la réclamation via `page_maybe_dma_pinned()` ; elle n'est jamais
déplacée vers la LRU `unevictable`. Donc le compteur que j'ai choisi pour
mesurer le verrouillage est aveugle à **la forme de verrouillage même qui a
paralysé la machine** — les poids exilés, épinglés pour le DMA.

## Le correctif

Ajouter le proxy global du DMA-pin, et l'inclure dans le terme « verrouillé ».

    def ram_pinned_mio():
        """Mémoire DMA-pinned active (pin_user_pages), en Mio.

        Ni Unevictable ni Mlocked ne la comptent : le pilote CUDA épingle au
        niveau de la page, pas de la VMA ni par mlock(). Seul foll_pin la voit.
        Preuve au repos le 8/09/2026 : 264 704 pages pinned, 0 en plus en
        Unevictable. Compteur GLOBAL (toute la machine), ce qui est justement
        ce que borne verifier_place : le verrouillage total contre la RAM
        totale, pas le nôtre seul.
        """
        acq = rel = None
        try:
            for l in open("/proc/vmstat", encoding="utf-8"):
                if l.startswith("nr_foll_pin_acquired"):
                    acq = int(l.split()[1])
                elif l.startswith("nr_foll_pin_released"):
                    rel = int(l.split()[1])
        except OSError:
            return 0
        if acq is None or rel is None:
            return 0
        return max(0, acq - rel) * 4096 // (1024 * 1024)

Et dans `verifier_place`, le terme déjà-verrouillé devient la somme :

    verrouille = (ram_verrouillee_mio() if verrouille_mio is None
                  else verrouille_mio) + ram_pinned_mio()

## Réserve, écrite d'avance — je ne la cache pas

**Double comptage possible.** Une page à la fois `mlock()`-ée ET DMA-pinned
serait comptée deux fois (Unevictable + foll_pin). C'est rare, et l'erreur va
dans le **sens du refus** : on surestime le verrouillé, on refuse un peu trop
tôt. Pour une garde de sécurité, c'est le bon sens d'erreur — l'inverse a coûté
la machine. Je le documente plutôt que de le prétendre exact.

**`foll_pin` fluctue.** Il inclut les pins transitoires de toute la machine
(I/O en vol, autres processus). Au repos il est stable à ~1 Go ; sous charge il
bouge. Une borne de sécurité s'en accommode ; un chiffre publié, non — donc ce
compteur sert à `verifier_place` (décider), jamais à un résultat.

## Ce que ce correctif ne fait PAS

Il ne rend pas `verifier_place` capable de prédire l'empreinte du chargement à
venir : il mesure ce qui est **déjà** verrouillé au moment de l'appel, pas ce
que le modèle va épingler. La prédiction de l'exil reste le premier test
(poids − VRAM libre). Ce correctif ne touche que le second test, le garde-fou
« que reste-t-il au système ».

## Dépendance au départage (A)/(B)

Ce correctif est valable **quel que soit** le résultat du test `memory.reclaim`
en cours : le fait que `Unevictable` sous-compte le pinned est déjà établi au
repos, indépendamment de savoir si les 24 Go de shmem du chargement sont
swappables ou non. Le départage (A)/(B) change la stratégie de barrière du
témoin, pas le compteur que doit lire `verifier_place`.
