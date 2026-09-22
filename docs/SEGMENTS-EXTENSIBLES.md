# Les segments extensibles coûtent un facteur 4 en débit

Mesuré le 8 septembre 2026, sur trois modèles, dans les deux sens.

## Le chiffre

Débits par passage (jetons/s), cinq passages, moteur acvram, même machine,
mode persistant actif, plafonds 500 W / 375 W :

| modèle | allocateur par blocs | `expandable_segments:True` | rapport |
|---|---|---|---|
| Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K_M | 144,4 · 161,2 · 159,2 · 142,5 · 140,9 | 36,1 · 36,4 · 36,5 · 36,5 · 36,2 | **×4,1** |
| Qwen3-30B-A3B-abliterated-erotic-AWQ | 167,4 · 180,5 · 180,4 · 180,8 · 180,7 | 45,6 · 48,1 · 47,9 · 48,1 · 47,7 | **×3,8** |
| Qwen3-VL-30B-A3B-abliterated-AWQ | 166,5 · 180,9 · 181,1 · 181,3 · 180,2 | 47,2 · 45,9 · 46,0 · 45,9 · 45,8 | **×3,9** |

## Le mécanisme, lu dans le journal et non déduit du chiffre

Les trois chargements avec l'allocateur par blocs annoncent la capture :

    graphes CUDA : actifs (decodage), 9 godets capturés d'avance
    graphes CUDA : actifs (decodage), 5 godets capturés d'avance
    graphes CUDA : actifs (decodage), 5 godets capturés d'avance

Les trois chargements avec segments extensibles **n'annoncent rien** : aucune
ligne « graphes CUDA » ne suit. Elles sont remplacées par des échecs de
mappage, 96 en tout sur la soirée :

    expandable_segments: memory mapping failed with OOM on device 0
    while trying to map 20971520 bytes (free: 2031616, total: 33670758400)

La capture de graphes réserve la mémoire par godets de 20 Mio. Avec les
segments extensibles, ces mappages échouent, la capture est abandonnée, et le
décodage retombe sur le chemin sans graphes — quatre fois plus lent.

## Ce que ça dit du démarrage à froid, et c'était la question ouverte

**Avec les graphes, le premier passage est lent ; sans les graphes, il ne
l'est plus.** 167,4 puis 180,x quatre fois avec la capture ; 45,6 · 48,1 ·
47,9 · 48,1 · 47,7 sans elle — le premier passage n'a plus rien de
particulier, et la dispersion tombe à 0,4 %.

Le coût du premier passage est donc celui de la **capture et du remplissage
des godets de graphes**, pas de la compilation de noyaux : le relevé des
caches le confirme par ailleurs — `~/.nv/ComputeCache` n'a pas été écrit
depuis le 24 août, `~/.triton/cache` pas depuis 13h07, alors que ces séries
courent de 18h25 à 19h. Rien n'a été compilé pendant les mesures.

Cela explique aussi pourquoi le mode persistant n'en retire qu'un cinquième
(10,7 → 8,7 % et 10,2 → 8,3 %) : il épargne la réinitialisation du contexte,
pas la capture.

## Comment le contrôle a failli prouver la mauvaise chose

Le premier essai a été lancé avec `ACVRAM_ALLOC_BLOCS=1`, un drapeau **retiré
du code quelques minutes plus tôt** et remplacé par `ACVRAM_ALLOC_EXTENSIBLE=1`
au sens inverse. Il ne désactivait donc rien : il tournait au défaut, qui était
redevenu l'allocateur par blocs. Le débit serait remonté à 160 et l'on aurait
conclu qu'une variable que personne ne lit répare le problème — le bon chiffre
pour la mauvaise raison. Le protocole a été refait dans le sens qui agit :
défaut d'un côté, `ACVRAM_ALLOC_EXTENSIBLE=1` de l'autre.

## Ce qui reste ouvert

Les segments extensibles avaient été posés pour une raison réelle : un
chargement mort sur 20 Mio manquants alors que 2,55 Gio étaient réservés et
inutilisés, c'est-à-dire de la fragmentation. Le facteur 4 interdit d'en faire
un défaut, mais la fragmentation, elle, n'est pas résolue. Les deux besoins
sont réels et s'opposent : il faudra soit capturer les graphes avant d'activer
les segments, soit traiter la fragmentation autrement.

Le modèle `Huihui-...-abliterated` disperse par ailleurs de façon atypique
avec l'allocateur par blocs (144,4 · 161,2 · 159,2 · 142,5 · 140,9 — pas de
motif de premier passage), alors que les deux modèles AWQ montrent la forme
habituelle. Non expliqué.
