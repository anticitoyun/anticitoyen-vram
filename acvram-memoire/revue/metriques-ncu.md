# Métriques `ncu` — vocabulaire commun

Établi par poste2 le 9 septembre 2026, après une mesure de trafic DRAM validée
à 0,11 % sur un cas de réponse connue. **Toute mesure part de ces noms.**

## Les noms qui marchent sur Blackwell (GB202, CC 12.0)

    dram__bytes_op_read.sum        octets lus de la DRAM
    dram__bytes_op_write.sum       octets ecrits
    dram__sectors_op_read.sum      secteurs lus (32 o chacun) -> coalescence
    dram__bytes.sum                total accede

## Les noms qui rendent `n/a` — et le piège qu'ils tendent

**`dram__bytes_read.sum` et `dram__bytes_write.sum` N'EXISTENT PAS** sur cette
carte. `ncu` ne refuse pas : il accepte la métrique et **rend `n/a`**.

Un lecteur qui somme les valeurs obtient alors **zéro**, et zéro octet lu est
un résultat que rien ne signale comme absurde si les autres chiffres sont
plausibles. C'est l'absence lue comme un résultat, appliquée à un compteur
matériel.

**Vérifier les noms disponibles plutôt que les supposer :**

    ncu --query-metrics | grep -E "^ *(dram|lts__|l1tex__)"

## Deux pièges de lecture du CSV, tous deux silencieux

**L'en-tête n'est pas la première ligne.** La sortie du programme profilé se
mêle au CSV sur le même flux ; `csv.DictReader` prend alors la première ligne
venue pour un en-tête et rend des colonnes vides. Sauter jusqu'à la ligne qui
commence par `"ID"`.

**Les grands nombres portent des espaces** comme séparateurs de milliers
(`33 592 576`). `float()` échoue dessus, et un `except: pass` fait disparaître
**exactement les grandes valeurs** — le total reste plausible et vaut le
millième de la vérité. Compter les valeurs illisibles et **le dire**, jamais
les ignorer.

Lecteur de référence : `scratchpad/lire-ncu.py`, qui traite les trois.

## Délimiter la mesure

`ncu` profile tous les noyaux, chargement compris. Marquer la plage utile avec
NVTX (`torch.cuda.nvtx.range_push("pas")`) et profiler avec
`--nvtx --nvtx-include "pas/"`. Trois pas de chauffe **hors** plage : la
première exécution capture les graphes et remplit les caches.

## Ce que `ncu` ne peut pas mesurer

**Les fréquences.** `ncu` **fige les horloges** pour rendre les compteurs
reproductibles. `mclk` et `pclk` se relèvent donc dans une exécution
**séparée**, sans profilage, par `nvidia-smi dmon`. Et le débit observé sous
`ncu` ne vaut rien : seuls les compteurs comptent.

## L'interpréteur : un seul possible

    venv acvram   torch 2.13.0+cu130   triton 3.7.1     <- celui-ci
    systeme       ni torch ni triton

Triton **est** dans le venv. Toute mesure comparable passe par
`anticitoyen-vram/.venv/bin/python` — deux piles CUDA différentes sous le même
instrument rendraient les chiffres incomparables.

## Valider l'instrument avant de l'utiliser

Une GEMV `4096 × 4096` en bf16 doit lire 33 554 432 octets. Mesuré :
**33 592 576**, soit **+0,11 %**. Un instrument qu'on n'a pas confronté à une
réponse connue n'est pas un instrument.
