# Diagnostic mla_decode confirmé sur carte — 13/09/2026

Laure, sur carte (`outils/carte.sh`), ~10 min. Suite à
[occupation-mla-decode-batching-slots.md](occupation-mla-decode-batching-slots.md),
avant que Laurine code `anticitoyen-vram-6wa`. `torch.profiler`, un pas de
décodage, GLM-4.7-Grande-Heretic-42B, `slots=12`, `ctx=2048`, **eager**
(`enable_cuda_graphs=False`, demandé par Jérôme) après amorçage des 12
créneaux.

## Chiffres mesurés

    lancements mla_scores_kernel + mla_reduce_kernel :  1 584  (792 + 792 = 12 créneaux x 66 couches)
    mla_reduce_kernel : 20,600 ms self-CUDA / 792 lancements  = 26,0 us/lancement
    mla_scores_kernel : 12,182 ms self-CUDA / 792 lancements  = 15,4 us/lancement
    pas mur (eager, sync autour)                        : 128,854 ms
    total CUDA busy (tous noyaux confondus)              :  70,331 ms
    part mla_* dans le pas mur                           :  25,4 %
    part mla_* dans le temps CUDA total                  :  46,6 %

## Verdict contre les seuils de Jérôme

**≥ 1 000 lancements et ≥ 25 % du pas : les deux sont franchis (1 584 et
25,4 %).** Le nombre de lancements confirme au chiffre près la lecture de code
(`occupation-mla-decode-batching-slots.md` §1 : 12 créneaux × 66 couches). Le
bead `anticitoyen-vram-6wa` est confirmé ; les chiffres y sont ajoutés
(`bd update`).

## Réserve — le régime mesuré n'est pas celui qui compte in fine

Cette mesure est **eager**, comme demandé, pour isoler les noyaux sans le
biais du replay de graphe (cf. la piste 1 du 11/09 : `.replay()` sans sync ne
mesure que le lancement). Le pas mur de 128,85 ms **n'est pas comparable** au
pas sous graphes capturés (~18,8 ms, campagne du 11/09) : l'essentiel de
l'écart eager/graphe est l'overhead de lancement Python par opération
(70,3 ms de calcul CUDA réel logés dans 128,9 ms de mur — 58,5 ms d'écarts,
files d'attente et lancement), pas les deux noyaux MLA eux-mêmes.

Ce que cette mesure établit : **le nombre de lancements et leur poids relatif
au sein du travail GPU réellement exécuté** (46,6 % du temps CUDA total,
robuste au régime). Ce qu'elle n'établit pas : le pourcentage exact sous
graphes capturés — à mesurer par événements CUDA intra-graphe si le verdict
du patch batché reste ambigu (protocole déjà écrit dans
`occupation-mla-decode-batching-slots.md` §5).

## Anomalie notée, non creusée

Trois échecs d'allocation transitoires (~400 Mo chacun) pendant l'amorçage
des 12 créneaux, avant le pas profilé — résolus sans interrompre la mesure.
Pas de `reset_peak_memory_stats` dans ce script (hors protocole demandé) ;
à surveiller si `slots=12` + `ctx=2048` tourne près du plafond VRAM en eager,
sans plus d'affirmation ici.

Script : `scratchpad/profil-mla-eager.py` (outil ponctuel, non versionné dans
`outils/`).
