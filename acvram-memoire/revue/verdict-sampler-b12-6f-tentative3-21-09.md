# Sampler vectorisé b=12, 6f — tentative 3 — 21/09

* instrument : `scratchpad/laurine-b12-21-09/chaine-sampler-6f.sh`
* commit : d12b2a08, worktree `manon-d12b2a08-21-09`, arbres A/B déjà chauds (15s/14s)
* régime : garde load1-écran ≤ 1,5 avant chaque fenêtre
* scellé : identique aux tentatives précédentes
* mesuré : A1 1541,3 t/s (watts 382,4, horloge 2512-2692) ; toutes les fenêtres suivantes (B1→B3, A2, A3) REJETÉES, load1 remonté à 1,92 juste après A1 et resté au-dessus du seuil
* verdict : **INDÉCIDABLE, 1 point A / 0 point B** — la garde consécutive (fenêtres dos à dos) est quasi infranchissable dans ces conditions : load1 (moyenne glissante 1 min) reste élevé juste après l'activité de mesure elle-même, jamais le temps de redescendre entre deux fenêtres rapprochées. Watts A1 = 382,4, sous le seuil de 400 mais proche.
* durée : 94 s (15:33:41–15:35:15)

nvidia-smi propre avant/après. Signalé à la Maîtresse : espacer les fenêtres ou assouplir le seuil, sinon la garde ferme systématiquement après la 1re mesure.
