# poste7 — point de pause (utilisateur 06 h 43 : « pause du groupe dès que possible ») : état servi, ce qui est en vol, par quoi reprendre, ce qui attend un mot de l'utilisateur (20/09, 06 h 43, horloge machine)

Source : `poste7-bilan-nuit-20-09` (point d'étape complet, 05 h 51) ; `poste7-reprise-20-09` (ordre des chantiers) ; `poste7-alerte-ppl-glm-8k-20-09` addendum 06 h 35 ; protocole de pause de chef (aucun nouveau lancement, chaînes finies ou arrêtées proprement, carte rendue, `-rgc`, verrou libéré, commit + push, dernière ligne à ETAT).

## 1. État à la pause
* **Servi** : défaut **0.6.24** (main 6c7bb533) — éco 2 700 (= 400 W à b=12), TF32 cœur MLA ≤ 2 048 clés, C14 + C14-c, C15 niveau 1, C15-3d glue compacte (8 warps) ; **0.6.25 / 0.6.27** construits (ligne de régime complète, GUI), feu vert donné, installation à la main de l'utilisateur ; opt-in : C4, C10 (b), C5-b, C17, `glue` niveau 2.
* **Cellules** : `poste7-bilan-nuit-20-09` § 2 (Coder b=1 devant ; b=12 1 397 · 0,210 derrière vLLM Marlin 1 626 · 0,136 ; prefill second ; GLM derrière en vitesse, devant en J ; 119B llama.cpp 24,2).
* **En vol à l'instant de la pause** : chaîne 9 tranches préfixée (poste2, d81aa312 + 5a3d5ba3) — elle finit son bloc ou s'arrête ; son verdict, s'il vient, se lit avec les deux prédictions du 06 h 35 (main contre d01eb2cb ≤ +0,5 % à 2 SE ; dispersion par tranche ≥ ± 1 %). Rien d'autre ne tourne.
* **Instrument à retenir** : `ppl-decode-kv-17-09.py` n'a jamais préfixé `[gMASK]<sop>` — toutes les PPL de décodage GLM de la nuit sont non préfixées ; écarts valides, absolus non ; la version préfixée est celle de la chaîne 9 tranches.

## 2. Par quoi reprendre (l'ordre de `poste7-reprise-20-09` tient tel quel)
poste1 : rangs restants de la clôture (gemma repli annoncé, C3 estimation) → niveau 2 (juge 9 tranches, pas de défaut à trouver) → gemma / Ornith → C15-prefill (≥ 20 500 j/s) → niveau 3 GLM (≤ 700 nœuds) → C14-b → C13-c réécrit (sonde bf16 d'abord) → sélection ≤ 3 µs et bande de Marlin. poste2 : fenêtres aux pointeurs ; hygiène (C5-b nsys, déterminisme ON/OFF Coder b=12, PPL longue tf32). chef : tri des worktrees, un .deb par défaut avec ses bras, ETAT seule entrée.

## 3. Ce qui attend un mot de l'utilisateur
1. **119B** : (a) 3080 Ti calculante, (b) experts sur processeur, (c) aucune — ou **(d) 5090 seule en × 16 + cache PCIe** (≈ 24 j/s sans cache, ≈ 48 avec, si le lien mesure ≥ 40 Go/s) ; mon avis : (d) si le × 16 se confirme, sinon (c).
2. **Configuration** : 5090 seule en × 16 (ou 3080 Ti sur un port × 4 du chipset) — à vérifier par `bench_link_bandwidth` ≥ 40 Go/s après le changement ; les trois services permanents à reloger.
3. Installation de 0.6.25 / 0.6.27 (`sudo dpkg -i`, puis `acvram doctor`).

Règles vivantes pour la reprise, inchangées : `date` avant toute heure ; pointeurs, jamais de réécriture ; ETAT ≤ 40 lignes seule entrée ; un scellé par fenêtre, prédiction et issue gênante écrites avant ; instrument = fichier suivi + commit ; scellé de J relatif à la fenêtre ; équivalence par noyau, PPL avec son SE. Silence de poste7 jusqu'à la reprise.
