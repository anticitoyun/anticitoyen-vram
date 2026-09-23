# ABBA sampler b=12 Coder : A(graphe)/B(graphe+épinglé) — TENU, gain supérieur à la prédiction — 22/09 (Manon)

* instrument : `scratchpad/laurine-b12-21-09/chaine-sampler-abba.sh` (A=défaut graphe, B=`ACVRAM_RAPATRIEMENT_EPINGLE=1`), `horloge()` corrigé (`-i 0`, artefact 210 MHz du 22/09 résolu — vérifié cette fois)
* commit : main à jour, correctif 0c17a2bf inclus
* régime : Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c, b=12, ordre A1 B1 B2 A2 A3 B3 B4 A4
* scellé (Océane) : ABBA t/s B/A ≥ 1,000 ; prédiction +2,0-2,4 % (1550 → 1581-1587) ; rejet si |écart horloge| > 3 %
* mesuré : A = 1533,2/1513,7/1576,3/1539,5 (méd 1536,3) ; B = 1649,6/1601,4/1633,4/1675,8 (méd 1641,5) ; **ratio B/A = 1,0684 (+6,84 %)** ; horloge_med A=2561,0 / B=2527,0 MHz (écart 1,33 %, sous le seuil 3 %, valeurs réalistes cette fois — plus de 210 MHz constant)
* verdict : script = **INTERMÉDIAIRE**, mais **TENU, nettement au-dessus de la prédiction** (+6,84 % contre +2,0-2,4 % attendu) — cohérent avec la frontière (2 bis) qui montrait `trou_gpu` 165→24,5 µs (bien meilleur que prédit initialement) : le gain dépasse ce que le seul `trou_gpu` expliquerait sur un pas de ~6 900 µs (≈2 %), signe possible d'un effet cumulé avec le recouvrement pipeline déjà en place (levier 1). **À nommer avant de publier ce chiffre comme définitif** : gain hors fourchette prédite, alarme à lever ou confirmer par une 2e passe indépendante (bruit intra-fenêtre déjà connu ~50 t/s sur ce banc).
* durée : ~8 min (chargements + 8 fenêtres de 20 s)

## Suite
nsys + familles-noyaux ensuite (table des familles, pointeur). PPL relative A/B sur wiki-gptq. Puis C9, TRT-LLM.
