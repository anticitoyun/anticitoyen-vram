# Reste à faire — 21/09 17 h (pause)

Objectif : plus rapide et moins de joules que llama.cpp, vLLM, TensorRT-LLM. Tenu : b=1 380,8 (vLLM 290,6), prefill 22 707 (vLLM 21 054). **Non tenu : b=12 1 540 contre vLLM 1 596 (−3,5 %) ; énergie jamais mesurée à 4 moteurs.**

## A — porte immédiate (utilisateur)
1. `sudo dpkg -i acvram_0.6.34_amd64.deb` (racine du dépôt, sha256 dans poste3.md) — feu vert donné, non installé.
2. Vibe série 2 : répondue (qrvibe01.md), verdicts 2.1-2.5 en fin de qr.md — 2.4 (ordre des leviers) et 2.5 (protocole énergie) intégrés ci-dessous.
3. **OUI utilisateur (21/09 17 h) : C9 119B et bf16 30B (60 Go)** → pièces 19-20 ci-dessous.

## B — objectif b=12 (poste1, poste2, poste4)
4. Chaîne ABBA sampler (poste4 226bf4da, ≈ 8 min, poste2) : tranche H1 ordre/dérive ; le protocole intercalé devient la règle des cellules A/B.
5. Frontière de pas 0,43 ms (poste1) : d'abord la décomposer (tête, argmax 55 µs, copie ids, lancement du graphe suivant, hôte), puis leviers dans l'ordre capture de l'échantillonnage dans le graphe → argmax fusionné (plafond ≈ 0,1 ms ≈ 1,3 % : insuffisant seul, verdict 2.4) ; tête nvfp4 exclue ; chaque levier = test au bit + cellule ABBA.
6. Cellule b=12 finale ≥ 1 596 ou écart nommé comme résultat.

## C — 30B-VL (défaut de conversion, pas du moteur)
7. Reconvertir Qwen3-VL-30B sous carte.sh, contrôler `plan.tiers[].weight_format` = nvfp4, rejouer P3 (4) puis (3) (poste2, ≈ 15 min). Prédit : TTFT ≤ 0,3 s, J ≤ 60, haut_2se ≤ 3 % ; sinon plafond de la source AWQ sur la fiche.
8. Fiche alias 30B-VL mise à jour ; --decoder parc (pièce 6) après.

## D — 31B 4sur6 (qualité nvfp4)
9. Reconversion gemma-4-31B `--echelle=4sur6` en mode service (≈ 75 min, tuée à 24/60 par DUREE_MAX 1800) ; weight_format contrôlé.
10. Scellé E : chaîne à composer sur decode-pas.py (verdict-decode-pas-31b-kv-20-09), A/B/T, KL max ≤ 1,2, part amax/4 publiée.

## E — énergie et TensorRT-LLM (pièces 10-11)
11. TRT-LLM : import complet + run minimal sous carte.sh (poste3, trou avec carte) ; cellules b=1/b=12/prefill.
12. J/jeton 4 moteurs (banc-4moteurs.py) selon verdict 2.5 : J net = ∫(P − P_repos), ≥ 6 fenêtres ≥ 20 s alternées, rejet charge > 5 % / sd > 10 % / throttle actif, horloge médiane par fenêtre écart ≤ 3 %, en-tête TSV ; colonne J/jeton du README à remettre.

## F — modularisation (4 bis, à sec)
13. model.py 2-5 : deepstack, attention, couches, moe (2-3 en stash poste1 ; moe touche 4 tests carte → rejeu poste2).
14. GUI module 3 fenetre.py (fait chez poste3, preuve Xvfb à jouer, puis commit) ; écart GLib.shell_quote à trancher.
15. Ensuite : convert.py (2 398), loader.py (1 749), layers.py (1 449), kernels/__init__.py (1 401).

## G — livraison
16. Après dpkg : rejeu 19/19 installé, --decoder parc, feu vert parc (pièces 4, 6).
17. Poste THP/EPP (pièce 12, dernier : change la signature).
18. Republier GitHub (`outils/publier-github.sh --pousser`) à chaque livraison ; release du prochain moteur.

## H — ouverts par le OUI du 21/09 (après B-D, jamais avant l'objectif b=12)
19. C9 — Mistral-Small-4-119B-2603 : cache d'experts (Vibe R9 série 1 à relire), plan mémoire par la formule du pic (M1 poste4), disque 119,4 G : conception poste1 à sec (une page : paliers, exil, prédiction t/s et J), puis prise poste2 ≤ 30 min par étape.
20. bf16 30B (60 Go) : alias `Qwen3-VL-30B-A3B-awq-dequant-bf16` servi par acvram (référence P3 (3), déjà sur disque 62,1 Go) — fiche alias, cellule b=1 en étagé, sert de témoin qualité aux conversions nvfp4/4sur6.

## Sans pièce
poste4 : verdict A/B quand une cellule tombe. chef : réarmer `/loop 20m` à la relance.
