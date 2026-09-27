# Pièce 276 g — recouvrement de la tour de vision : à sec, scellé AVANT le code

poste6, 27/09/2026 11 h 4x. Ordre chef : l'encodage des images d'une requête part HORS de `_admit`, dès la préparation, et se
recouvre avec le pas en cours, sans grouper. Levier nommé par la 276 f § 4.2 (main gagne le TTFT moyen par ses vagues : le
préfill des premières recouvre la préparation des dernières ; grouper dégrade ; le lot n'est pas au bit).

## Mécanisme
Aujourd'hui (main a6268af2e) : préparation dans la boucle HTTP (12 × 2 ms), puis `_admit` (fil moteur) fait la tour image par
image (≈ 9 ms/image, 45 ms médian par admission de vague, 107 ms pour 12) AVANT le préfill : le pas en cours attend la tour.
276 g : après `preparer_images`, le gestionnaire appelle `engine.encoder_images` dans un fil (`asyncio.to_thread`) : MÊME appel
`traits_niveaux` par image (aucun lot), sur un flux CUDA annexe, sous le verrou de capture des graphes (une capture en cours et un
noyau lancé d'un autre fil s'invalident mutuellement — leçon runner.py:492) ; `add_request(traits=…)` pose `image_embeds`,
`image_niveaux` et les positions M-RoPE, `_admit` ne fait plus rien pour ces images. La boucle reste libre pendant la tour (fil),
la tour d'une requête se recouvre avec la préparation de la suivante et avec le pas moteur.

## Critère d'identité (écrit avant de coder)
Pas de lot → **au bit** exigé : traits et niveaux de la tour sur flux annexe = tour dans `_admit` (torch.equal, 12 images distinctes,
script `identite-276g.py`, 12/12 ou refusé) ; b = 1 : 24 jetons = main 5/5. b = 4/12 : jetons relevés, inapplicables comme
critère (276 f : main non reproductible à b = 12 par la composition des lots) — publiés, pas jugés. Au bit tenu → **DÉFAUT 1**
(`ACVRAM_TOUR_PREPARATION`, 0 = opt-out, régime déclaré) ; sinon opt-in.

## Prédiction et seuils (échelle M G G M, arbre contrôlé `ACVRAM_ARBRE`, charge hôte par cellule, serveur neuf par bras)
Qwen3-VL-2B, une image 448×448, b = 12 × 7 tours, solo × 5. Référence 276 f pour M : TTFT moyen ≈ 220 ms, mur ≈ 291, `_admit` méd 45.
* **TTFT moyen G ≤ 0,85 × M** (prédit ≈ 165-185 : les vagues restent, chaque admission perd sa tour) ; **mur G ≤ 0,85 × M**
  (prédit ≈ 200-240 : les ≈ 100 ms de tour sortent du chemin critique, moins ce que le flux annexe vole au pas).
* `_admit` médian (≥ 2 requêtes) : 45 → **≤ 10 ms**.
* Solo b = 1 : G = M ± 1 ms (même tour, en série, juste déplacée).
* Si l'hypothèse est fausse (la tour n'est pas sur le chemin critique, ou le flux annexe sérialise avec le pas) : G = M ± 3 %
  sur les deux — le contrôle rend « faux ».
* **Issue nommée** (celle qui me gênerait) : le flux annexe ralentit le pas en cours (contention SM) → mur meilleur mais TTFT
  moyen ≥ M : gain de mur seul, refusé comme défaut (opt-in), cause à nommer.
