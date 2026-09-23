# Pièce 116 — profil nsys couche MoE b=1, verdict scellé : piste morte

- instrument : `nsys profile -t cuda --cuda-graph-trace=node` sur `frontiere-pas.py`
  (b=1, 20 pas demandés → 67 pas capturés), sous `outils/carte.sh` ; classification par
  familles de noyaux (`outils/gpu/mesure/familles-noyaux.py`, régimes déjà en service) via
  `scratchpad/poste3-piece116-nsys-moe-23-09/classer.py` (fenêtres par couche : bornes =
  chaque `_route_fusee_kernel`, 1/couche, pas 1/pas — couches=1 au lieu de couches=48)
- commit : `c597b24f` (worktree `poste3`, main fusionné avant prise, y compris c7w de chef)
- régime : plein (compteur remis à zéro), alias servi `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`,
  mode serve par défaut, graphes actifs, décodage établi (après 32 jetons, chauffe
  `frontiere-pas.py` standard)
- scellé (chef, avant mesure) : > 30 % d'auxiliaires (routeur, top-k, alignement,
  permutation, quantification) → la fusion vaut une pièce ; < 10 % → la piste est morte
- mesuré : couche 24 seule (65 fenêtres médianes) et moyenne des 48 couches MoE du même pas
  (3120 fenêtres, mêmes 65 pas) — quasi identiques, couche 24 REPRÉSENTATIVE :
  | | GEMM experts | auxiliaires | reste |
  |---|---|---|---|
  | couche 24 | 35,9 % (0,0196 ms) | **6,4 %** (0,0035 ms) | 57,7 % (0,0315 ms) |
  | moyenne 48 couches | 36,2 % (0,0196 ms) | **6,5 %** (0,0035 ms) | 57,4 % (0,0311 ms) |
  Détail auxiliaires : `routage` (route+topk fusionnés, 0,0026 ms, 1 lancement) +
  `experts_glue` (reduce/pack, 0,0009 ms, 1 lancement) — pas de `experts_quant_a4` ni
  `routeur_gemm` au pas servi (chemin Marlin/i8c, cohérent avec la table de familles).
  `reste` dominé par `proj_etroites_int8` (0,0176 ms, projections QKVO étroites) et
  attention/normes/rope_kv — capturé dans la fenêtre [route_fusee(i), route_fusee(i+1))
  qui déborde sur l'attention de la couche i+1 (limite connue de la méthode, dite ici,
  pas cachée : cette fenêtre mesure « du routeur au routeur », pas « MoE seul »)
- verdict : **auxiliaires 6,4-6,5 %, sous le seuil de 10 % scellé — PISTE MORTE.** Le
  routage/permutation/alignement de la couche MoE est déjà négligeable au pas servi
  (fusions déjà en place : `route_fusee_kernel`, `route_prep`, chemin groupé Marlin à
  trois lancements) ; aucune fusion supplémentaire du côté auxiliaires ne vaut une pièce.
  Le poids réel est ailleurs : GEMM experts (36 %) et surtout les projections étroites
  int8 + attention/normes/rope (58 %, hors MoE proprement dit, à instruire séparément
  si une piste y est cherchée).
- durée : prévue ~5 min (chargement + chauffe + 20 pas + post-traitement), tenue ~4 min
  (deux essais : le premier a échoué à sec sur un chemin de venv faux dans
  `profil.sh`, corrigé avant relance — aucune carte perdue, juste un aller pour rien)
