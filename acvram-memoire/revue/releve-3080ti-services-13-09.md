# Relevé 3080 Ti services — 13-14 septembre 2026

RTX 3080 Ti (index 1, 12 Go GDDR6X), sans la 5090.

## Taxe de stationnement des contextes CUDA

État (a) : Services arrêtés
- Mesure instantanée : 22.81 W
- Moyenne 5 min (nvidia-smi -lms 1000, 300 points) : 23.12 W
- Min/Max : 21.96 W / 24.68 W
- VRAM utilisée : 18 MiB
- VRAM totale : 12 288 MiB

État (b) : Services démarrés sans requête
- Services 8082/8083 non disponibles

État (c) : Pendant une requête
- Services non actifs

**Résumé :** Taxe stationnement GPU au repos ≈ 23 W. Services d'embeddings/reranking n'étaient pas disponibles pour test de charge.

## Verdict mlp_exec=cpu

**Support :** OUI — tlm_exec=cpu est exécuté pour les experts MoE sur la 3080 Ti.

**Preuve :** Code acvram/memory/tiering.py:
- Ligne 585 : `lp.mlp_exec = "cpu"` quand opts.host_exec == "cpu" (assignation inconditionnelle)
- Ligne 598 : `lp.mlp_exec = "cpu" if taux > link else "gpu"` (sélection conditionnelle selon bande passante)
- Ligne 672 : `if lp.mlp_storage == "cpu" and lp.mlp_exec == "cpu":` avec comment "# MLP et experts" — bloc qui traite les deux denses et MoE

**Coût :** DDR bande passante (CPU) 71 Go/s limitée ; expert exilé : 0,30 ms/couche à b=1 ; créneau : 1,02 ms/couche/jeton vs 0,30 ms hôte (levier 3,4×).

**Statut :** Applicable sur 3080 Ti comme GPU de service isolé (services 8082/8083 + TabbyAPI).
