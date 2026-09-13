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
