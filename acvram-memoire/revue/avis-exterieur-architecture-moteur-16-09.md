# Avis extérieur — architecture d'un moteur C++/CUDA dédié (16/09, apporté par l'utilisateur)

Proposition externe (pas de duck.ai, source utilisateur) pour un moteur bâti pour « une
seule GPU + modèle qui dépasse largement la VRAM » : cœur C++20/CUDA, Rust pour serveur
et sécurité mémoire, Python pour conversion/calibration/bancs. Développement par étapes
mesurables (moteur CPU minimal → quantification CPU → backend CUDA → mémoire hybride →
MoE → production), jamais un modèle géant en premier.

Points qui recoupent des chantiers déjà ouverts ou scellés ici :

1. **Ne pas transférer les poids à chaque jeton, double buffering RAM→VRAM** — recoupe
   `_reajuster_plan` (loader.py) et le chantier RAM hôte du plan
   `golden-wandering-cray.md` (Chantier 3).
2. **Expert Residency Manager (partagés/fréquents en VRAM, rares en RAM, éviction)** —
   au-delà de notre placement figé au manifeste ; à évaluer si utile après le chantier
   MLA (poste7, `poste7-duel-verdict-16-09`).
3. **CUDA Graphs sur la portion statique, code dynamique pour routage MoE/KV/sampling**
   — proche de nos `_forward_grouped` / graphes CUDA existants ; vérifier qu'on suit déjà
   ce découpage plutôt qu'un graphe global.
4. **Quantification différenciée par composant (routeur BF16, attention FP8, experts
   fréquents INT4, rares INT3/INT2, KV FP8/INT8)** — recoupe le plancher SNR et les
   promotions int8 déjà en place ; la variante « plusieurs versions du même expert en
   RAM selon la charge » n'existe pas chez nous.
5. **Référence CPU lente par opérateur, comparée avec tolérance** — déjà notre pratique
   (tests_quant_act_echelle.py, `noyau == référence Python`), à généraliser.
6. **KV cache paginé, quantifié, réutilisation de préfixes** — à comparer à notre cache
   actuel.
7. Sources citées : blog NVIDIA CUDA Graphs, TensorRT-LLM KV cache reuse, doc TensorRT-LLM,
   GitHub llama.cpp (« améliorer le planificateur mémoire d'un moteur existant plutôt que
   réécrire »).

Pertinence à trancher par poste7 : lequel de ces points mérite un chantier, lequel est déjà
couvert, lequel ne s'applique pas à notre cas (nous avons déjà un moteur, pas à en écrire
un depuis zéro).
