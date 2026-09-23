# Sage — 12 papiers arXiv (section C) : deux servent de témoins externes au comparatif, un attend Hadamard, neuf sans objet (16/09)

1. **Ce qu'un papier peut faire pour nous, et rien d'autre** : donner un chiffre **avec son régime** (modèle, quantification, lot, contexte, version du moteur, carte) sur notre matériel, contre lequel notre instrument se vérifie (REGLES §4 bis). Un ratio sans régime (« ×1,6 », « 151 contre 92 ») n'entre nulle part.
2. **Maintenant, à sec, Océane (1 h, ne touche pas la carte)** : lecture de **2605.00519** (Silicon Showdown) et **2601.09527** (Private LLM Inference on Consumer Blackwell), et pour chacun une table : modèle · quant · lot · ctx · moteur + version · carte · t/s · J si publié. Deux usages scellés : (i) **une ligne « témoin externe »** dans la table à cinq pour chaque point qui recoupe un de nos bras (TRT-LLM NVFP4 sur 5090 ; vLLM ou llama.cpp NVFP4/W4A16) ; (ii) **si un point porte sur un modèle qu'on a** (GPT-OSS-20B, Qwen3-8B, Gemma3-27B sont dans l'inventaire ou téléchargeables en < 20 Go), Laure le **reproduit une fois** (10 min, même moteur, même lot, même ctx) : scellé **± 15 %** sur le t/s ; au-delà, l'écart se nomme (version, plafond W, horloge) avant de citer le papier. Le « ×1,46 vLLM » de notre TRT-LLM et le « ×1,6 NVFP4/BF16 » du papier ne se comparent pas : ce ne sont pas les mêmes rapports.
3. **2609.04852 (KVMem)** : après Hadamard et la table, avec le chantier « KV 4 bits par rotation » (`sage-atomic-chat` § 2) — il dimensionne le palier hôte/NVMe, pas le nôtre. Non lu avant.
4. **Les neuf autres** : sans objet (matériel B200/GH200, compilateurs, vidéo, pré-entraînement). FlashGPU-sim (2609.15311) est le seul à garder en signet : un simulateur validé sur 5090 pourrait un jour remplacer une passe de carte pour dimensionner un noyau ; pas maintenant.
5. **Règle de lecture** (celle du carnet du 9/09, inchangée) : résumé ≠ lu ; un chiffre entre dans une note avec sa page et son tableau d'origine.

## Ordre

* **Jérôme** : `ETAT.md` +1 ligne (« témoins externes arXiv — Océane à sec, puis reproduction Laure 10 min si modèle disponible »).
* **Océane** : point 2, tables dans `revue/temoins-arxiv-16-09.md`, ≤ 40 lignes.
* **Laure** : reproduction du point 2 (ii) après ses PPL en cours, seulement si un modèle recoupe.
