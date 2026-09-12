# Index — revue/ (67 documents)

Quatre thèmes : **mesure**, **moteur**, **quantification**, et **protocoles et avis**.
Un document ambivalent est classé par son chiffre principal.

**Contrôle (absents) :**
```bash
for x in $(ls revue/*.md | xargs -n1 basename | grep -v INDEX.md); do grep -q "($x)" revue/INDEX.md || echo "$x"; done
```

---

## Mesure

Benchmarks, protocoles, performance, comparatifs, états du jour.
- [Campagne chrono sync 11-09](campagne-chrono-sync-11-09.md) — ACVRAM_CHRONO_SYNC sur GLM-42B, slots=12 : replay = 98 % du pas, prefill_seconds réhabilitée
- [Comparaison llama.cpp vs acvram](comparaison-llamacpp-vs-acvram-df03399.md) — Cinq mécanismes contre llama.cpp df03399 (poste8)

- [Avis de performance par joule](avis-exterieur-performance-par-joule.md) — Réponse externe à la question de poste2 : peut-on consommer moins de joules par jeton qu'llama.cpp
- [Cadre 2,45 ms inexpliquées](cadre-2450-us-inexpliques.md) — Écrit avant mesure (poste1, 9/09) : prédit 86 % du surcoût en latence de decodage
- [Chronomètres du moteur](chronometres-du-moteur.md) — 26 chronomètres : `time.perf_counter()` autour de CUDA ne mesure que s'il y a une synchronisation
- [Capture échoue à neuf séquences](capture-echoue-a-neuf-sequences.md) — 287 ms : au-delà, les graphes CUDA ne capturent plus sans asserts
- [Contrôle positif grille paged_attn](controle-positif-grille.md) — poste1 10/09 avant mesure : prédit que la grille est sous-parallélisée
- [Courbe du quota deux points](courbe-du-quota-deux-points.md) — 5,4141 PPL (étalon GPTQ) : la courbe n'est pas plate, le coût par PPL **décroît**
- [Courbe du quota six points](courbe-du-quota-six-points.md) — Monotone sans plateau : bits/PPL varie de 9,22 à 30,99 dans l'intérieur de la courbe
- [Décodage faible lot état de l'art](decodage-faible-lot-etat-de-l-art.md) — 66 us en chunk 64 : nos gains +88 % et +104 % en débit surpassent les optimisations connues
- [Grille et couverture gardes](gardes-couverture-reelle.md) — 4 Gio : les gardes du dépôt mesurent le périmètre réel, pas le périmètre déclaré
- [Métriques ncu vocabulaire commun](metriques-ncu.md) — Établi par poste2 le 9/09 : les mesures de trafic DRAM doivent être validées par NCU
- [Objectif cinq concurrents](objectif-cinq-concurrents.md) — Plus rapide ET plus économe que llama.cpp, TabbyAPI, vLLM, YALS, JAX
- [Où nous sommes 10-09](ou-nous-sommes-10-09.md) — 1,82× le débit de llama.cpp, 1,36× jetons/kJ, TTFT 2,29× plus lent, à douze séquences concurrentes
- [Protocole comparatif tiering](protocole-comparatif-tiering.md) — Le chantier qui teste si le tiering économise (40 Go)
- [Protocole comparatif](protocole-comparatif.md) — 122 t/s : benchmark acvram vs cinq concurrents, régime et protocole nommés
- [Protocole verdict 995 Go/s](protocole-verdict-etabli-995gos.md) — Décide ETABLI.md:2476-2481 : 995 Go/s est le plancher accessible
- [Recensement bancs publics 5090](recensement-bancs-publics-5090.md) — Résultat : nous n'avons aucun point de comparaison public pour la RTX 5090
- [Seuil attention 995 Go/s](seuil-part-attention-995-gos.md) — Le protocole du profil par noyau (poste3) doit trancher 26,09 Gio / 28,14 ms
- [État RTX 5090 10-09](ÉTAT-JOUR-10-09-RTX5090.md) — Destiné à chef, poste2, poste1, poste4 : trois natures de faits (mesuré / lu code / lu ailleurs)

---

## Moteur

Optimisations, noyaux, architecture, patches, ordonnancement, fusions.

- [Doublon fusion portée réelle](doublon-de-fusion-et-portee-reelle.md) — MLP.fuse() et Attention.fuse() construisent une pile sans libérer les tenseurs
- [Forme lancement paged_attn_partial](forme-du-lancement-attention.md) — Chantier : trouver la grille optimale pour paged_attn_partial
- [MLA décodage par lot](mla-decodage-par-lot.md) — Le goulot : MLA n'économise de VRAM qu'au prefill, pas au décodage par lot
- [Mécanismes engine dimensions](mecanismes-engine.md) — Recensement : où chaque dimension est traitée, lesquelles ne le sont pas, et pourquoi
- [Ordre sac à dos établi](ordre-du-sac-a-dos-etabli.md) — 9859 Gio : l'ordre de parcours du sac à dos est mal orienté, le défaut est structurel
- [Patch source non épinglée](patch-b-source-non-epinglee.md) — 12 Gio non attestés : proposition de ne plus épingler la source
- [Patch pool hôte épinglé](patch-c-pool-hote-epingle.md) — Pool hôte aligné sur des puissances de 2 : non écrit dans le code
- [Patch double épinglage](patch-double-epinglage.md) — 125 Gio : allouer une arène, la copier à l'avance, libérer l'originale
- [Patch page cache safetensors](patch-page-cache-fadvise.md) — 1 Go : rendre le page cache des safetensors après chargement
- [Patch repli chargement](patch-repli-chargement.md) — Forme exacte du repli au chargement : non écrit dans l'arbre
- [Patch verifier_place pinned](patch-verifier-place-pinned.md) — `verifier_place` est aveugle à la mémoire épinglée : correctif soumis
- [Plomberie pas décodage](plomberie-du-pas-de-decodage.md) — 53 Gio : ce qui ne devrait pas être dans le pas de décodage
- [Prédiction repartition noyaux](prediction-repartition-noyaux.md) — Écrite avant classement : poste4 prédit le classement des 36 486 noyaux
- [Stream-K paged_attn_partial](stream-k-evaluation-papier.md) — Stream-K supprime les blocs vides, pas l'absence de travail : évaluation théorique
- [sm_120 formats graphes](sm120-et-graphes-chez-les-concurrents.md) — vLLM utilise sm_120f, impose bf16, et capture les graphes par batch
- [Comparaison vLLM vs acvram](comparaison-vllm-vs-acvram-ae71862.md) — Cinq mécanismes : vLLM pose un par dimension, acvram en pose plusieurs par cas
- [ggrun lanceur placement MoE](ggrun-lanceur-placement-moe.md) — Fonction de coût, preuve d'allocation, inventaire mesuré : ce qu'un lanceur llama.cpp pose et que notre plan ne pose pas

---

## Quantification

Formats, densité, bits, précision, genre de tenseur, compression.
- [Agrégat à budget fixe](agregat-a-budget-fixe.md) — Aucun agrégat par tenseur ne suit la PPL à budget fixe ; la monotonie du quota gonflait les corrélations
- [base_croissant confondant fermé](base-croissant-confondant-ferme.md) — Bras témoin : trier par SNR de base croissant sans coût ; le confondant « on promeut les mal quantifiés » est fermé
- [Compte égal 149 prédiction](compte-egal-149-prediction.md) — Scellée avant conversion : « B reste le meilleur à compte égal → l'ordre compte »
- [Compte égal 149 résultat](compte-egal-149-resultat.md) — Prédiction réfutée ; le confondant a migré vers les octets
- [Croisement PPL(octets)](croisement-ppl-octets.md) — Deux campagnes croisées contre la courbe du quota : le levier résiduel de l'ordre, mesuré

- [Critère AWQ par groupe](critere-awq-par-groupe.md) — Écrit avant mesure (poste1, 9/09) : AWQ par groupe (fusion q/k/v et gate)
- [Densité int8 trois copies](densite-int8-trois-copies.md) — L'incohérence venait du calcul de la densité, pas du format int8
- [Genre tenseur coût](genre-confondu-avec-le-cout.md) — Le genre du tenseur est indissociable de son coût : 4096× impact sur les noyaux
- [Incohérence bytes_per_tier](incoherence-bytes-per-tier-embed-device.md) — `bytes_per_tier` contredit `device` au manifeste : trouvée en creusant Qwen2.5-Coder-14B
- [Genre coût structurel](le-confondant-genre-cout-est-structurel.md) — Le confondant genre/coût ne se casse pas : c'est un défaut structurel, pas un calcul
- [Queue variation genre tenseur](queue-de-variation-et-genre-de-tenseur.md) — La queue de variation révèle le mécanisme : masque la vraie divergence
- [Question sm_120 formats](question-1-formats-sm120-v2.md) — wgmma/TMA sur RTX 5090 : statut vérifié, priorité haute, wgmma n'existe qu'en Hopper

---

## Protocoles et avis

Protocoles, avis externes, audits, méthodologie, tests d'isolation, recherche.
- [Provenance champs classe 3](provenance-champs-classe-3.md) — Remède aux écritures inventées : Optional seul ne suffit pas, la provenance doit être portée

- [Alpha partage 59 fusions](alpha-partage-recuperer-59-fusions.md) — Les 59 fusions refusées : un scalaire par tenseur d'un groupe (calibrate.py:222)
- [Audit lecture complète c6](audit-lecture-complete-c6.md) — poste8 c6 : les informations recherchées ont-elles été **lues** ou juste **comptées**
- [Avis déterminisme lot](avis-exterieur-determinisme-et-lot.md) — GPT-5.6 via duck.ai : le partage dynamique ne détermine pas le lot final
- [Avis projection finale](avis-exterieur-projection-finale.md) — GPT-5.6 : séparation affirmé / vérifié / à valider sur le coût de projection
- [Avis plancher coalescing](avis-exterieurs-plancher-et-coalescing.md) — Deux questions à duck.ai (Gemma 4, GPT-OSS) sur plancher et coalescing
- [Chantier spéculation](chantier-speculation.md) — Trois motifs : occupation, énergie, amortissement du démarrage
- [Documentation ce qui tranche](documentation-ce-qui-tranche.md) — Un document ne ferme une question que s'il **prédit un nombre** que la mesure aurait pu démentir
- [Protocole découplage fréquences](protocole-decouplage-frequences.md) — Demandé 10/09 : peut-on découpler fréquence cœur et mémoire
- [Protocole ngram mtp](protocole-ngram-mtp.md) — Soumis 8/09 : comparer ngram et mtp pour le test de regressions
- [Protocole témoin MoE](protocole-temoin-moe.md) — 94 Gio : protocole du témoin MoE bf16 soumis à validation des trois
- [Rapport recherche exhaustive](rapport-exhaustif-recherche-20260910.md) — Recherche du 10/09 sur LLM Inference, Quantization, vLLM, llama.cpp, TabbyAPI
- [Revue attn_isole.py](revue-poste2-attn-isole.md) — poste1 10/09 avant mesure : montage recevable, deux clarifications nécessaires
- [Revue valider_campagne.py](revue-poste2-critere-campagne.md) — poste2 9/09 : quelle campagne défectueuse serait acceptée par ce validateur
- [Revue protocole témoin MoE](revue-poste2-temoin-moe.md) — poste2 8/09 : répond à quatre questions sur le protocole du témoin MoE
- [Suite non isolée carte](suite-non-isolee-de-la-carte.md) — Deux tests ont échoué pendant qu'une autre session occupait la 5090
- [Trois avis extérieurs triés](trois-avis-exterieurs-tries.md) — Les trois réponses, triées par ce qui est vérifiable vs conjecturé
- [Synthèse poste1 10-09](SYNTHESE-poste1-10-09.md) — Consolide onze documents : chaque chiffre porte son régime, ce qui n'est pas mesuré est marqué

---

**Total : 58 documents — Mesure 18 | Moteur 16 | Quantification 7 | Protocoles 17**
