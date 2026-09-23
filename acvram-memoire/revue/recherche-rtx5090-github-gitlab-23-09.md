# Recherche — projets GitHub/GitLab autour de la RTX 5090 (23/09, à sec)

Demande utilisateur directe (hors ordre chef) : balayer GitHub et GitLab pour tout projet actif autour de la
RTX 5090 / Blackwell consumer (sm_120), et rapporter. Méthode : API GitHub (`search/repositories`, quota 10/min
respecté), API GitLab (peu de résultats exploitables, recherche basique nom/description seulement). Toutes les
entrées sont vérifiables par leur URL `github.com/<org>/<repo>` ; aucune ne vient de duck.ai. Repères d'étoiles et
dates de push relevés le 23/09 vers 20 h 50 (heure système de cette session).

## 1. Bibliothèques de noyaux — concurrents directs de nos noyaux (`acvram/kernels/`)

* **`local-inference-lab/b12x`** (★261, Apache-2.0, Python, poussé aujourd'hui) — **le plus proche de notre
  périmètre**. Bibliothèque CuTe DSL + Triton pour SM120/SM121 (RTX 5090, RTX 6000 Pro, DGX/RTX Spark) : GEMM
  bloc-échelonné NVFP4/MXFP4/MXFP8/block-FP8, attention paginée FP8, **MLA compressée avec les MÊMES formats
  528 o (SWA) / 288 o (principal) que nous cherchons pour K8V4** (pièce 103-104), MoE fusé FP4 (décodage +
  chemin dynamique persistant + EP), **fusion de rétroaction MTP** (`sequence.mtp_feedback`, directement
  pertinent pour la pièce 106), GDN prefill/decode et KDA prefill (recoupe `acvram/engine/gdn.py`, `kda.py`).
  Le README dit lui-même ne pas viser la production/datacenter (« architecture mismatches, fast-moving pace »)
  et renvoie vers FlashInfer/CUTLASS/TRT-LLM pour ces cas — même positionnement que nous : outil expérimental
  pour cartes consumer. **À lire en détail avant tout chantier K8V4 ou MTP** : peut soit confirmer une piste,
  soit exposer un piège déjà rencontré par eux.
* **`kacper-daftcode/vLLM-Moet`** (★539, Apache-2.0, langage **Sass** = SASS écrit à la main, poussé aujourd'hui)
  — patch vLLM + noyaux SASS SM120 manuels : experts MoE en 2 bits + un cache "delta" FP4 qui restaure la
  qualité du checkpoint (NV)FP4 officiel sur cartes Blackwell consumer. Directement comparable à notre AWQ par
  expert (pièce awq-pile) mais en 2 bits au lieu de 4.
* **`pjordanandrsn/grouped-nf4-gemm`** (★2, MIT, Python+Triton) — noyaux Triton pour MoE 4 bits : GEMM groupé
  NF4/MXFP4, GEMV INT4, attention paginée FP8, et **streaming d'experts CPU/NVMe** (recoupe notre chantier
  d'exil d'experts). Petit projet mais topics propres et à jour.
* **`Antoniomv7/gb300-microbench`** (★2, BSD-3, Python) — microbenchmarks Blackwell : LDGSTS vs TMA, débit et
  scaling UMMA, CuTe DSL vs cuBLASLt, GEMM BF16/FP8/NVFP4. Vise GB300/B300 (datacenter) mais méthodologie
  directement transposable à nos propres bancs `nvcc -Xptxas -v` / occupation (REGLES § 3).
* **`flashinfer-ai/flashinfer`** (★6496, référence établie) — apparaît systématiquement dans les recherches
  Blackwell/kernel ; c'est la bibliothèque dont `unified_attention` de vLLM s'inspire (cf. pièce 93/Q18, notre
  comparatif 9,5 µs contre 5,9 µs). Pas nouveau mais à surveiller pour tout changement de kernel de décodage.

## 2. Moteurs de service complets ciblant la RTX 5090

* **`JustVugg/colibri`** (★37 319, licence non relevée, C pur, créé le 01/07/2026 — **croissance virale en moins
  de 3 mois**) — « Run frontier MoE models on hardware you already own — pure C, zero deps, experts streamed from
  disk ». Pas spécifiquement 5090 dans la description mais ressort sur la requête llama.cpp+Blackwell+5090 ;
  vu son ampleur (37k★), à vérifier séparément s'il cible sm_120 nommément — poids d'écosystème trop gros pour
  l'ignorer.
* **`kekzl/imp`** (★41, MIT, CUDA, topics `sm120`/`rtx-5090`) — serveur LLM pour une seule RTX 5090, orienté
  agents (appels d'outils, longues conversations), revendique être « l'un des moteurs les plus rapides sur cette
  carte » à b=1 et à plusieurs dizaines de flux concurrents, chiffres dans le dépôt.
* **`nibor1896/crow-nest`** (★3, Apache-2.0, Rust) — moteur d'inférence CUDA en Rust, un modèle (Qwen3.8-Flash-Next
  NVFP4), une RTX 5090, noyaux et conteneur maison, API compatible OpenAI avec vision.
* **`local-inference-lab/blackwell-llm-docker`** (★82, Python) — images Docker SGLang + vLLM pour GPU Blackwell
  (SM120, CUDA 13.2) — même auteur que `b12x`.
* **`AviBackToBlack/vllm-windows-native`** (★2, PowerShell, topics `rtx-5090`/`blackwell`) — build et runtime vLLM
  natif Windows pour RTX 5090/SM120, sans WSL ni Docker.
* **`mudler/vllm.cpp`** (★427, C++) — moteur façon vLLM en C++ (continuous batching, KV paginé), ressort sur la
  requête Marlin/Blackwell.

## 3. Quantification et têtes MTP — recoupe directement nos pièces 103/104/106

* **`RobTand/prismaquant`** (★103, Python) — quantification mixte par couche selon la sensibilité, export
  compressed-tensors natif, **validé sur Qwen3.6-35B-A3B MoE avec décodage spéculatif MTP** — même famille de
  modèle que nos essais GLM/Qwen, à comparer à notre approche AWQ par expert.
* **`lucacadalora/sglang-rtx5090-mtp`** (★0, Apache-2.0) — Qwen3.8-27B NVFP4 avec décodage spéculatif MTP sur une
  RTX 5090 sous WSL2 (SGLang). Petit dépôt mais pertinent tel quel pour la pièce 106 (aucune mesure MTP sourcée
  n'existait pour Qwen3.5/3.6 selon duck.ai — celui-ci pourrait combler le trou, à vérifier s'il publie un
  taux d'acceptation).
* **`Yisau7070/llama-cpp-mtp-turboquant-sm120-blackwell-windows`** (★1, topics `mtp`/`sm-120`/`rtx-5090`) —
  llama.cpp + MTP + TurboQuant sous Windows en sm_120 natif.

## 4. Mesure et efficacité énergétique

* **`SRSWTI/Silicon-Joules`** (★0, Python) — mesurer le travail d'inférence utile par accélérateur, dollar et
  joule — même angle que notre `energie.py`/J-par-jeton.
* **`hongping-zh/ecocompute-dynamic-eval`** (★2, TypeScript) — comparatif Précision×Coût×Carbone ; revendique
  que « les benchmarks RTX 5090 montrent que la quantification 4 bits gaspille de l'énergie sur les petits
  modèles » — **affirmation à vérifier avant de la citer**, contraire à l'intuition mais pas à notre propre
  régime (nos gains 4 bits sont mesurés sur un MoE 30B actif ~3B, pas un petit modèle dense).

## 5. Ressource méta

* **`moduvoice/awesome-rtx-5090`** (créé le 18/09/2026, liste organisée) — « Curated RTX 5090 benchmarks, model
  recipes, inference engines, creative AI workflows, and system tools ». Bon point d'entrée pour une veille
  récurrente plutôt qu'une recherche à refaire de zéro à chaque fois.

## GitLab

Recherche nettement moins fructueuse (l'API GitLab basique ne cherche que nom/chemin/description, pas le
texte intégral, et est pauvre pour ce type de requête technique). Rien de comparable trouvé à ce qui précède ;
seuls des projets personnels isolés (bancs d'offres RTX 5090, générateur d'adresses TRON accéléré CUDA) sont
ressortis, sans intérêt pour le projet.

## À faire, si utile

1. Lire `local-inference-lab/b12x` en détail avant la prochaine pièce K8V4 ou MTP — c'est le seul projet qui
   couvre les DEUX chantiers en cours chez nous avec un format de cache très proche du nôtre.
2. Vérifier si `lucacadalora/sglang-rtx5090-mtp` publie un taux d'acceptation MTP mesuré — comblerait le trou
   nommé dans la pièce 106.
3. Vérifier l'affirmation de `hongping-zh/ecocompute-dynamic-eval` sur le coût énergétique du 4 bits avant de
   la citer où que ce soit — REGLES § « un chiffre exact hors de son régime est faux comme décision ».
