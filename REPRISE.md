# Reprendre ce projet — note de passation

> Soutenir : buymeacoffee.com/anticitoyen


Ce fichier existe pour qu'une nouvelle session Claude, sur n'importe quelle
machine, puisse reprendre le travail sans rien deviner. Il se lit en premier.

Toute la prose du dépôt est en **français** : commentaires, docstrings,
documentation, aide en ligne de commande, messages d'erreur. Les *identifiants*
restent en anglais lorsqu'ils constituent un contrat externe — champs de l'API
OpenAI (`choices`, `finish_reason`), clés des configurations Hugging Face
(`num_hidden_layers`), méthodes PyTorch (`forward`, `state_dict`), noms des
tenseurs dans les safetensors. Les traduire romprait l'interopérabilité.

---

## 1. Ce qu'est le projet, en un paragraphe

`acvram` est un serveur d'inférence pour grands modèles de langage, compatible
avec l'API OpenAI, écrit intégralement dans ce dépôt (moteur compris, pas un
enrobage de llama.cpp ou de vLLM). Il repose sur deux idées : donner à chaque
GPU le format de quantification que son silicium sait lire (NVFP4 sur Blackwell,
INT4 sur Ampere) plutôt que d'aligner les deux sur un dénominateur commun ; et
traiter la mémoire comme une hiérarchie VRAM → VRAM → RAM dont un planificateur
mesure le coût, au lieu de vérifier simplement que le modèle « tient ».

Il ne reprend **rien** de `github.com/bogdanovby/emuv`, le projet dont la
demande initiale partait. Ce dernier était décoratif : table PCI vide, `probe()`
jamais appelé, fonctions de lecture de VRAM non câblées, et un `/dev/emuv` qui
renvoyait en texte les nombres passés en paramètres du module. Aucune ligne n'en
a été conservée.

## 2. État actuel (19/09/2026, 0.6.15)

* Version **0.6.15** (`acvram/__init__.py`) : **régime livré = P2** pour les convertis `attn_int8: canal`
  (`ACVRAM_PREFILL_INT8=cublas` par défaut, 0ccd17c ; six lignes tenues, `revue/poste7-p2-au-defaut-19-09`),
  GUI en 32 langues, `.deb` avec `cuda-toolkit[nvcc,cccl]==13.0.*` — l'épingle est obligatoire :
  nvcc 13.4 contre le runtime 13.0 de torch cu130 rend `cuda_toolkit.h:41 #error` (19/09, doctor)
  (sans cccl, le nvcc des roues pip n'a pas `nv/target` : doctor rendait « repli noyaux
  de référence »). ~1 500 tests dont ~150 sur processeur ; les noyaux, les graphes CUDA et
  toute mesure exigent la machine cible (RTX 5090 bridée à 400 W, RTX 3080 Ti à 275 W —
  tous les chiffres ci-dessous sont mesurés dans cet état).
* **Régime livré par défaut** (`regime.py`, `regime_ligne()` imprimée par chaque
  instrument) : prefill `bf16` + GEMM groupée Marlin (`PREFILL_GROUPED=marlin`), décodage
  GEMV lisant la disposition Marlin (`GEMV_LAYOUT=marlin`, disposition unique, 0 couche
  exilée sur Coder-30B), `chemin_moe=mma`, attention paginée Triton, graphes CUDA ;
  contrôlé sans variable le 19/09 (`verdict-controle-p1-defaut-19-09`).
* **Coder-30B-A3B, harnais égal contre llama.cpp Q4_K_M** (`comparatif-cinq-moteurs-17-09`) :
  b=1 363,3 t/s · 0,798 J (llama.cpp 340,1 · 1,151) ; b=12 1 361 t/s · 0,2265 J net
  (1 066 · 0,2136) ; prefill 16 426 j/s (15 717) ; PPL privée 1,0155 géo (classé ≤ 1,02).
  Devant en vitesse partout ; derrière de 6 % en J net à b=12 au régime libre — le **mode
  éco `-lgc 2700`** (E1, 19/09) rend 1 339 t/s · 0,2071 J net : devant en vitesse ET en
  énergie à b=12 ; `-lgc 2100` : 1 138 · 0,1739 (−19 % de J).
* **P2 au défaut** (projections q/k/v/o int8 par canal, cublas `_int_mm` au prefill,
  vue g128 `etroit_triton` à 2 ≤ M ≤ 16 — C11, c17ea89) : PPL **1,0094 géo**, prefill 18 850 j/s
  (+14 %), b=1 396,0, b=12 1 365 t/s · 0,2243 J net, J prefill 0,89-0,93 ×, équivalence tenue
  (`poste7-p2-au-defaut-19-09`). Un poids inéligible garde la déquant bf16 (jamais W8A8 en
  silence). Le classé Coder lisait **déjà** q/k/v/o en int8 g128 : l'i8c ne change que l'échelle. Split-K b=1 : opt-in `ACVRAM_GEMV_SPLITK=1` (PPL +0,0042, non tranché).
* **Spéculation n-gram déjà au défaut à b ≤ 2** (`runner.py:429`, `ACVRAM_SPECULATION_LOT_MAX=2`,
  `GardeSpeculation` conditionnée au lot réel) : taux d'acceptation 1,61 mesuré le 13/09 sur du
  code ; la cellule b=1 ci-dessus le contient. Invariant : jamais un jeton différent du greedy.
* GLM-4.7-Flash (MLA) : classé 1,0143 (`-k48-calibA`), prefill 5 502 j/s ; b=12 en cours
  (G1). W4A4 experts **fermé** (deux verdicts). Modèles convertis sous
  `/mnt/2TO_2023_980PRO/Modeles/models_acvram/` (161 alias dans le catalogue), originaux
  sur `/mnt/4TO_SATACMR_2022/Modeles/`.
* Équipe : poste7 (décide, `revue/poste7-*.md`), chef (fusions, catalogue, lien utilisateur),
  poste1 (noyaux/code), poste2 (mesure/PPL) ; la carte est une file unique, le verrou
  `outils/carte.sh` est la seule vérité (REGLES § 2). Dépôt :
  `https://outils.nuages.noho.st/gitlab/anticitoyen/anticitoyen-vram` (privé).

## 2b. État à la 0.7.0 (26/09/2026)

Détail complet et chiffré : `docs/notes/v0.7.0.md`. Ici, ce qui change pour qui reprend le
code — chaque ligne cite sa pièce, aucun chiffre sans source.

* **Défauts changés** :
  * `ACVRAM_MARLIN_PAR_LIGNE=0`, à nouveau (pièce 209/226 l'avait mis à 1, décision revenue
    en pièce 232b, poste6, sur le 3-bras de la pièce 229, poste3 c40a952b1 : B/C −15,3 %
    TENU en charge soutenue, contredit la salve unique de la 226 — la pièce 209 seule porte
    la régression). `=1` reste disponible en opt-in.
  * `ACVRAM_INT8_GEMV_MAX_PARTAGE=16` (pièce 243, poste5 cdf709c80) : +9,70 % t/s mixte
    b=8, −9,3 % J/jeton, KL tenue (`revue/poste5-piece243-verdict-26-09.md`).
  * `ACVRAM_ETROIT_CANAL=1` (pièce 195b, décision déléguée par l'utilisateur, HORS BIT
    « ± 1 ulp » — ordre des sommes fp32) : +4,01 % débit, −3,76 % J/jeton à b=8.
  * `ACVRAM_GDN_AB=auto` (pièces 175/175b, poste6) : portes α/β GDN en un seul appel,
    exact au bit contre les deux `F.linear` séparés.
  * `TRANCHE_COPIE_MIN` (pièce 201) : capacité KV désormais correcte pour les modèles
    vision/MTP (Qwen3.8-nvfp4 −7,4 %, gemma-4-31B-vision −21,4 % de capacité annoncée,
    plus juste, pas moins).
  * `_KV_MARGE_MIN_GDN = 3072 Mio` au lieu de 1536 (pièce 212, `acvram/engine/loader.py:1201`) :
    marge doublée pour les architectures GDN/KDA/mamba2, ≈ −12 192 jetons de capacité KV
    annoncée sur les Qwen3.8 concernés, plus de risque d'OOM silencieux à la capture de
    graphes.
* **RoPE à dtype fixe** (pièce 213b, poste5 d57159556 + dd28d98dd) : le 1er lot de décodage
  d'un processus divergeait des suivants — cache RoPE cos/sin (`layers.py:674/721/863`)
  construit en fp32 non arrondi au 1er préfill, en bf16 ensuite. Correctif : dtype fixe du
  module + `loader.py:313`. La calibration portait la même dérive (`quant/collect.py:257`),
  corrigée par la pièce 235 (poste2 bbad9e189).
* **Paquets** : AUR (`packaging/aur/`, dépend de `python-pytorch-cuda`) et RPM/COPR
  (`packaging/rpm/`, pas de torch CUDA empaqueté sous Fedora → venv privé bootstrapé au
  premier lancement) livrés par la pièce 236a (poste3 749ef7e46) ; Flathub
  (`packaging/flathub/`, embarque `org.freedesktop.Platform.GL.nvidia`, ≈ 1,9 Go, noyaux
  PRÉCOMPILÉS obligatoires — pas de nvcc dans le bac à sable) par la pièce 236b (poste6
  e19ca029c), job CI complet par la pièce 241 (poste6 772c8bf61) ; noyaux précompilés
  eux-mêmes par la pièce 240 (poste6 a7302f0a2, `_precompile_utilisable`,
  `ACVRAM_KERNELS_PRECOMPILES`) ; Weblate sert les chaînes d'interface (le README reste
  hors Weblate, pièce 236c, poste2 fae740994) ; `.deb` inchangé dans son mécanisme (venv
  bootstrapé, torch choisi selon les GPU présents). `SECURITY.md` ajouté (main 29a8ff944).

## 3. Matériel cible

i9-14900K · ASUS ROG Maximus Z790 Dark Hero · 96 Go DDR5 · ASUS RTX 5090 Astral
LC OC 32 Go (`sm_120`) · ASUS RTX 3080 Ti 12 Go (`sm_86`) · Linux Mint 22.3.

La machine devait être disponible à partir du **31 août 2026 au soir**. Si vous
lisez ceci sur elle, la première chose à faire est la séquence de la section 7.

## 4. Installation sur une machine neuve

```bash
git clone https://outils.nuages.noho.st/gitlab/anticitoyen/anticitoyen-vram.git
cd anticitoyen-vram
./install.sh          # choisit la roue torch adaptée aux GPU présents
acvram doctor         # dit ce qui s'est compilé et ce qui manque
pytest -q             # doit afficher 67 passed
```

`install.sh` détecte la capacité de calcul des GPU : s'il trouve du Blackwell il
installe torch pour CUDA 12.8, sinon cu124, sinon la version processeur. Sans
GPU, tout fonctionne quand même par le chemin de référence — lent, mais
numériquement identique, et c'est ainsi que ce projet a été développé.

### Accès au dépôt

Le jeton GitLab est chiffré en AES-256 sous une phrase de passe, dans
`~/.config/acvram/gitlab-token.gpg`. Un auxiliaire d'identifiants git le
déchiffre à la demande, et **uniquement** pour l'hôte `outils.nuages.noho.st` en
`https` :

```
~/.config/acvram/git-credential-acvram    l'auxiliaire
~/.config/acvram/chiffrer-jeton.sh        à lancer une fois, pose la phrase de passe
```

Sur une machine neuve, recopiez ces deux fichiers et le `.gpg`, puis :

```bash
git config --local credential.https://outils.nuages.noho.st.helper \
    ~/.config/acvram/git-credential-acvram
git config --local credential.https://outils.nuages.noho.st.username oauth2
```

`gpg-agent` garde la phrase une heure (`~/.gnupg/gpg-agent.conf`), donc en
pratique elle est demandée une fois par session de travail.

**Le jeton ne doit jamais entrer dans le dépôt.** `.gitignore` bannit
`vacancesgitlab`, `*.token`, `*token*` et `.env`. Le jeton actuel porte des
droits très larges (`api`, `manage_runner`, `k8s_proxy`, `ai_features`) alors
qu'un push n'exige que `write_repository` : en créer un restreint reste une
amélioration en attente.

## 5. Organisation du code

```
acvram/
  hardware/detect.py     sonde les GPU ; associe capacité de calcul -> format de poids
  hardware/profiles.py   rigs déclarés, pour planifier sans la machine
  quant/nvfp4.py         codec E2M1 + E4M3 par 16          (5090)
  quant/int4.py          uint4 + échelle/zéro fp16 par 128  (3080 Ti)
  quant/formats.py       registre ; comptabilité des bits par poids
  quant/calibrate.py     mise à l'échelle AWQ, rotation de Hadamard
  quant/collect.py       statistiques d'activation, couche par couche
  quant/convert.py       point de contrôle HF -> fragments acvram + manifeste
  kernels/acvram_kernels.cu   déquantification et GEMV fusionnés, CUDA
  kernels/acvram_cpu.cpp      GEMV 4 bits processeur, AVX2 + scalaire, ABI C
  kernels/cpu.py              le compile et le charge par ctypes
  kernels/fp4_gemm.py         chemin FP4 tensor cores, sondé
  kernels/__init__.py         compilation à la volée, avec repli PyTorch
  memory/tiering.py      le planificateur de placement  <- le fichier intéressant
  memory/kvcache.py      cache KV paginé et quantifié + cache de préfixe
  engine/config.py       ModelSpec depuis config.json
  engine/layers.py       QuantLinear, StreamedWeight, RoPE, RMSNorm, masques
  engine/model.py        attention, MLP, MoE, le modèle assemblé
  engine/loader.py       manifeste -> modèle placé et exécutable
  engine/runner.py       lot continu, ordonnancement
  engine/speculative.py  propositeurs n-grammes et brouillon, acceptation exacte
  engine/sampler.py      température / top-k / top-p / pénalités
  server/protocol.py     schémas de requête et de réponse OpenAI
  server/app.py          FastAPI, diffusion SSE
  evaluate.py            perplexité par fenêtre glissante
  bench.py               mesures : liens PCIe, DDR, noyaux, décodage
  cli.py                 doctor / detect / plan / convert / serve / eval / bench
```

## 6. Invariants à ne pas casser

* **`quant/*.py` est la spécification ; les noyaux doivent s'y conformer.** Les
  implémentations PyTorch de `nvfp4.py` et `int4.py` définissent la numérique.
  `tests/test_quant.py` les fige. Si un noyau et la référence divergent, c'est
  le noyau qui a tort.
* **Tout fonctionne sans CUDA.** `kernels/__init__.py` retombe sur la référence,
  et toute la suite de tests tourne sur processeur. C'est la seule raison pour
  laquelle ce code a pu être écrit avant que le matériel n'existe.
* **Le planificateur ne tronque jamais en silence.** Si un modèle ne tient pas,
  il le dit, avec le nombre de bits par poids qu'il faudrait. Ne « corrigez »
  pas un débordement en laissant tomber des couches.
* **AWQ sans statistiques ne fait rien, et le CLI doit le dire.** La recherche
  sur grille inclut alpha = 0, donc une recherche calibrée ne peut pas faire
  pire que l'arrondi au plus proche — mais une recherche *non* calibrée ne fait
  rien du tout.
* **Une mise à l'échelle s'applique à l'activation, jamais repliée dans le
  poids.** Le repli défait exactement ce pour quoi elle avait été cherchée.
* **Une optimisation ne doit pas changer la sortie.** Le cache de préfixe, le
  décodage spéculatif et l'attention groupée ont chacun un test affirmant qu'ils
  produisent exactement ce que produit le chemin lent. Si vous en ajoutez une,
  ajoutez son test d'équivalence dans le même commit.
* **Jamais `is_causal=True` pour un bloc de requêtes qui commence en cours de
  séquence.** SDPA aligne le triangle en haut à gauche ; utilisez
  `causal_mask(...)` ou passez `q_offset` à `attention(...)`. Un test affirme
  que les deux diffèrent, précisément pour que cela ne soit pas « simplifié ».
* **L'acceptation spéculative est exacte et doit le rester.** La règle est
  `min(1, p/q)` avec rééchantillonnage du résidu. Le raccourci « accepter si ça
  correspond » n'est valide qu'à température nulle.

## 7. Première heure sur la machine cible

```bash
acvram doctor                    # ce qui s'est compilé, ce qui non, et pourquoi
acvram bench --what bandwidth    # les deux liens PCIe et la DDR
acvram bench --what kernels      # noyaux CUDA et processeur face à la référence
pytest -q                        # doit rester vert
acvram plan ~/modeles/Qwen3-32B  # comparer à docs/MATERIEL.md
```

Attendez-vous à corriger des erreurs de compilation dans `acvram_kernels.cu` :
nvcc ne l'a jamais vu. La partie la plus risquée, le décodage des quartets, a
été validée en la compilant seule sur processeur (`nibble()` dans le fichier),
mais le reste est écrit à l'aveugle.

## 8. Pièges déjà rencontrés, à ne pas re-découvrir

* **Le masque causal de SDPA.** Voir la section 6. Le drapeau intégré aurait
  produit un cache de préfixe silencieusement faux.
* **La fixture de test tirait des poids aléatoires sans graine.** La suite
  passait dans l'arbre de travail et échouait sur une extraction propre. Elle
  est désormais initialisée à `torch.manual_seed(20260830)`.
* **`tty` écrit « pas un tty » sur sa sortie standard** quand elle échoue, si
  bien qu'un `GPG_TTY=$(tty 2>/dev/null)` naïf affecte cette phrase et casse la
  saisie de la phrase de passe. L'auxiliaire d'identifiants vérifie que le
  résultat est un périphérique caractère.
* **AWQ jugé sur la mauvaise métrique.** AWQ dégrade volontairement l'erreur de
  reconstruction des *poids* pour améliorer l'erreur en *sortie de couche*. Le
  juger sur la première le rejette systématiquement.
* **Le FP8 pour le cache KV est moins bon que l'INT8** à taille égale, dès lors
  qu'on met une échelle par (jeton, tête) : 32 dB contre 44. L'avantage du FP8
  est la plage dynamique, que l'échelle fournit déjà.
* **`torch.utils.cpp_extension` exige `ninja` et `Python.h`.** Le noyau
  processeur a donc été bâti en ABI C pure et chargé par ctypes, ce qui
  supprime ces deux dépendances sur la machine de déploiement.

## 9. Décisions prises et pourquoi

* **Un format par GPU** plutôt qu'un format commun : la 5090 a des tensor cores
  FP4, la 3080 Ti n'a ni FP4 ni FP8. Aligner les deux gâcherait la première.
* **Cache KV en INT8 partout**, y compris sur la 5090 qui saurait faire du FP8 —
  mesuré meilleur (section 8).
* **Hadamard seulement pour l'INT4** : les blocs de 16 du NVFP4 portent déjà
  leur échelle, ceux de 128 de l'INT4 non.
* **Le planificateur laisse volontairement la 3080 Ti oisive** quand le modèle
  tient sur la 5090 : les tranches d'un pipeline s'exécutent en série, ajouter
  une étape plus lente ralentit le décodage mono-flux. Mesuré : 93 → 77 jetons/s
  sur Qwen3-32B. D'où l'idée d'y mettre plutôt un modèle brouillon.
* **Étage hôte calculé sur processeur** plutôt que streamé : la DDR5 est plus
  large que le PCIe, et cela libère le GPU.
* **Dépôt privé** par défaut. Le projet ne contient aucun secret, mais la
  visibilité est une décision qui appartient au propriétaire.

## 9b. Règles d'outillage nées le 26/09 (0.7.0)

Toutes trouvées en marge d'une pièce de fond, jamais cherchées pour elles-mêmes —
chacune ferme une classe d'incident réellement observée ce jour-là.

* **`outils/carte.sh` tue tout le groupe de processus de la commande** (pièce 244, poste3
  5eb5e05a7, `tests/test_carte_groupe_signal_244.py`) : un TERM/INT/HUP reçu par `carte.sh`
  lui-même (un `kill` externe) tuait le script sans jamais toucher la commande, encore
  vivante hors verrou — 3e contamination en trois jours. `setsid` + signal au groupe avant
  de rendre le verrou ; refuse la prise suivante si un `.qui` pointe un groupe encore vivant
  (cas `KILL -9`, qu'aucun trap ne peut rattraper).
* **`ACVRAM_TESTS_PENDANT_MESURE` ne dispense plus seule du contrôle de verrou-mesure**
  (pièce 246, poste3 a3b11cfc5, `tests/test_conftest_verrou_mesure.py`) : seul un pytest qui
  tourne SOUS la prise `carte.sh` en question (`ACVRAM_CARTE_TENUE` = pid du `.qui`) passe ;
  sinon REFUS nommant la prise, même avec la variable — 3e contournement de la garde en
  trois jours, dont un par la même session.
* **Garde contre les invites répétées dans une cellule MoE** (pièce 256/256b, poste3
  62eb805b4/12ad66e8c, `tests/test_banc_repetition_256.py`, REGLES § 4) : un texte réel mais
  RÉPÉTÉ (remplissage de longueur par tuilage) déplace le coût du GEMM experts de
  ± 0,6-0,9 ms/pas SELON LE SENS (pièce 237, poste1) — aussi faux qu'une invite à jetons
  tirés (pièce 227). Seuil de trigrammes répétés mesuré (0,5, sur le tokenizer réel), pas
  supposé, dans `banc-llamacpp-16-09.py` et `banc-chat-openai.py`.
* **Cliquet d'index des notes de revue** (pièce 263/263b, poste3 7fbc6b7df/189bd7190,
  `tests/test_index_revue_263.py`) : `acvram-memoire/revue/INDEX.md` ne référençait aucune
  note par une garde exécutée — deux notes (231, 244) en étaient sorties sans que rien ne le
  voie. Le compte de notes non indexées ne doit plus JAMAIS augmenter ; toute note neuve
  s'indexe dans le même commit qui l'ajoute.
* **La relance d'une session reprend le worktree le plus récemment commité, pas le premier
  de `git worktree list`** (pièce 252, poste3 15cc02520, `outils/session-acvram-worktree.sh`) :
  plusieurs worktrees d'un même prénom existent presque toujours ; `git worktree list | grep
  -i $p` rendait la première ligne dans l'ordre d'AJOUT du worktree, jamais celui de
  fraîcheur du travail — poste2 et poste4 ont repris sur un vieux worktree au redémarrage du
  26/09 à cause de ce piège. Classe les branches `origin/<prénom>[-*]` par
  `git for-each-ref --sort=-committerdate`, rend le worktree de la première qui en a un ;
  purge `CLAUDE*`/`ANTHROPIC*`/`ACVRAM_SESSION` hérités avant de lancer `claude` (une
  relance depuis une session désactivait les transcripts). **Appliquée le 26/09** à
  `~/.local/bin/session-acvram` (chef, sauvegarde `.avant-26-09`).
* **Une version sort par `outils/sortir-version.sh vX.Y.Z`, jamais à la main** (pièce
  281/282, poste3 1bd8f719f, `tests/test_sortir_version_281.py`) : codifie la sortie que
  chef faisait manuellement — tag annoté + push GitLab, `outils/publier-github.sh
  --pousser` sous `outils/carte.sh`, `gh release create` (notes lues depuis
  `docs/notes/release-vX.Y.Z-github.md`, qui doit exister AVANT d'appeler le script),
  attente du run GitHub Actions « Release packages » (`gh run watch`, espacé — jamais une
  boucle serrée), puis `outils/verifier-release.sh`. `--simule` trace chaque commande sans
  rien pousser ni créer de tag ; le jeton GitHub est lu comme dans `publier-github.sh`
  (coffre, entrée `github-anticitoyun`), jamais affiché ni journalisé. La vérification de
  propreté ignore les fichiers NON suivis (`git status --porcelain --untracked-files=no`) :
  un `scratchpad/<pièce>/` d'un autre membre, sur `main`, est l'état normal du dépôt et ne
  bloque pas une sortie — seul du contenu SUIVI modifié le fait (282, trouvé en rejouant le
  script `--simule` sur la vraie `main`, qui refusait à tort sur des dossiers scratch d'un
  autre poste). Codes de refus : `64` usage/version malformée, `65` main sale ou version de
  `pyproject.toml` différente, `66` notes GitHub absentes, `67` le tag existe déjà, `68`
  `publier-github.sh` a échoué, `69` `gh release create` ou jeton absent, `70` le run a
  échoué ou n'est jamais apparu, `71` `verifier-release.sh` a signalé un défaut.

## 11. Terminé — définition (20/09/2026, `revue/poste7-tests-rapides-cloture-20-09`, mot pour mot)

État servi au 20/09 09 h 23 : **0.6.30** (`acvram_0.6.30_amd64.deb` à la racine ; C15-prefill au défaut, capture gemma, éco 2 700) ; l'état vivant est `acvram-memoire/revue/ETAT.md`. Le § 2 ci-dessus décrit 0.6.15 (19/09) et reste vrai pour ce qu'il nomme ; les versions 0.6.16-0.6.30 sont dans `revue/INDEX.md` (verdicts `verdict-paquet-*`).

### 11.1 « Terminé » — quatre conditions, chacune rendue par une prise ou un fichier
| | condition | rendu par | état |
|---|---|---|---|
| T1 | comparatif Coder + GLM à harnais égal, `-lgc 2700`, régime d'horloge dans chaque cellule, revendication mot pour mot (devant / derrière par cellule) | INDEX + REPRISE.md | Coder : fait (08 h 43) ; **GLM : cellules vLLM sans régime nommé (858 · 0,397, prefill « libre ») → une prise de 10 min si elles ne sont pas à 2 700, sinon T1 est fait** |
| T2 | toute cellule où acvram est derrière porte SOIT une pièce de cette note avec scellé et prise ≤ 15 min, SOIT « hors périmètre, cause chiffrée » (§ 3) | cette note, § 2 et § 3 | écrit ci-dessous |
| T3 | paquet servi installé (`dpkg -s acvram` = 0.6.30), trois bras éco tenus sur l'arbre livré, **parc S2 : un jeton décodé par alias au godet 1, ok / repli / échec publié** | `verdict-paquet-0630` (fait) + S2 (une prise 15 min) | S2 à faire sur 0.6.30 |
| T4 | `pytest -q` vert à sec sur main ; suite carte du gabarit ≤ 30 min ; aucune branche hors main sans verdict (fusionnée ou close « opt-in ») | poste1 / chef | 3 rouges signalés 08 h 00 → à relire sur main 572fd9ae |
**Terminé = T1-T4 tenus.** Un « faux » publié ferme une pièce autant qu'un « tenu » : le projet se termine avec ses derrières nommés, pas avec un chiffre reconstruit. Prédiction : T1-T4 tenus **demain 21/09 avant midi** si les prises de § 2 tiennent leur durée ; pièce 3 est la seule qui coûte plus d'une heure de code.

### 11.2 Hors périmètre de « terminé » — publié avec sa cause, pas de chantier
* **Coder b=12 énergie** : 0,210 contre vLLM Marlin 0,136 J/jeton — experts Marlin à 400 W = 54 % du pas, W constante sous plafond (MECANISMES), **aucun chemin connu** ; **vitesse** 1 397 contre 1 626 (−14 %) : sélection C15-3d faux (4,83 µs), reste la bande Marlin 1,07 contre 1,24-1,41 To/s = noyau à réécrire, ≥ 3 jours, prédiction × 1,10-1,15 au mieux, ne rattrape pas 0,136 J.
* **GLM prefill** 7 268 contre 18 117 (× 2,5) : structure du flash `tl.dot` sur sm_120 (tuile 32×64, × 26 forme 1) ; C13-c réécrit ≈ 10 000 prédit = encore × 1,8 derrière, ≥ 2 jours. **GLM b=12** 660 contre 858 : C14-b (M1) et niveau 3 (M3-M4) sont les seules pièces courtes ; l'écart restant (≈ −20 %) est publié tel quel.
* **119B** : (c) tenu tant que l'utilisateur n'a pas répondu a/b/c/d ; parité PCIe × 8 au mieux (22,6 Go/s).
* Ce qui reste opt-in nommé (C4, C10 b, C5-b, C17, C13-c flash, `MLA_GLUE=2` si M3 tombe, `MLA_BATCH_FUSION`) est listé dans REPRISE.md avec son chiffre — c'est une fermeture, pas une dette.

## 10. Ce qui vient ensuite (19/09/2026 — `revue/poste7-pistes-evolutions-19-09`)

**La borne à connaître avant toute piste de prefill** (`poste7-nuit-sens2-19-09` § 1, corrigée par
`poste7-poursuite-chantiers-19-09` § 0) : Coder-30B fait ≈ 12,4 TFLOP par pas de prefill de 2 047
jetons ; la RTX 5090 rend 209,5 TFLOPS bf16 denses (acc. fp32) → plancher 59 ms = 34 700 j/s,
et le défaut mesure 124,6 ms = 16 426 : **47 % du plancher** (une première version disait 95 %
avec 105 TFLOPS : faux d'un facteur 2). Conséquence : **il reste ~2× à prendre au prefill sans
quantifier les activations** — le poste est le noyau Marlin lui-même (déquantification refaite
par tuile de M, conçu pour M petit), pas le débit des tensor cores. Correctif du soir (`poste7-c2-c4-c5-tranches-19-09`) : la déquant transitoire vers un tampon bf16
en DRAM coûte 50,8 ms/prefill et tue C2 comme produit — **C2 = infrastructure de C1, C1 (W4A8)
seul chantier prefill**, scellé T_experts ≤ 0,55 × Marlin au budget nsys (`poste7-c1-budget-19-09`).
**P2 et C11 sont clos : au défaut 0.6.15** (§ 2).

**Chantiers ouverts le 19/09 au soir** (utilisateur : « les chantiers non terminables démarrent
maintenant »), un fichier `revue/chantier-c<N>-19-09.md` chacun, pointés dans INDEX : C2 prefill
par déquant transitoire + `_grouped_mm` · C1 W4A8 experts · C3 MTP GLM · C4 godets sur `b` ·
C5 KV int8 · C6 conversion GPTQ + Hadamard · C7 GLM MLA FP8 · C8 gouverneur d'horloge par lot.

Par ordre de valeur, chacune avec la mesure qui la rendrait fausse :

1. **Mode éco `-lgc`** — tenu à b=12 (`verdict-eco-lgc-b12-19-09` : 2700 = −10 % J à
   débit égal, 2100 = −25 % J pour −15 % t/s) ; reste E1-bis (b=1, genou), la commande
   `acvram eco {2700|2100|off}`, puis un gouverneur d'horloge par lot (libre b ≤ 2,
   2 700 b 3-7, 2 100 b ≥ 8). Faux si b=1 à 2700 < 340,1 t/s.
2. **Prefill W4A8 experts** (MMA int8/FP8 avec échelle E4M3 par bloc 16) — 18 850 →
   26 000-30 000 j/s attendus, J prefill −30 % ; porte à sec d'abord (`ACVRAM_PREFILL_A8`,
   `verdict-porte-a8-19-09`, faux si PPL fausse-quant − 1,0155 > 0,004), noyau 3-5 jours.
   Le seul ×2 du prefill.
3. **Décodage, instructions par octet** — nos GEMV font 1,3-2,6 instr/octet DRAM contre
   0,18 pour un GEMM vLLM et saturent seuls 400 W : projections q/k/v/o à b ≥ 8 par
   `narrow_gemm`/MMA, après ncu M1/M2 (`verdict-ncu-m1/m2-19-09`). −10 à −20 % J à b=12.
4. **Spéculation exacte** — MTP de GLM-4.7-Flash (tête livrée), n-gram code (taux 1,61
   mesuré) : b=1 +30 à +60 % t/s sur code ; invariant : jamais un jeton différent du greedy.
5. **GLM : C7 (MLA en FP8) fermé comme levier** — les projections MLA des convertis sont déjà
   int8 g128/i8c et FP8 ×6 l'erreur ; le poste est le **cœur d'attention en fp32 sans TF32**
   (8,6 TFLOP/pas ≥ 82 ms/372) → **C13** : C13-a TF32 scoped, C13-b noyau bf16 après nsys GLM
   (`poste7-c7-clos-c13-attention-glm-19-09`). 6. **Godets sur `b`** : faits depuis le 11/09 ;
   C4 a corrigé le vrai défaut (`static_bind` store[-1], Mamba2).
7. **Cache d'experts** (modèles > VRAM : Devstral, 119B — suspendu utilisateur).
8. **Conversion : GPTQ + Hadamard sur Coder** (1,0155 → 1,010, de la marge pour A8).
9. **Cache KV int8** : déjà le défaut effectif (`kvcache.py:241`) — C5 ne fait que le nommer (`regime_ligne()`). 10. **Produit** : `.deb`, lanceurs
   refusant sans verrou, `acvram eco`, GUI (32 langues, vedettes), PPL sur le chemin servi.

**Ce qui ne se fera pas** (pour ne pas y revenir) : W4A4 experts (plancher E2M1 ≈ 9 %
d'erreur par GEMM, PPL +0,010 contre 0,0045 de marge) ; horloge mémoire, split-K b=1 au défaut, lm_head
int8 à b=12, exil par expert à b=12 — tous réfutés par mesure, verdicts dans INDEX.
