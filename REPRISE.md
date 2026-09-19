# Reprendre ce projet — note de passation

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

## 2. État actuel (19/09/2026, 0.6.14)

* Version **0.6.14** (`acvram/__init__.py`), `.deb` reconstruit avec `cuda-toolkit[nvcc,cccl]`
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
* **P2** (projections q/k/v/o int8 par canal, `torch._int_mm`) : **opt-in
  `ACVRAM_PREFILL_INT8=cublas`**, prefill 18 850 j/s (+14 %), J/jeton 0,89-0,93 × défaut,
  équivalence tenue ; PPL ligne 2 en cours (instrument `perplexity()` à aligner sur le
  chemin servi). Split-K b=1 : opt-in `ACVRAM_GEMV_SPLITK=1` (PPL +0,0042, non tranché).
* GLM-4.7-Flash (MLA) : classé 1,0143 (`-k48-calibA`), prefill 5 502 j/s ; b=12 en cours
  (G1). W4A4 experts **fermé** (deux verdicts). Modèles convertis sous
  `/mnt/2TO_2023_980PRO/Modeles/models_acvram/` (161 alias dans le catalogue), originaux
  sur `/mnt/4TO_SATACMR_2022/Modeles/`.
* Équipe : poste7 (décide, `revue/poste7-*.md`), chef (fusions, catalogue, lien utilisateur),
  poste1 (noyaux/code), poste2 (mesure/PPL) ; la carte est une file unique, le verrou
  `outils/carte.sh` est la seule vérité (REGLES § 2). Dépôt :
  `https://outils.nuages.noho.st/gitlab/anticitoyen/anticitoyen-vram` (privé).

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

## 10. Ce qui vient ensuite (19/09/2026 — `revue/poste7-pistes-evolutions-19-09`)

**La borne à connaître avant toute piste de prefill** (`poste7-nuit-sens2-19-09` § 1) :
Coder-30B fait ≈ 12,4 TFLOP par pas de prefill de 2 047 jetons ; les tensor cores bf16
denses (~105 TFLOPS) bornent le pas à 118 ms = 17 300 j/s, et le défaut mesure 16 426 :
**95 % du plancher**. Aucune optimisation de lecture des poids ne gagnera plus de 5 % ;
le ×2 exige des experts qui calculent en FP8/FP4 sur les tensor cores, donc des
activations quantifiées — d'où la porte A8 ci-dessous.

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
5. **GLM : MLA en FP8 au prefill** (porte fausse-quant à sec d'abord ; l'int8 par canal a
   réfuté, REGLES § 9). 6. **Godets sur `b`** (prérequis `_bind_hybrid`, MECANISMES).
7. **Cache d'experts** (modèles > VRAM : Devstral, 119B — suspendu utilisateur).
8. **Conversion : GPTQ + Hadamard sur Coder** (1,0155 → 1,010, de la marge pour A8).
9. **Cache KV int8** (×2 séquences à VRAM égale). 10. **Produit** : `.deb`, lanceurs
   refusant sans verrou, `acvram eco`, GUI (32 langues, vedettes), PPL sur le chemin servi.

**Ce qui ne se fera pas** (pour ne pas y revenir) : W4A4 experts (plancher E2M1 ≈ 9 %
d'erreur par GEMM, PPL +0,010 contre 0,0045 de marge) ; « lire les poids une fois au lieu
de deux » au prefill (±0 % trois fois) ; horloge mémoire, split-K b=1 au défaut, lm_head
int8 à b=12, exil par expert à b=12 — tous réfutés par mesure, verdicts dans INDEX.
