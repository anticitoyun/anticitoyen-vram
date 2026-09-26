# Journal des changements

## 0.7.2 (26/09/2026)

* **26/09/2026 — pièce 070 b (poste6) : `acvram doctor` sortait 1 partout, hôte et Flatpak.** `_doctor_eco` (poste7-eco
  19/09) appelait `subprocess.run` sans que `cli.py` importe `subprocess` : `NameError` en toute dernière ligne, après le
  rapport complet — invisible à qui ne lit que les « ok / ECHEC ». Vu par verif-070 (doctor du bac à sable Flatpak, code
  1 avec 0 ECHEC sur l'hôte). `import subprocess` ; `tests/test_doctor_eco_070b.py` (sudo factice, rouge sur l'ancien).
* **26/09/2026 — pièce 273 (poste6, décision chef) : l'extra `vision` (transformers, pillow) est une ALERTE du doctor,
  pas un ÉCHEC, et le Flatpak l'embarque.** Le doctor de la v0.7.0 en Flatpak rendait `ECHEC transformers est absent`,
  `ECHEC PIL est absent` : la liste de `sources-pypi.py` dans `release.yml` ne prenait pas l'extra, et le doctor exigeait
  ce que `pyproject` déclare optionnel (décision du 21/09, trou P3, renversée : l'utilisateur de .deb ou de pip sans
  multimodal ne voit plus un doctor rouge ; un modèle multimodal sans `vision` échoue toujours au chargement, en clair).
  `_doctor_modules()` : requis (safetensors, fastapi, uvicorn, tokenizers, jinja2) en ÉCHEC ; extras (`vision`) en
  `alerte vision indisponible (… absent) : pip install 'acvram[vision]'`, code 0. Flatpak : transformers et pillow dans
  la liste de `sources-pypi.py` (44 roues cp314/abi3/py3 résolues à sec en 18 s, aucune sdist). Gardes :
  `tests/test_doctor_modules_273.py` (imports factices : sans vision → 0 + alerte, avec → ok), `tests/test_flathub_vision_273.py`.

## 0.7.1 (26/09/2026)

* **26/09/2026 — pièce 268 (poste1) : `/metrics` hors de la boucle HTTP, AU DÉFAUT, sortie identique champ par champ.** Il
  était un `async def` qui appelait `engine.regime()` sept fois (cinq à six parcours de l'arbre des modules chacun) : 343 ms
  par appel DANS la boucle — aucune requête lue ni rendue pendant ce temps. Désormais `def` (pool de fils), `regime()` une
  fois par appel, un seul parcours. Sous un lecteur de `/metrics` à 20 Hz, TTFT à 12 requêtes (Qwen3-Coder-30B-A3B
  srcQ4_K_M-nvfp4) **0,78 → 0,24 s** (témoin sans lecteur 0,247 s) ; `/metrics` **343 → 60 ms**. Étape 2 : gabarit et
  tokeniseur de `/v1/chat/completions` dans un fil (au bit, neutre à 12 requêtes). `revue/poste1-268-verdict-26-09.md` ;
  tests cassants `tests/test_metrics_hors_boucle_268.py` (rouges sur l'ancien code).
* **26/09/2026 — pièce 262 (poste1) : le « TTFT 2,29 × plus lent que llama.cpp à 12 séquences » est requalifié.** Mesuré
  sans lecteur `/metrics` concurrent, le TTFT à 12 de la 0.7.0 vaut **0,242 s** contre 0,66 s pour llama.cpp `-np 1` ; le
  duel (`duel-moteurs.py`) interrogeait `/metrics` pendant le tour et mesurait surtout ce défaut (0,80 s).
  `revue/poste1-262-verdict-26-09.md`.
* **Notes sans changement de code** : 269 (poste6, fenêtre d'admission 10/15 ms FAUX, défaut 5 ms inchangé,
  `revue/poste6-piece269-*`) ; 270 (poste1, préfill du Coder décomposé, MoE 55 % au solo, levier au bit ≤ 7 ms, clos,
  `revue/poste1-270-decomposition-26-09.md`).

* **26/09/2026 — pièce 209 (Marlin par ligne, poste2, chaîne 237b-c-d) : qualifiée par tâches (237d, McNemar
  apparié, 4 tâches, n=650) — aucune différence significative, mais borne basse IC95 moyenne 93,4 % < 97 % :
  reste à la demande.** `revue/poste2-piece237d-verdict-26-09.md`.
* **26/09/2026 — pièce 260 (poste5, décision chef) : `ACVRAM_I8C_FP8_PREFILL=cublas` À LA DEMANDE, pas au défaut.** Les int8
  ré-encodés du fp8 (manifeste « origine: fp8 », 233 tenseurs du Qwen3.8-27B-unsloth-mixte-i8c) étaient exclus du chemin W8A8
  int8 du préfill depuis la 139 (copie signée persistante, 10,6 Go) ; la 201 a rendu cette copie transitoire, l'opt-in les y
  remet. **Gain** : préfill par lot 8 × 78 **−26,8 %** (0,4105 → 0,3005 s), mur par lot seulement **−2,4 %**. **Prix** : KL de
  décodage (b=8, 32 pas après le préfill) max **0,215**, 9 × le seuil admis (0,0242) ; argmax **94,1 %** contre 99,6 % au
  témoin ; PPL wiki-gptq 2048 **+1,05 %** (7,0273 → 7,1013). Au mur, le gain ne paie pas la qualité : défaut inchangé (bf16).
  Aide : `acvram serve --help`. `revue/poste5-piece260-{scelle,banc,moteur}-26-09.md` ; tests `tests/test_i8c_copie_260.py`.
  Écartés en chemin (255) : le FP8 natif W8A8 (`_scaled_mm`, CUTLASS sm_120) — plus lent au décodage, et deux fois plus
  d'erreur que l'int8 W8A8 au préfill.
* **26/09/2026 — Paquets (poste6, 259/266 a-j, décision chef) : le Flatpak se livre par un dépôt OSTree signé sur
  `gh-pages/flatpak` et un `.flatpakref` joint à la release, torch et sa fermeture CUDA en extra-data.** Un bundle seul ne
  peut pas porter d'extra-data (« Extra data missing in detached metadata ») et GitHub plafonne un fichier de release à
  2 Gio : le job `flatpak` de `release.yml` construit hors ligne (roues épinglées par `packaging/flathub/sources-pypi.py`
  et `sources-torch.sh`, index PyTorch puis PyPI en repli si la somme diffère — `roue_url.py`), signe le commit
  (`flatpak build-sign`, clé `FLATPAK_GPG_KEY`), publie `flatpak/` et `.nojekyll` seuls sur `gh-pages` sous l'identité
  du bot Actions (266 j, `-c user.name/user.email`, pas de `git config`), joint `acvram-<version>.flatpakref` (`GPGKey`)
  et retire l'ancien bundle. Installation : `flatpak install --user <url du .flatpakref>` (32 README) ; `apply_extra`
  dépaquette torch 2.14.0+cu130 (Python 3.14, ABI `cp314` dans l'empreinte des noyaux précompilés) dans
  `/app/extra/site-packages`. Preuve locale sous carte.sh : import torch, CUDA, cuDNN, cuBLAS dans le bac à sable
  (carnet `acvram-memoire/poste6.md`, 26/09 12 h). Vérification d'une release fichier par fichier : `outils/verifier-release.sh vX.Y.Z
  [--flatpak-installer | --flatpak-doctor]` (259, SHA256SUMS 259b). Gardes : `tests/test_flathub_*`,
  `tests/test_release_*`, `tests/test_verifier_release_259.py`.

* **26/09/2026 — pièce 260x (poste5, décision chef) : copie signée du chemin cublas par un xor, AU DÉFAUT
  (`ACVRAM_I8C_COPIE=xor` ; `int16` = témoin).** q − 128 (uint8 à zéro 128 → int8) en un noyau au lieu de l'aller-retour int16 ;
  la copie est transitoire depuis la 201, donc payée à chaque appel cublas. Copie xor au bit, gain mesuré −1,98 ms (L=512) et
  −2,62 ms (L=2047) sur le TTFT b=1 du Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c (ABBA ×4, médianes B toutes sous A), sous le seuil
  annoncé de 2,5 ms à L=512 : non revendiqué. Prédiction (−3,5 à −5,5 ms) fausse. Contrôles : L=78 −0,6 %, banc chat b=8
  +0,20 %. `revue/poste5-piece260x-{scelle,verdict}-26-09.md` ; test cassant `tests/test_i8c_copie_260x.py` (une seule opération
  aten `bitwise_xor` au défaut ; l'ancienne copie le rend rouge).
## 0.7.0 (26/09/2026)

* **26/09/2026 — pièce 232 b (poste6, décision chef) : `ACVRAM_MARLIN_PAR_LIGNE` revient à 0 par défaut pour la 0.7.0 — la 209 est À LA
  DEMANDE.** Deux protocoles, deux résultats sur le même Qwen3-Coder-30B-A3B-nvfp4 pur à b=8 : la 226 (banc chat 102, invites réelles,
  salve unique de 20 s, 5 + 5) donnait 1 = **+12,8 % / −18,4 % J** ; la 229 (poste3, `banc-llamacpp-16-09.py`, invites réelles, lots
  répétés en débit SOUTENU, 5 passes par bras) donne 1 = **−15,3 % / +25,4 % J** (B 1 547,3 contre C 1 784,0 t/s ; A = be837ca1 contre C :
  +1,9 %, neutre). Tant que l'écart entre les deux protocoles n'est pas expliqué (pièce nsys à venir), la release garde l'ancien
  comportement (piles à sous-normales refusées, naturel + decode_mma) et `ACVRAM_MARLIN_PAR_LIGNE=1` reste disponible ; la 209 réempaquette
  les poids au bit (aucun poids changé), mais la SORTIE servie n'est pas identique : à 0 et à 1, les 8 séquences réelles
  du Coder b=8 divergent dès le jeton 6 à 55 selon la séquence (237 b, poste2) ; KL contre HF bf16 en cours. Elle reste
  gagnante sur qkvo-i8c (209 c). Test cassant si le défaut revient à 1 :
  `tests/test_marlin_pile_par_ligne_209.py::test_232_le_facteur_par_ligne_est_a_la_demande_et_0_le_defaut`.

Chaîne du 24 au 26/09, deux faits distincts, chacun avec sa pièce :
* **parité du pas de décodage contre NInfer** (202, poste1) : Qwen3.8-27B-unsloth-mixte-i8c b=8, pas
  hôte 16,09 ms contre 15,98 chez NInfer (**+0,7 %, parité**). Le banc chat servi reste derrière,
  422,6 t/s contre ≈ 463 chez NInfer (**−8,7 %**) : l'écart est dans le service (banc − pas, 2,84 ms
  contre 1,29 au bit), pas dans les noyaux — décomposition en cours (204/221).
* **gain interne de la nuit du 24-25/09** (190, bilan poste2, main contre main, sans comparaison à un
  autre moteur) : même alias, même b, débit 323,4 → 394,8 t/s (**+22,08 %**), composé de
  172/175b/176/179/182/187 (`revue/poste2-piece190-cellule-mixte-b8-25-09.md`).

Gains propres à cette version, au-dessus de ces deux faits :
* **175/187/194/195b** au défaut : portes GDN α/β en un appel (`ACVRAM_GDN_AB=auto`, 175/175b) ;
  GEMV int8 par tranches de 6 au lieu de 16 (187, +6,65 % b=8 mixte) ; β‖α GDN sur un second flux
  (194 b2, `ACVRAM_GDN_AB_FLUX=1`, +2,20 % b=8) ; GEMM int8 étroit à K entier par canal (195b,
  `ACVRAM_ETROIT_CANAL=1`, +4,01 % b=8, −3,76 % J/jeton).
* **201** (poste5) : un modèle vision/MTP qui ne tenait pas n'est plus chargé en silence — capacité
  KV annoncée baisse pour en tenir compte (Qwen3.8 nvfp4 −7,4 %, gemma-4-31B-vision −21,4 %) ; copie
  int8 transitoire (i8c servi sans OOM), coût nul au banc. `revue/poste5-piece201-verdict-25-09.md`.
* **209 à la demande** (poste6, 232 b) : le facteur Marlin par ligne d'expert (209, exact au bit, +5,77 % sur qkvo-i8c) reste
  disponible par `ACVRAM_MARLIN_PAR_LIGNE=1`, à 0 par défaut. Sur Qwen3-Coder-30B-A3B-nvfp4 pur il gagne en salve unique à invites
  réelles (226 : +12,8 %) mais perd en débit soutenu (229 : −15,3 %, +25,4 % J) ; l'écart entre protocoles n'est pas expliqué, la release
  garde l'ancien comportement (`revue/poste6-piece226-verdict-26-09.md`, `revue/poste3-piece229-verdict-3bras-26-09.md`).
* **210/210b** (poste5) : `/v1/completions` — logprobs d'un jeton à texte vide gardés (suit
  `token_ids`, plus `text_delta`) ; usage omis dans le flux sans `stream_options.include_usage`
  (`CompletionChunk`) au lieu de {0,0,0} sur chaque fragment ; l'outil TTFT comptait ces jetons
  vides comme « aucun jeton reçu » — corrigé, ce n'était pas le service.
* **212** (poste4) : la marge de VRAM avant capture des graphes double sur les modèles à couche
  récurrente (GDN/KDA/mamba2) — `_KV_MARGE_MIN_GDN` 3 072 Mio au lieu de 1 536, ≈ −12 192 jetons de
  capacité KV sur les Qwen3.8.
* **Purge d'historique** (171, poste6) : `outils/purge-historique.sh` exécuté réellement le 25/09,
  1 015 Mio → 67 Mio, 146 réfs réécrites, Dolt intacte, vérifié sur clone neuf.

* **213 b** (poste5) : la contamination servie de la 213 (mixte b=8 : le 1er lot différait des lots suivants,
  requête après requête ≠ requête seule) est corrigée — le dtype du RoPE est fixé et `loader.py:313` ne
  laisse plus la première passe choisir une précision différente : **le 1er lot égale désormais les lots
  suivants au bit** ; la calibration fixe aussi sa précision (235).
* **26/09/2026 — pièce 250 (poste6) : `depaqueter_marlin(noyau="cuda")` acceptait à nouveau l'échelle globale PAR COLONNE d'un
  tenseur dense (E = 1, pièces 134/147).** La 209 (a) (25/09, a256676f8) avait fermé le noyau CUDA à toute échelle par colonne, pile
  OU dense, alors que seule la pile E > 1 (g [E, N]) lui est inconnue : deux tests p147 rouges depuis, sur main comme sur la branche
  232 (rejeu seul, même prise, 04:07). Le service n'était pas touché (`auto`, jamais `"cuda"` explicite, et aucune pile E > 1 n'est
  dépaquetée en service). Le choix `auto` refuse désormais lui aussi le noyau CUDA à une pile E > 1 par colonne (il aurait lu g[e]
  comme scalaire, sans erreur) ; test cassant `test_pile_par_colonne_refusee_au_cuda_et_auto_evite_cuda`.

* **243** (poste5, décision chef) : seuil GEMV → GEMM int8 abaissé à 16 dans la portée de déquant PARTAGÉE du préfill
  GDN (`ACVRAM_INT8_GEMV_MAX_PARTAGE=16` par défaut, 80 = témoin, sortie d'avant). HORS BIT, KL scellée tenue (b=8, 32 pas :
  max 0,00603 ≤ 0,00793 = 2 × le témoin déjà servi, argmax 254/256, PPL 6,0483 → 6,0657). Qwen3.8-27B-unsloth-mixte-i8c b=8 :
  préfill 8 × 78 0,821 → 0,410 s par lot (**−50 %**) ; banc chat servi (ABBA ×5, serveur neuf, -lgc 2700) 423,6 → **464,7 t/s
  (+9,70 %)**, J/jeton net 0,7615 → 0,6906 (**−9,3 %**) — le niveau de NInfer (≈ 463). Réserve : les dix fenêtres sont bridées en
  puissance (plafond 400 W), les deux bras également. Hors portée, le seuil reste 80 : la déquant non partagée régresse (733,7 µs
  contre 636,4 de GEMV à n = 78). `revue/poste5-piece243-verdict-26-09.md` ; test cassant `tests/test_int8_seuil_partage_243.py`.

* **26/09/2026 — pièce 226 (poste6, décision chef) : `ACVRAM_MARLIN_PAR_LIGNE=1` REVIENT AU DÉFAUT — la « régression » de la 209
  sur le Coder nvfp4 pur était un artefact du banc.** L'histoire vraie : la 209 (25/09) sert en Marlin les piles d'experts à échelles
  sous-normales par un facteur par ligne, exact au bit, +5,77 % sur qkvo-i8c ; la 220 (25/09) l'a remise en opt-in sur le banc de la 217
  (Coder pur b=8 : −8,9 % t/s, +20,7 % J, confirmé par quatre bras isolés) ; la 226 (26/09) a démontré l'artefact : ce banc
  (`banc-llamacpp-16-09.py decode`) génère librement depuis des invites de **jetons tirés** (`invite(k, n)`), ses sorties sont dégénérées
  (8/16 = un caractère répété) et **divergent entre bras** (9/16), donc le routage et les octets lus aussi — à jetons FIXES la 209 est plus
  rapide (pas GPU 4 862 contre 4 946 µs), à piles égales le GEMM Marlin ne diffère que par les données. **ABBA à invites réelles** (banc
  chat 102, celui du 209 c), Coder pur b=8, 5 + 5, serveur neuf par passe : défaut 0 **1 630,9 t/s · 0,1364 J** (bridage puissance) contre
  1 **1 839,7 · 0,1113** (aucun bridage) = **+12,8 % / −18,4 %**. Ni le préfill (égal), ni l'épilogue par colonne (+2 µs/couche) n'y étaient
  pour rien ; le compte direct d'experts distincts reste un résidu (trace de routage non écrite à l'arrêt du serveur).
  `revue/poste6-piece226-verdict-26-09.md`. REGLES § 4 : une cellule de débit MoE en génération libre se prend à invites réelles ou à
  jetons fixes, jamais à jetons tirés. Test cassant si le défaut revient à 0 :
  `tests/test_marlin_pile_par_ligne_209.py::test_226_le_facteur_par_ligne_est_le_defaut_et_0_le_temoin` (remplace celui de la 220).
* **25/09/2026 — pièce 220 (poste6, décision chef) : la 209 (facteur Marlin par ligne d'expert) revient en OPT-IN —
  `ACVRAM_MARLIN_PAR_LIGNE` vaut 0 au défaut, 1 = témoin de la 209.** Sur **Qwen3-Coder-30B-A3B-nvfp4 PUR** (67 477 échelles
  sous-normales sur deux couches, puis 128, 207, 262), le 1 ne bascule pas quatre couches mais les **48** : `experts_layout` passe de
  `naturel` (tout le modèle refusé, comme depuis la 157) à `marlin-w13`, et le service perd — banc de la 217 (poste3, be837ca1 → main :
  **−8,91 % t/s, +20,65 % J/jeton**) ; 220, quatre bras isolés sur main, serveur neuf par passe, b=8 : défaut 1 539,8 t/s / 0,1521 J,
  `ACVRAM_MARLIN_PAR_LIGNE=0` **1 689,3 / 0,1272 (+9,7 %, −16 %)**, `ACVRAM_ETROIT_CANAL=0` 1 533,2 / 0,1541 (hors de cause).
  La 209 ne gagne que là où peu de piles basculent : qkvo-i8c, 4 couches sur 48, +5,77 % b=8 (`poste6-piece209c-verdict-25-09.md`) ;
  elle y reste disponible par `ACVRAM_MARLIN_PAR_LIGNE=1`. Mécanisme à établir (pièce à venir : quand le Marlin MoE gagne-t-il
  contre la naturelle — préfill contre décodage, fraction de piles basculées). Test cassant si le défaut revient à 1 :
  `tests/test_marlin_pile_par_ligne_209.py::test_220_le_facteur_par_ligne_est_opt_in_et_0_le_defaut`.
* **25/09/2026 — pièce 212 (poste4, sur la 201 de poste5) : la marge de VRAM avant capture des graphes
  DOUBLE sur les modèles à couche récurrente (GDN/KDA/mamba2/lfm2) — `warm_graphs` grossit la mémoire de
  la carte APRÈS le chargement, hors de tout ce que la réserve de préfill voyait.**
  `warm_graphs` (`engine/graphes.py:40`) capture un graphe CUDA par forme (`engine/graphs.py:983`) ; le
  pool mémoire qui les porte (`graphs.py:1075-1086`, `graph.pool()`) n'est jamais libéré et grossit à
  chaque forme nouvelle — coût mesuré (B=8, CTX=2048, 5 modèles) : Qwen3.8-27B-nvfp4 1 880 Mio,
  -unsloth-mixte-i8c **2 734 Mio**, -attn-gdn-i8c 1 720 Mio, TOUS AU-DESSUS de l'ancienne marge fixe
  (1 536 Mio) — pas seulement l'i8c de la 153/201. gemma-4-31B-vision et Coder-30B-A3B (aucune couche
  récurrente) restent sous cette marge (−868 et 266 Mio). Dérivation structurelle tentée (config.json,
  sur le modèle de 172/201) et abandonnée : les trois Qwen3.8 ci-dessus partagent un `config.json`
  identique mais divergent de plus de 25 % — aucune fonction de l'architecture seule ne peut couvrir les
  trois sans en sur-réserver un. Correctif : `loader._KV_MARGE_MIN_GDN` (3 072 Mio, +12 % sur le pire
  mesuré), appliqué quand le manifeste porte une couche `.linear_attn.` (`_a_des_couches_lineaires`) ;
  les modèles denses gardent l'ancienne marge (1 536 Mio). **Prix pour les modèles GDN** : ≈ 12 192
  jetons de capacité KV en moins (1 536 Mio de marge en plus ÷ 132 096 o/jeton, Qwen3.8). Verdict :
  `revue/poste4-piece212-warmgraphs-25-09.md`.
* **25/09/2026 — pièce 209 (poste6) : piles d'experts NVFP4 à échelles sous-normales servies en MARLIN, facteur par ligne d'expert
  (`ACVRAM_MARLIN_PAR_LIGNE=1` ; 0 = témoin : préparation d'avant, piles refusées)**. Depuis la 157, une pile qu'un facteur
  Marlin commun écraserait (448 et 2⁻⁹ dans le même expert — Qwen3-Coder-30B couches 0, 1, 2, 4, 43 experts en couche 0) restait
  naturelle : `decode_mma` W4A4 à b=8, GEMV naturel à b=1, préfill « groupe ». Le facteur devient PAR (expert, ligne) et l'échelle
  globale par (expert, colonne) [E, N] : `preparer_pile` (`marlin_port/__init__.py`), épilogue du Marlin MoE porté (`gs_par_colonne`,
  geste du dense 101), GEMV Marlin CUDA (`gs_ld`), w13 par colonne. **Exact au bit des poids** (8 piles réelles : 0 valeur fausse /
  70,8 M, tests 209 a/b) ; les piles sans écrasement gardent leur préparation au bit. **La sortie servie du Coder change** (4 couches
  sur 48) : KL de décodage b=8 max 0,324 ≤ 2 × témoin 0,286 (témoins à échantillon égal), argmax 0,961 ≥ 0,957, PPL 12,80 → 12,61.
  Servi, banc chat ABBA ×5 : Coder b=8 1 669,9 → 1 766,3 t/s (**+5,77 %**), J/jeton net −8,58 % ; b=1 +0,52 %.
  `revue/poste6-piece209{a,b,c}-verdict-25-09.md`. Ligne de régime : `experts_layout=marlin-w13`, plus aucun « pas de piles Marlin ».
* **25/09/2026 — pièce 210b (poste5, suite de la 210) : `/v1/completions` — logprobs d'un jeton à texte vide gardés,
  usage omis en flux sans `include_usage`.** (a) `acvram/server/app.py:1201` : la boucle de logprobs (hors flux) ne
  retenait un pas que si `out.text_delta` — un jeton décodé en texte vide (octet UTF-8 partiel, jeton spécial) sortait
  des listes (`tokens` = [] pour `completion_tokens` = 1, logprob perdu) ; suit désormais `out.token_ids`, listes
  alignées sur les jetons générés. (b) `acvram/server/protocol.py:323` : les fragments de flux étaient des
  `CompletionResponse`, `usage` par défaut {0, 0, 0} sur chaque fragment ; `CompletionChunk` (`usage: Optional[Usage] =
  None`, omis par `exclude_none`) ne le porte plus que sur le dernier fragment, avec `stream_options.include_usage`
  (contrat OpenAI). Tests fabriqués (`tests/test_server.py`) : tokens ["", "hello"], logprobs [−1,5, −0,25] ; sans
  `include_usage` aucun `usage`, avec : dernier fragment {3, 2, 5} ; témoins sur l'ancien code → ROUGES. 83 verts.
  `revue/poste5-piece210b-verdict-25-09.md`.
* **25/09/2026 — pièces 210 + 211 (poste5, poste2) : l'outil TTFT comptait un jeton à texte vide comme « aucun jeton
  reçu » — outil faux, pas le service ; 13 outils de mesure important l'installation editable au lieu de leur propre
  worktree.** 210 : `outils/gpu/mesure/ttft-service-p145.py:40` (`premier_fragment`) n'acceptait comme premier jeton
  qu'un fragment au texte non vide ; le service génère bien le jeton (`completion_tokens` = 1), il se décode en "" —
  toutes les passes L = 512 du mixte-i8c et de l'attn-gdn-i8c de la 201 (`revue/poste5-piece201-verdict-25-09.md`)
  étaient rendues nulles par ce défaut d'instrument, pas par le service. Corrigé : tout fragment à `choices` compte
  (texte vide compris), `jetons_texte_vide` compté à part. Servi (rejoué) : L = 78 **180,78 ms** (n 40, 0 vide), L = 512
  **299,08 ms** (n 27, 1 vide). `tests/test_ttft_texte_vide_210.py` : jeton vide compté, ancien critère lève sur le même
  flux, flux sans `choices` reste une erreur. 211 : même mécanisme que la 168 (`_regime_noyaux()` ne pose pas
  `sys.path`, à l'appelant de le faire) — 13/22 scripts de `outils/gpu/mesure/` dérivaient leur racine du `cwd`, d'un
  chemin en dur ou pas du tout ; corrigés pour dériver de `__file__` (`tests/test_arbre_outils_mesure.py:1`, garde qui
  rejoue la logique `sys.path` de chaque fichier et se rend faux sur un faux outil fabriqué exprès). Périmètre non
  couvert (à la demande de chef) : `outils/` hors `gpu/mesure/`, une soixantaine de candidats non revus.
  `revue/poste5-piece210-verdict-25-09.md`, `revue/poste2-piece211-25-09.md`.
* **25/09/2026 — pièce 201 (poste5, sur la 153c de poste4) : un modèle qui ne tient pas n'est plus chargé en silence ;
  la capacité KV annoncée BAISSE sur les modèles à vision ou à MTP — c'est le prix d'un compte juste.**
  Qwen3.8-27B-nvfp4-attn-gdn-i8c chargeait sans exil puis tombait en OOM au premier pas (0/64 couches exilées,
  148 Mio libres). Deux poids que le planificateur ne voyait pas : (1) la tour de vision (runner) et les têtes MTP
  (`_charger_mtp`), chargées après la borne du KV — `_octets_reels` ne comptait que couches, embed et tête ; (2) la
  copie int8 signée de chaque poids par canal (`kernels._i8c_poids`), gardée à vie dès le premier préfill cuBLAS :
  6,84 Gio sur ce modèle, fabriqués pendant `warm_graphs`. Correctifs : `loader._octets_annexes` compte par
  exclusion tout bloc hors couches/embed/tête dans la borne et l'exil ; la copie i8c est transitoire (par appel, ou
  une fois par portée de partage B'), au bit par construction ; la tranche nvfp4 de la 153 (PPL de la tête à
  vocabulaire étendu) ne s'active plus qu'au-delà de 1 Gio de copie fp32 (`ACVRAM_TRANCHE_COPIE_MIN`, 0 = témoin) : au
  seuil de 256 Mio elle tranchait les projections servies et changeait les logits d'un préfill 8 × 512 (retour au bit
  de main, sha256 à l'appui). Capacité KV annoncée, B = 8 :
  Qwen3.8-27B-nvfp4 à 32 k **128 960 → 119 440 jetons (−7,4 %, MTP 299,7 Mio)** ;
  gemma-4-31B-it-nvfp4-vision à 8 k **10 848 → 8 528 (−21,4 %, vision + MTP 1 098 Mio)** ; l'i8c à 32 k passe de
  l'OOM à 9/64 couches exilées. Coût de la copie transitoire au banc chat b=8 mixte (ABBA ×4 contre main) :
  médianes 426,05 → 426,05 t/s (0,00 %), J/jeton +0,28 %, TTFT L = 78 +0,04 %. Verdict : `revue/poste5-piece201-verdict-25-09.md`.
* **25/09/2026 — pièce 195 (poste6) : GEMM int8 étroit à K ENTIER PAR CANAL, AU DÉFAUT** (`ACVRAM_ETROIT_CANAL=1` ;
  0 = témoin nommé, le noyau à tranches d'avant ; décision déléguée par l'utilisateur). Les linéaires int8 symétriques par
  canal (tous les int8 du mixte-i8c, attention et GDN de `Qwen3.8-27B-nvfp4-attn-gdn-i8c`) à 2 ≤ b ≤ 16 sont servies par
  `_etroit_canal_kernel` (`kernels/gemm_etroit.py`) : K entier par programme, sans tranche ni partiel ni atomique, géométrie
  de NInfer, table par forme mesurée (o‖out 25,2 → 22,7 µs, GDN qkv‖gate 65,6 → 56,5, gate‖up 141 → 117, down 66 → 59,
  qkv attention ≈ 0 ; 0,86 ms/pas au banc). **La sortie servie change** : somme fp32 sur K entier au lieu de 2-5 tranches
  (mode « ± 1 ulp », REGLES § 1), jamais au bit du témoin. Servi, banc chat ABBA ×5, mixte b=8 : **397,7 → 413,6 t/s
  (+4,01 %)**, J/jeton net 0,804 → 0,773 (**−3,76 %**), W égaux. Qualité en décodage b=8 (KL par position, 8 × 32) : deux
  prises — la 1re donnait NON tenu sur le mixte (max 0,0049 > 2 × 0,00054), mais son témoin T1 ne couvrait qu'une séquence
  (32 positions contre 256 pour B : défaut d'instrument) ; le rejeu scellé par chef, témoins à échantillon égal (8 séquences
  seules à b=1), donne T1 max 0,0052 → seuil 0,0104, **KL A‖B max 0,0049 tenue**, argmax identique (0,984 = 0,984), PPL
  9,728 / 9,732 / témoin 9,716 ; sur `attn-gdn-i8c` : 0,0038 ≤ 0,0202, argmax 0,992. Tests : référence ± 2⁻⁷ qui refuse un
  zéro à ± 1 et K/2 (fautes injectées une fois, 198 poste2), témoin 0 au bit du noyau d'avant sur carte, défaut 1 verrouillé
  par test. Formes hors table (α/β int8 48 × 5120) restent sur le noyau d'avant. `revue/poste6-piece195-verdict-25-09.md`.
  Ligne de régime : ` etroites=serie+canal(table|temoin|BNxBKxWxS)`.
* **25/09/2026 — pièce 194 b2 (poste1) : β‖α des couches GDN sur un second flux, AU DÉFAUT** (`ACVRAM_GDN_AB_FLUX=1` ;
  0 = témoin série). Les portes α‖β bf16 (3 programmes, 9-14 µs, jusqu'ici sur le chemin critique) tournent pendant la
  pile qkv‖gate int8 qui lit la même entrée ; jointure avant la récurrence. Au bit par construction, test qui casse quand
  la jointure manque (`tests/test_gdn_ab_flux_194.py`). Servi, banc chat ABBA ×5 : mixte b=8 399,2 → 408,0 t/s
  (**+2,20 %**, z 8,3), J/jeton net −1,9 % ; b=1 +0,46 % ; Qwen3.8-27B-nvfp4 inerte (−0,02 % / 0,00 %) ; capture
  godets {1, 2, 8, 16} 4/4 — `revue/poste1-194-b2-verdict-25-09.md`. Ligne de régime : ` abflux`.
* **25/09/2026 — pièce 190 (bilan de la nuit 24-25/09)** : cellule officielle ABAB×5 mesurant
  ensemble les six pièces fusionnées depuis 24bcdd07 (172, 175/175b, 176, 179, 182, 187).
  Qwen3.8-27B-unsloth-mixte-i8c b=8 : débit médian **323,4 → 394,8 t/s (+22,08 %)**, J/jeton net
  0,8784 → 0,8125 (−7,50 %) — `revue/poste2-piece190-cellule-mixte-b8-25-09.md:5`. Attribution par
  composition (chef) : le gain se décompose en 175b (−12,0 % de pas_gpu,
  `revue/poste6-piece175-verdict-25-09.md:64`) × 187 (+6,65 % au banc b=8,
  `revue/poste5-piece187-verdict-25-09.md:10`) × ≈+2 % (fenêtre d'admission, 179,
  `revue/poste5-piece179-verdict-25-09.md:78`) ≈ 1/(1−0,12) × 1,0665 × 1,02 ≈ +24 %, cohérent avec
  le +22,08 % mesuré ; 172 (déquant partagé préfill, −12,5 %/−11,3 % forward,
  `revue/poste5-piece172-verdict-25-09.md:20`), 176 (GDN qkv‖gate pile int8, 0,9967 à b=8 — FAUX
  sur l'ampleur mais sans régression, `revue/verdict-176-gdn-qkv-gate-25-09.md:16`) et 182 (z sans
  cast +0,40 %, GQA neutre au banc, `revue/poste1-182-verdict-25-09.md:7`) contribuent au décodage
  mais pèsent peu au b=8 servi. Qwen3.8-27B-nvfp4 b=8 (témoin, hors chemin int8/GDN mixte) :
  481,9 → 499,0 t/s (+3,55 %), `revue/poste2-piece190-cellule-mixte-b8-25-09.md:9` — dans la
  fourchette du scellé, TENU. gemma-4-31B-it-nvfp4-vision b=8 (falsificateur direct, aucune des
  six pièces ne le touche) : 402,3 → 406,2 t/s (+0,97 %, NEUTRE ± 2 %),
  `revue/poste2-piece190-cellule-mixte-b8-25-09.md:14` — TENU, confirme que le gain mixte est bien
  localisé aux pièces GDN/int8. Qwen3.8-27B-unsloth-mixte-i8c b=1 : 62,2 → 64,3 t/s (+3,38 %),
  J/jeton plat (dominé par la puissance de repos à ce B) — `revue/poste2-piece190-cellule-mixte-b8-25-09.md:19`.
  Le scellé (`revue/poste2-piece190-scelle-25-09.md`) ne portait pas 175b ni 187 :
  faute de PÉRIMÈTRE du scellé, pas de mesure.

* **25/09/2026 — pièce 175 b** : `ACVRAM_GDN_AB=auto` **au défaut** (était `separe`, opt-in) : les portes α et β bf16 des
  couches GDN de l'alias mixte en un appel par couche — M = 1 concat, 2 ≤ M ≤ 8 GEMM étroite fp32 (`kernels/gemv_bf16_etroit.py`),
  au-delà les deux appels ; AU BIT des deux F.linear à chaque M (`test_gdn_ab_175`, `torch.equal`), donc sans critère KL.
  Décodage b=8 mixte −2,3 ms/pas (−12 %), b=1 −0,25 ms ; inerte sur des α/β nvfp4 (défaut). `separe` reste le témoin.
  Détail : revue/poste6-piece175-verdict-25-09.md.

* **25/09/2026 — pièce 176** (poste1) : les projections GDN qkv‖gate de l'alias mixte-i8c en une
  pile int8 (`stack_int8_linears`, `GatedDeltaNet.fuse`, servie à M ≤ 16), témoin
  `ACVRAM_GDN_QKV_GATE=0`. Sortie inchangée au bit. ABBA certifie-b12 : b=8 mixte 0,9967 (**FAUX
  sur l'ampleur**, −0,065 ms prédit −0,25, sans régression), b=1 mixte 0,9863 (TENU, −1,4 %) ;
  Qwen3.8 nvfp4 témoin inchangé aux deux B. Détail : revue/verdict-176-gdn-qkv-gate-25-09.md.

* **25/09/2026 — pièce 182** (poste1) : (1) GDN `z` sans cast bf16→fp32 intermédiaire
  (`ACVRAM_GDN_Z_BF16`), au bit, +0,40 % méd. au banc chat mixte b=8 (TENU au critère scellé,
  seuil +0,2 %). (2) attention GQA optimisée (`ACVRAM_PA_GQA`), au bit ; gain en processus TENU
  (−0,18 ms ctx≈600, −0,42 ms ctx≈2000) mais NEUTRE au banc (+0,12 %, sous 2 σ). Les deux restent
  au défaut (au bit, aucune régression). Détail : revue/poste1-182-verdict-25-09.md.

* **25/09/2026 — pièce 187** : le GEMV int8 traite ses activations par tranches de **6** au lieu de 16
  (`ACVRAM_INT8_TRANCHE` pour N ≤ 16, `ACVRAM_INT8_TRANCHE_PREFILL` au-delà ; 16 = témoin d'avant). Sortie **identique au
  bit** : test sur les poids réels du mixte (N 2-80, bf16 et fp32) et bras cassant. À NV ≤ 6, le noyau tient en 128
  registres, soit 2 blocs par SM au lieu d'un, ce qui l'emporte sur les relectures des poids. GEMV −22 à −44 % de 16 à 78
  jetons ; servi sur Qwen3.8-27B mixte-i8c : **+6,65 %** au banc chat b=8, **+10,93 %** à b=16, J/jeton −3 à −5 % ;
  Qwen3.8-27B-nvfp4 (pas d'int8 servi) inchangé. Détail : revue/poste5-piece187-verdict-25-09.md.
* **25/09/2026 — pièce 179** : (1) B' (172) couvre aussi la déquantification int8 du préfill (`int8_matmul` au-delà de
  `ACVRAM_INT8_GEMV_MAX` = 80 lignes) : au bit (tests, bras cassant, logits de l'alias mixte) ; préfill 8 × 92 −51 %,
  8 × 120 −49 % sur l'alias mixte ; sans effet sous 80 jetons par invite (GEMV). (2) **Fenêtre d'admission au défaut**
  (`ACVRAM_ADMISSION_FENETRE_MS=5`, 0 = coupé) : moteur vide et au moins 2 requêtes en file, le serveur attend que la
  file cesse de grossir (5 ms, 20 au plus) avant de préfiller, pour grouper une rafale au lieu de la couper en deux pas.
  Banc chat b=8 : Qwen3.8 +2,0 % de débit, −2 % de J/jeton ; une requête seule n'attend pas (TTFT +0,07 ms à 78 jetons,
  +0,15 à 512, débit b=1 inchangé). La composition des lots de préfill, qui dépendait déjà du moment d'arrivée, change ;
  l'arithmétique d'un lot, non. Détail : revue/poste5-piece179-verdict-25-09.md.

* **25/09/2026 — pièce 172** : au préfill de plusieurs séquences, la boucle par séquence des couches à récurrence
  linéaire (Gated DeltaNet de Qwen3.5/3.8) déquantifie chaque poids NVFP4 UNE fois au lieu d'une fois par séquence
  (`ACVRAM_DEPAQ_PARTAGE`, défaut 1 ; 0 = témoin). Sortie **identique au bit** : tests, bras cassants, logits de
  Qwen3.8 et Qwen3.5-35B-A3B. Forward de préfill −12,5 % / −11,3 % sur Qwen3.8 et −3,8 % / −3,6 % sur Qwen3.5-35B ; TTFT
  servi sous 8 requêtes −9,2 % / −8,1 % et −3,1 % / −3,4 % (8 × 78 jetons / longueurs mêlées). La cause de l'écart de
  sortie de `GDN_PREFILL_LOT=1` est trouvée (cuBLAS bf16 réduit en bf16 selon M, pièce 169). Détail :
  revue/poste5-piece172-verdict-25-09.md.

* **24/09/2026 — pièce 166** : l'opt-in `ACVRAM_PREFILL=marlin` (pièce 147 L2, GEMM Marlin W4A16 au préfill de la disposition
  unique, sans dépaquetage) est RETIRÉ, verdict FAUX : TTFT servi b=1 +4 / +27 / +35 % à 512 / 2 048 / 4 096 jetons, J/préfill
  +5 / +28 / +36 %, KL 2,3-3 × les témoins (PPL par fenêtre tenue) — Marlin perd à grand M contre dépaquetage + cuBLAS ;
  `ACVRAM_PREFILL` revient à `bf16 | w4a16 | w8a8 | w4a4`, `ACVRAM_PREFILL_MARLIN_MAX_M` disparaît. Détail :
  revue/poste6-piece147L2-verdict-24-09.md ; mécanisme : acvram-memoire/MECANISMES.md.
* **25/09/2026 — pièce 165** : `ACVRAM_GDN_PREFILL_LOT=1` est l'option TTFT des modèles Gated DeltaNet (Qwen3.5/3.8) :
  au préfill de plusieurs séquences, les projections GDN du lot passent en un appel au lieu d'un par séquence. TTFT servi
  sous 8 requêtes simultanées (Qwen3.8-27B-nvfp4) : **−18,6 %** (8 × 78 jetons, 431,5 → 351,1 ms) et **−15,7 %**
  (longueurs mêlées, 466,6 → 393,6 ms). Elle reste en OPT-IN, car elle change la sortie au-delà du critère scellé :
  accord d'argmax 98,38 % contre 99,19 % pour les séquences servies seules (Qwen3.8, 8 × 78), KL_max jusqu'à 2,2 × le
  témoin (0,319 contre 0,146 ; 1,240 contre 0,568 sur Qwen3.5-35B-A3B). La PPL par séquence reste au niveau des témoins.
  Détail : revue/poste5-piece165-verdict-24-09.md.

* **24/09/2026 — pièce 156** : les linéaires NVFP4 des modèles DENSES sont servis par défaut en disposition Marlin unique
  (GEMV v2, TPB par forme) : +57 à +90 % de débit à b = 8, b = 1 inchangé (0,979 à 0,996), sortie qualifiée (KL sous 2 ×
  témoin, PPL identique), TTFT +2 à +4 ms (B/A 1,001 à 1,012 sur gemma4 31B et Qwen3.8-27B, invites de 512, 2 048 et
  4 096 jetons : revue/poste6-piece147-verdict-24-09.md, dépaquetage CUDA de la pièce 147) ; MoE inchangés ; repli
  `ACVRAM_PROJ_MARLIN=0`. Cinq variables pilotent la disposition (`ACVRAM_PROJ_MARLIN` défaut 1, `ACVRAM_PROJ_MARLIN_PORTEE`
  défaut `denses` — un modèle à `MoEBlock` garde son chemin naturel, ses 32 alias touchés par les linéaires hors experts
  n'étant pas mesurés : revue/poste1-piece142-inventaire-denses-24-09.md —, `ACVRAM_GEMV_MARLIN_V2` défaut 1,
  `ACVRAM_GEMV_MARLIN_TPB` et `ACVRAM_GEMV_MARLIN_S` défaut 0 = règle automatique) ; replis nommés au chargement si la
  capacité KV ou la mémoire manquent (`ACVRAM_PROJ_MARLIN_CAPACITE`, refus explicite, jamais un exil silencieux).

* **24/09/2026 — pièce 157** : les échelles de bloc NVFP4 sous-normales en S0E5M3 (le format d'échelle du port Marlin, plage
  ≈ 2^14,8 contre 2^17,8 pour l'E4M3 d'origine) faisaient perdre des blocs de 16 poids entiers, mis À ZÉRO plutôt que
  représentés, dans la disposition Marlin — touché au défaut servi (MoE : max|Δ logits| = 4,52 sur Qwen3-14B, trouvé par
  la suite complète de la bascule 156). Corrigé : facteur d'échelle PAR LIGNE pour les poids denses (au lieu d'un facteur
  par tenseur/pile), poids encore inexact exclu de la disposition et rendu au chemin naturel (raison nommée) ; les piles
  MoE dont un poids écrase restent en disposition naturelle. Preuve : Qwen3-14B au bit contre le naturel après correctif
  (avant : 4,52 ; après : 0,0) ; coût ≤ 2 % (b = 1 1,006, b = 8 1,017, Qwen3-Coder-30B).
  Détail : revue/verdict-157-marlin-sous-normales-24-09.md.

* **24/09/2026 — pièce 146** : trois défauts corrigés dans le chemin par défaut du cache KV. (1) Une séquence tronquée
  par épuisement du KV n'était JAMAIS livrée à la requête HTTP (chemin pipeline et spéculatif) — la sortie finie était
  jetée sans clore la requête ; corrigé, les séquences épuisées sont livrées dans le pas même où `step` les détecte.
  (2) Le budget KV tombait à 6 % de la VRAM dès qu'une seule séquence y tenait (marge de 7 % posée en dur, avant la 146
  au chargement) ; corrigé par un départage min(KV, demande) après exil. (3) La tête liée (`lm_head`) laissée en bf16
  puis convertie en fp32 à la volée (5,25 Gio) provoquait un OOM à la chauffe sans `empty_cache()` préalable ; corrigé.
  Preuve bout en bout (uvicorn réel, graphes + pipeline) : rouge sur le commit d'avant, vert après, pour chacun.
  Détail : revue/verdict-146-kv-defaut-24-09.md.

* **24/09/2026 — pièce 156 (fusions GDN)** : six fusions du décodage des couches à récurrence linéaire (GDN), toutes
  numériquement identiques (au bit ou à l'ulp) au chemin qu'elles remplacent, mesurées sur Qwen3.8-27B, b = 8, disposition
  Marlin qualifiée, **toutes par défaut** (poste5, 255042e8) : **F2** conv de décodage fusionnée, **F4** état GDN mis à
  jour en place, **F5** résidu différé des couches GDN (ensemble : −7,8 % de temps de pas,
  revue/poste5-piece156c-verdict-24-09.md, au bit) ; **F6** RMSNorm en registres (+8,1 % à b = 8 seule avec F1/F3, au bit
  — l'ordre de sommation d'un fil n'est pas observable en sortie bf16 sur ≤ 8 carrés) ; **F1** portes dans le noyau fla et
  **F3** norme gated Triton (± 1 ulp bf16, KL/PPL tenues contre le témoin, initialement laissées en opt-in le temps de la
  revue — désormais par défaut). Détail : revue/poste5-piece156c-verdict-24-09.md, revue/poste5-piece156d-verdict-24-09.md.

* **24/09/2026 — pièce 147** : le dépaquetage Marlin → bf16 au préfill (disposition unique) coûtait +26 à +34 ms par
  requête (v1, un fil par tuile, copies `.contiguous()` des vues q/k/v) ; réécrit (v2, un fil par colonne, lignes
  entières en deux uint4, vues à pas libre sans copie) : **+2 à +4 ms** à toute longueur d'invite (512 à 4 096 jetons),
  au bit contre les chemins Triton et torch de référence. `ACVRAM_DEPAQUETAGE=auto` choisit CUDA si l'extension l'a,
  sinon Triton. Détail : revue/poste6-piece147-verdict-24-09.md.

* **24/09/2026 — pièce 161** : le `.so` compilé du port Marlin était PARTAGÉ entre worktrees sous un nom de cache fixe —
  deux arbres aux sources différentes alternant sur la même carte se recompilaient l'un l'autre (25 s de nvcc à chaque
  changement d'arbre, y compris hors verrou carte.sh) sans jamais désigner l'autre arbre comme cause. Corrigé : cache
  keyé par empreinte sha256 des sources (comme `kernels/__init__.py`, pièce antérieure sur l'extension principale),
  sources copiées dans le cache, le moteur en service ne relance jamais ninja (charge le `.so` de son empreinte ou
  replie au naturel, raison imprimée). Détail : revue/poste6-piece161-verdict-24-09.md.

* **24/09/2026 — pièce 162** : bilan chiffré matin/soir (`c10cfee5` contre `HEAD` `5375945b`), ABAB × 5, -lgc 2700,
  Qwen3.8-27B-nvfp4 et gemma-4-31B-it-nvfp4-vision, b = 1 et b = 8, plus TTFT à une invite de 2 048 jetons. Débit à
  b = 8 : Qwen3.8 **+66,66 %** (274,4 → 457,3 t/s), gemma **+56,95 %** (252,4 → 396,8 t/s, dans la bande prédite
  57-60 %). Énergie à b = 8 (J/jeton net, BAISSE = gain) : Qwen3.8 1,164 → 0,691 J/jeton (**+40,65 %** d'économie),
  gemma 1,265 → 0,803 J/jeton (**+36,53 %**). TTFT inchangé aux deux modèles (± 1 %, Marlin/GDN sont des leviers de
  décodage, pas de prefill). Isolation des deux leviers (Qwen3.8, HEAD seul, b = 8, `ACVRAM_GDN_ETAT_EN_PLACE`,
  `ACVRAM_GDN_CONV_FUSEE`, `ACVRAM_GDN_RES_DIFFERE`, `ACVRAM_GDN_PORTES_NOYAU` (F1), `ACVRAM_GDN_NORME_FUSEE` (F3),
  `ACVRAM_NORME_REGISTRES` (F6) tous à 0) : GDN seul **+8,76 %** de débit (bande prédite 3-15 % tenue), Marlin seul
  (déduit) **+53,5 %** ; composition vérifiée 1,535 (marlin) × 1,0876 (gdn) = 1,670 contre 1,667 mesuré directement
  (écart 0,2 %). Détail : scratchpad/poste2-piece162-bilan-24-09/verdict-final.md.
