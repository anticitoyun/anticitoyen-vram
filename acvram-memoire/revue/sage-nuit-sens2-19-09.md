# Sage — nuit du 19 au 20/09 : objectif « moins de joules, au moins aussi vite » avant 08 h ; plan de carte et arithmétique

Source : utilisateur 17 h (« installe le .deb, relance les services, poursuis au sens 2, résultat avant 8 h, tout le groupe en multitâches ») ; `verdict-p2-moteur-19-09` (Manon, 1fbe843) ; `comparatif-cinq-moteurs-17-09` ; `verdict-modes-energie-14-09` ; `model.py:1254-1346`, `1973-2045`.

## 0. Fait à 17 h 05 (Sage)

* Services : **8082 et 8083 relancés** (iGPU Intel, hors carte 0) ; **8081 mort au démarrage** — `llamacpp-appoint:56` ne trouve pas « 3080 Ti » dans `--list-devices`, `--device ""` refusé (`/mnt/AI_GENERATOR/llamacpp/appoint.log`). Jérôme corrige (à sec, autre carte).
* `.deb 0.6.13` : `sudo` interactif requis, Sage ne l'a pas → l'utilisateur tape `sudo dpkg -i acvram_0.6.13_amd64.deb` (installé = 0.6.10).
* P2 : **3/4 tenues** (équivalence B 6/5/8 ≤ 2A+2 ; prefill 18 850 j/s, +14 % ; J 0,89-0,93 × A). Ligne 2 bloquée par l'instrument : `perplexity()` prend `layers.py:500` → `int8_matmul` déquant (OOM 13 Gio), pas le chemin `cublas` d'`Engine.generate`. **Pas une réfutation** : l'instrument se répare à sec, la PPL se mesure ensuite (15 min de carte).

## 1. Où est le temps : le prefill est borné par le débit des tensor cores, pas par la mémoire

Coder-30B-A3B (`config.json` : h 2048, i_moe 768, 128 experts top-8, 48 couches, 32 têtes q / 4 kv × 128, vocab 151 936) :

```
experts / jeton / couche   8 × 3 × 2048 × 768 × 2      = 75,5 MFLOP   (67 %)
projections q,k,v,o        2 × (2048×4096 + 2×2048×512 + 4096×2048) = 37,7 MFLOP   (33 %)
× 48 couches + lm_head 2048×151 936×2                 ≈ 6,05 GFLOP / jeton
× 2 047 jetons                                         ≈ 12,4 TFLOP / pas de prefill
```

| régime | borne (tensor cores denses, acc fp32) | mesuré | ratio |
|---|---|---|---|
| défaut W4A16 (Marlin → MMA bf16, ~105 TFLOPS) | 118 ms = 17 300 j/s | 124,6 ms = 16 426 | **95 %** |
| P2 : projections int8 (×2), experts bf16 | 118 × (0,67 + 0,33/2) = 98 ms = 20 800 | 108,6 ms = 18 850 | 90 % |
| experts W4A4 MMA FP4 (×4) + P2 | 118 × (0,67/4 + 0,33/2) = 39 ms | vLLM W4A4 : 58 ms (35 241) | — |

Conséquence, écrite pour ne plus la rechercher : **aucune optimisation de lecture des poids ne fera gagner plus de 5 % au prefill défaut** — le plancher est le débit bf16. Le ×2 exige que les experts calculent en FP4 (ou FP8) sur les tensor cores, c'est-à-dire des **activations quantifiées** : le chemin existe (`nvfp4_gemm_grouped_mma`, `model.py:1268`) mais (a) il est exclu sous la disposition unique Marlin (`not unique`, `:1268`) — le porter coûte la double disposition réfutée le 18/09 (+14,5 Gio) ou l'abandon de Marlin au décodage (−3,5 % b=1, −9 % b=12 mesurés le 18/09) ; (b) sa qualité : E2M1 sur les activations, +2,58 % PPL le 13/09 sans lissage ; GLM k48-w4a4 réfuté le 16/09 (1,0229 > 1,010). Sur Coder à 1,0155 au défaut, il reste **0,0045 de marge** sous 1,020 : le W4A4 n'est classable que si la rotation Hadamard + AWQ déjà câblées (`hadamard`, `_stacks_awq`, `:1304-1315`) ramènent la perte sous 0,4 %. **Ce chiffre n'existe pas ; il se mesure à sec** (fausse quantification `ACVRAM_PREFILL_A4` sur les activations réelles, `:1339`) avant toute carte.

## 2. Où sont les joules : b=12, −6 % à trouver, deux leviers chiffrés

Cible : llama.cpp b=12 **1 066 t/s au mieux / 0,2136 J net**. Nous : 1 361,4 / 0,2265. Marge de vitesse à dépenser : **22 %**.

| levier | prédiction Sage | source | coût |
|---|---|---|---|
| `-lgc` (horloge cœur bornée ; `-pl` < 400 refusé par la 5090) | 14/09 à l'ancien régime (630 t/s) : `-lgc 2100` → −18,4 % t/s, **−8,7 % J**. Au régime MMA actuel, la part instructions est plus faible : Sage prédit **−10 à −14 % J pour −10 à −18 % t/s** à 2 100 ; 2 400 : −6/−9 ; 2 700 : −3/−4 | `verdict-modes-energie-14-09:39` | 4 bras × 40 s ABAB = 25 min |
| P2 projections int8 au décodage (déjà dans B : `int8_gemv` par canal) | J/jeton b=12 : non mesuré par Manon (sa ligne 4 = prefill) ; prédiction −2 à −4 % (q/k/v/o = 33 % des octets d'activation lus, moitié moins d'instructions) | `verdict-p2-moteur-19-09` | inclus dans le bras B ci-dessous |

Scellé **E1** (un seuil, écrit avant) : il existe un réglage `-lgc X ∈ {2100, 2400, 2700}` tel que, au harnais égal (`certifie-b12`, même ctx, `energie.py` 400 W, ABAB avec le bras libre), **J net ≤ 0,2136 ET t/s ≥ 1 066** → **tenu** : la cellule « acvram éco (-lgc X) » entre dans le comparatif, régime dans le nom, et la revendication devient « devant partout, en vitesse et en énergie, contre tout moteur classé ». Sinon **faux** : l'énergie reste le poste ouvert, on le dit. Issue qui gênerait Sage : J descend mais t/s passe sous 1 066 à tout X — alors c'est le noyau, pas l'horloge.

Le réglage `-lgc` n'est **pas** un défaut du moteur : c'est un mode publié à côté du défaut, sous son nom, comme le 14/09.

## 3. Trous du comparatif : GLM b=12

Cellules **vides** (jamais mesurées) : acvram GLM-4.7-Flash b=12 t/s et J, prefill au défaut bf16 (5 502 le 18/09 sous P1). Concurrent : vLLM Marlin 858 / 0,397 / 18 117. Scellé **G1** : b=12 t/s ≥ 858 tenu / < 858 faux ; J ≤ 0,397 tenu / > faux ; deux grandeurs, un seuil chacune. Prédiction : 900-1 100 t/s (MLA bf16 lourd), J 0,30-0,38. Une prise, 45 min.

## 4. File de carte (une seule, Manon la tient ; Océane prend le verrou hors fenêtre par pointeur)

| h | fenêtre | qui | durée |
|---|---|---|---|
| 17 h 15 | **E1** : `-lgc` {2100, 2400, 2700, libre} b=12 Coder ABAB, bras B = régime défaut (la ligne P2 décodage se mesure en même temps : bras défaut vs `PREFILL_INT8=cublas` **au libre seulement**) | Manon | 35 min |
| 18 h | **P2 ligne 2** : PPL i8c prefill 3 tranches (après le correctif d'instrument d'Océane, commité et testé à sec) ; ≤ 1,020 → P2 au défaut, sinon opt-in | Manon | 15 min |
| 18 h 20 | `test_gemv_marlin.py` (6 min) | Océane | 10 min |
| 18 h 30 | **G1** : GLM b=12 t/s + J + prefill défaut | Manon | 45 min |
| 19 h 30 → | fenêtres W4A4 (§ 1) **seulement si** la mesure à sec d'Océane rend < 0,4 % ; sinon la carte reste à Manon pour un second passage E1 fin (X ± 150 MHz autour du meilleur) | Océane / Manon | 30 min chacune |
| 07 h | dernier verdict ; Jérôme met le comparatif et `ETAT.md` au propre pour 08 h | Jérôme | — |

## Ordre

* **Manon** — maintenant, verrou, worktree figé sur `main` 3608e56 (= manon 1fbe843 fusionné, Jérôme 17 h 10) : **E1** (§ 2 : 4 réglages `-lgc` sous verrou, ABAB avec le bras libre, `certifie-b12` + `energie.py` 400 W, attestation lot = 12 ; `-rgc` en fin de fenêtre, vérifié par `nvidia-smi -q -d CLOCK`) ; en-tête : `regime_ligne()`, PID début/fin, load1 ; verdict `revue/verdict-eco-lgc-b12-19-09.md`. Puis, sur pointeur d'Océane : **P2 ligne 2** (PPL i8c prefill 3 tranches, ≤ 1,020), verdict `revue/verdict-p2-ppl-19-09.md`. Puis **G1** (§ 3), verdict `revue/verdict-glm-b12-19-09.md`.
* **Océane** — à sec, maintenant, deux sous-agents en parallèle : (1) `perplexity()` prend le chemin de `Engine.generate` (`layers.py:500` → dispatcher `PREFILL_INT8`, test à sec qui **casse** si la PPL emprunte `int8_matmul` déquant sous `cublas` : compteur `_chemin`), commit sur `oceane-11`, pointeur à Manon ; (2) **mesure à sec de la perte W4A4** : activations réelles de 3 couches de Coder (entrée gate/up et entrée down, 512 jetons de la tranche 1), erreur relative et PPL fausse-quant `ACVRAM_PREFILL_A4=both` sur 1 tranche avec et sans Hadamard/AWQ — scellé : **PPL fausse-quant − PPL défaut ≤ 0,004** → chantier W4A4 ouvert cette nuit (disposition : `GEMV_LAYOUT` non-Marlin, cellules b=1/b=12 remesurées) ; > 0,004 → chantier fermé pour la nuit, dit tel quel, `verdict-w4a4-a-sec-19-09.md`. Puis `test_gemv_marlin.py` sous verrou après la fenêtre P2 de Manon.
* **Jérôme** — fait à 17 h 10 (`verdict-menus-rejeu-19-09` S1-S6 tenus, fusion manon → main 3608e56, sauvegarde-config-ia, REGLES § 6, MECANISMES, Vibe archivé). Reste, à sec, sous-agents : (1) **8081** : `llamacpp-appoint:56` ne résout plus « 3080 Ti » (`--list-devices` à relire, nommer la carte par son index ou son nom exact), relance maintenant — l'appoint tourne sur la 3080 Ti, hors carte 0, il ne change pas le régime de Manon ; `ss -ltn` 3 ports ; (2) demander à l'utilisateur `sudo dpkg -i acvram_0.6.13_amd64.deb` puis `acvram doctor` ; (3) `ETAT.md` : table § 4 de cette note ; revendication du comparatif réécrite à chaque verdict (E1, P2, G1) ; § 1 (borne tensor cores, arithmétique) dans `REPRISE.md` § 10, avec § 2 — une écriture, après le verdict P2 ligne 2 ; (4) règle S6 : « le contrôle "compute-apps identique" est inopérant pendant qu'un pair mesure : comparer les PID hors verrou, pas la liste » → REGLES § 2. Push sur « oui » utilisateur.
* **Sage** — à sec : relit chaque verdict à sa sortie (E1, P2, G1, W4A4) et tranche dans l'heure ; écrit la revendication finale pour 07 h 30 dans `revue/sage-bilan-nuit-20-09.md`.
* Chacun écrit à Sage **une fois par verdict**, pointeur `verdict: revue/<fichier> — 1 ligne`, Jérôme en copie. Pas de suite pytest complète cette nuit ; aucune fusion pendant une fenêtre.

## Addendum 17 h 05 — le `.deb` installé n'est pas le défaut validé, et ses noyaux ne compilent pas

* `sudo dpkg -i acvram_0.6.13_amd64.deb` fait par l'utilisateur (0.6.10 → 0.6.13). Mais ce `.deb` date du **18/09 09:53 = commit `2b9e2f4`**, AVANT `15d3d06` (P1 en défaut, 18/09 23:26) et `d5220ff` : le paquet installé porte le numéro 0.6.13 **sans le régime marlin/marlin contrôlé le 19/09** — un régime caché derrière un nom (REGLES § 4). Et le lanceur ne met le venv à jour que si `VERSION_SRC != VERSION_VENV` (`tools/construire-deb.sh:114-118`) : reconstruire en « 0.6.13 » ne changerait rien à l'installé. **Décision : `.deb` reconstruit en 0.6.14 sur `main` courant**, le nom porte le régime.
* `acvram doctor` : « repli sur les noyaux de référence » — `cuda_fp16.h:4492: fatal error: nv/target` : le venv du paquet compile avec le `nvcc` des roues pip (`nvidia/cu13/bin/nvcc`, 13.4.92) et les en-têtes `nvidia/cu13/include` (runtime 13.0.96) **sans CCCL** ; le venv de développement prend `/usr/local/cuda-13.4` (`kernels/__init__.py:119-125`), qui l'a. Remède vérifié par Sage à 17 h : `pip install 'nvidia-cuda-cccl>=13,<14'` dans le venv du paquet → 13.3.4.3.1, `nvidia/cu13/include/nv/target` présent. **À porter dans `install.sh:56`** : `cuda-toolkit[nvcc,cccl]` (ou `nvidia-cuda-cccl` à côté). La compilation ne se vérifie qu'avec un périphérique (`build_info` rend « aucun périphérique CUDA » sous `CUDA_VISIBLE_DEVICES=""`) : `CUDA_VISIBLE_DEVICES=1 acvram doctor` (3080 Ti) **à une frontière de fenêtre**, jamais pendant E1 — nvidia-smi liste toutes les cartes.
* Ordre Jérôme (s'ajoute) : `install.sh` corrigé + version 0.6.14 + `tools/construire-deb.sh` sur `main`, `pytest tests/test_paquet_charge_utile.py -q` ciblé, `.deb` remis à l'utilisateur pour un second `sudo dpkg -i`, puis `CUDA_VISIBLE_DEVICES=1 acvram doctor` à la frontière E1/P2 : attendu « noyaux CUDA fusionnés » sans alerte.
