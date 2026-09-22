# Préparation de la livraison 0.6.36 (poste4, 22/09, à sec)

Document de préparation, sur le modèle de `livraison-0.6.35.md` (poste3), indépendant
des chiffres finaux. Contient : (1) le journal des changements depuis 0.6.35, (2) le
gabarit de la section « Résultats mesurés » à reporter dans README.md + les 31
traductions, avec les chiffres tenus aujourd'hui et `__B12_NVFP4__` laissé pour la
cellule de l'alias nvfp4 en cours, (3) la procédure de livraison .deb + GitHub, écrite
et NON lancée. Source : `git log 665eeacc..main --no-merges` (622 commits, sous-liste
retenue ci-dessous).

## 1. Journal des changements — 0.6.36

### Qualité et sécurité de sortie (« replis non muets »)
- **Pièce 38 : deux replis silencieux rendus non muets** (sortie au bit inchangée) —
  un chemin dégradé ne s'active plus sans laisser de trace dans le journal ou le
  résultat ; la valeur produite reste identique au bit, seule l'absence de mention
  disparaît. (538781f1)
- **Graphes CUDA : gardes contre l'interblocage de capture** — refus nommé sous un
  seuil de mémoire disponible, et alerte + abandon propre au-delà d'un délai de
  capture, plutôt qu'un blocage silencieux du processus. (fd866fe6)

### Conversion et quantification
- **Pièce 32 : observations par expert à la calibration** — refus nommé (code
  dédié) quand un expert reçoit moins de 512 observations pendant la calibration
  AWQ, réglable par `--obs-min` ; remplace un silence qui laissait passer des
  échelles mal calibrées. (90c371f7)
- **Pièce 35 : occupation des noyaux étroits int8, AU BIT** — forme du noyau revue
  pour les GEMV étroits (occupation par SM) ; l'hypothèse split-K est RÉFUTÉE au
  banc et son crochet est retiré (aucun gain mesuré, code mort ôté). (19a10889)

### Qualité par logprobs (pièce 36, en deux temps)
- **Moteur** : top-K logprobs par pas et logprobs de l'invite (echo), sans toucher
  la sortie par défaut au bit. (4c6fe61c)
- **Serveur** (à venir, pas encore lancé) : `CompletionChoice.logprobs`
  (tokens/token_logprobs/top_logprobs/text_offset) depuis `GenerationOutput`, echo
  via `logprobs_invite`, sortie au bit inchangée sans logprobs demandés, avertit si
  top-K est demandé sous un graphe CUDA capturé (chemin non couvert). Tests écrits
  (sans-logprobs identique, logprobs+echo, top-K CPU) mais NON LANCÉS — garde CPU
  posée pendant la mesure de poste2, en attente de carte libre. (c96d0b47)

### Tests et cas limites
- **Pièce 31 : cas limites du levier 1 (sampler glouton dans le graphe), au bit** —
  lot partiel (1, 5, 12) et plein (16) avec fantômes du godet jamais lus, slot EOS
  réadmis au pas suivant (la nouvelle séquence reçoit SON jeton, pas celui du
  slot précédent), séquence finie dans le lot → ancien chemin (repli), température
  > 0 avec lot mêlé → ancien chemin avec tirage reproductible, égalité vérifiée sous
  `CUDA_LAUNCH_BLOCKING=1` (sous-processus dédié). 21 tests voisins verts. (d4d23c1a)

## 2. Gabarit « Résultats mesurés » 0.6.36 (chiffres tenus aujourd'hui, nvfp4 en attente)

À reporter dans README.md ET les 31 `docs/README.<code>.md` (même structure : ne pas
changer le nombre de lignes de tableau ni de titres — cf `test_readme_traductions`).
`__B12_NVFP4__` reste à remplir par la cellule officielle de l'alias nvfp4 en cours ;
tout le reste ci-dessous est déjà mesuré et sourcé (protocole 2.5, ≥ 20 s/fenêtre).

```
## Résultats mesurés (JJ/09/2026, RTX 5090 à 400 W, régime ≥ 20 s au compteur d'énergie)

Débit TensorRT-LLM 1.3.0rc15 par taille de lot (protocole 2.5, Qwen3-Coder-30B-A3B) :

| b | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| jetons/s | 46 | 49 | 51 | 51 | 1 998 |

Énergie, quatre moteurs, b=12, même carte, même protocole NVML (J/jeton net) :

| | TensorRT-LLM | vLLM | acvram | llama.cpp |
|---|---|---|---|---|
| J/jeton net | **0,154** | 0,163 | 0,197 | 0,206 |

acvram nvfp4 (alias en cours) : **__B12_NVFP4__ t/s**, cellule à venir.

KL acvram/bf16 (référence locale, decode-pas, Qwen3-Coder-30B-A3B) : 5/5 fenêtres,
max 0,519. KL TensorRT-LLM/bf16 : indicatif seulement (API Python TRT-LLM, pas
encore scellé au même protocole que le bras acvram).
```
Prose : la courbe TRT-LLM confirme un régime dominé par la latence jusqu'à b=8 (débit
quasi plat 46-51 t/s) puis un saut au plafond de lot b=12 (1 998 t/s) — cohérent avec
un noyau borné par la latence mémoire à faible occupation (cf pièce 35, Q(10) duck.ai
22/09). L'énergie classe les quatre moteurs dans le même ordre qu'au 5e passage
(rejugé sans nouvelle prise, sha 2ea1779f) : TensorRT-LLM < vLLM < acvram < llama.cpp,
écart acvram/TensorRT-LLM ≈ +28 %, piste ouverte (Q(9)/Q(10) duck.ai : fusion tête +
échantillonnage, occupation GEMM étroits).

## 3. Procédure de livraison 0.6.36 + GitHub (ÉCRITE, NON LANCÉE)

Prérequis : `__B12_NVFP4__` connu, main figé sur le SHA de livraison, feu explicite
de l'utilisateur pour les étapes 6-7 (publication GitHub — jamais de sa propre
initiative).
1. Bumper la version : `acvram/__init__.py` → `__version__ = "0.6.36"` (source unique,
   actuellement encore à 0.6.35).
2. Reporter le tableau ci-dessus (avec `__B12_NVFP4__` rempli) dans README.md + les 31
   traductions, dater, vérifier `test_readme_traductions` vert (structure identique).
3. Construire le .deb sur le SHA de livraison :
   `bash tools/construire-deb.sh` (worktree détaché sur ce SHA) → `acvram_0.6.36_amd64.deb`.
4. Relever le sha256 : `sha256sum acvram_0.6.36_amd64.deb` (écrit dans un fichier,
   jamais recopié en clair au terminal — crochet secrets).
5. Vérifier le paquet : `dpkg-deb -f … Version` = 0.6.36 ; Homepage + « Soutenir »
   buymeacoffee présents ; extra vision (transformers 5.17.0 + pillow) dans le wrapper.
6. Publier l'instantané public GitHub : `outils/publier-github.sh --pousser`
   (GitHub = instantané sans historique ; FEU EXPLICITE requis avant cette étape et
   la suivante).
7. Release GitHub v0.6.36 : joindre le .deb, coller le journal des changements
   (section 1), le sha256 en pied.

Rien de tout ceci n'est lancé ici : ce document est la procédure ; l'exécution
attend `__B12_NVFP4__` + le feu de l'utilisateur pour la publication GitHub.
