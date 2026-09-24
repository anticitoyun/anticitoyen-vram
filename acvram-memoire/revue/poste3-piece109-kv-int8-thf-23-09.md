# Pièce 109 — KV int8 servi diverge du jumeau (anticitoyen-vram-thf) — EN COURS, pause du groupe

Ordre (chef) : `acvram_kernels.cu:4217/4220/4223` calcule `m/127.f` (division approchée sous
`--use_fast_math`) puis `x·(1/sc)`, alors que le jumeau `kvcache.py` fait `x/scale` (IEEE).
Correctif : `__fdiv_rn` et division directe, comme la pièce 104. Puis : test au bit noyau =
jumeau qui casse sur l'ancien code, puis KL b=1 ≤ 0,74 et lot mêlé sur carte, en fin de file,
seuils scellés avant. **Pause du groupe demandée avant la fin de la validation : livré dans
l'état, testé partiellement, la suite reste ouverte.**

## Fait

- `acvram/kernels/acvram_kernels.cu` : deux correctifs, PAS un seul —
  1. `kv_write_int8_kernel` (`:4190`, chemin par jeton servi par défaut pour K **et** V) :
     `red[0] = fmaxf(__fdiv_rn(m, 127.f), 1e-8f)` puis `__float2int_rn(__fdiv_rn(x, sc))`
     (plus de `inv = 1.f/sc` multiplié).
  2. `kvc_quant_par_jeton` (`:4415`, trouvée en lisant : COPIE du même calcul pour V du chemin
     canal C5-b — portait le même défaut, non nommé par chef mais du même ressort). Sa
     voisine `kvc_quant_bloc_canal` (K du chemin canal) était déjà juste (`__fdiv_rn`, commentée
     comme telle) : l'incohérence entre les deux fonctions voisines est le même genre de faute
     que le ticket signale.
- `tests/test_kv_write_int8_thf_carte.py` : deux tests, sur carte.
  - `test_ecriture_int8_par_jeton_au_bit_contre_le_jumeau` (un tirage, graine 1) : **VERT**,
    K et V au bit contre `PagedKVCache._quantize`, échelles au bit.
  - `test_plusieurs_tirages_toujours_au_bit` (5 graines, 100-104) : **ROUGE, un seul cas** —
    graine 103, V seulement (K identique sur les 5) : `assert False` sur `torch.equal`.

## Pas fait — le désaccord n'est PAS expliqué, à rouvrir

Deux mesures contradictoires sur EXACTEMENT le même tirage (graine 103, mêmes k/v vérifiés
identiques par construction — `torch.manual_seed` déterministe) :
- Dans la boucle des 5 graines (4 `PagedKVCache` créés avant lui) : **V diverge**.
- Isolé, dans un script neuf, seed 103 posée directement (`scratchpad/poste3-thf-23-09/diag.py`,
  rejoué deux fois, y compris une seconde fois DANS le même run que le test rouge, sans
  recompilation) : **0 écart, échelles égales**.

Donc le désaccord n'est pas dans la formule (le correctif tient à l'isolement) : quelque
chose lié à L'ÉTAT ACCUMULÉ par les créations/destructions précédentes de `PagedKVCache`
dans la boucle change le résultat du MÊME calcul sur les MÊMES données. Hypothèses nommées,
aucune vérifiée :
1. **Réutilisation mémoire de l'allocateur CUDA** (cache PyTorch) : les 3 instances
   précédentes libérées avant la 4e pourraient laisser une adresse partagée dans un état
   qui influence un noyau qui, lui, ÉCRIT tout — improbable mais pas exclu (pointeurs
   non alignés différemment ?).
2. **Un noyau voisin de l'extension** (compilation JIT, lancement) laisse un état de flux
   CUDA ou un `cudaStreamSynchronize` implicite différent selon l'historique du process.
3. Script de diagnostic supplémentaire écrit mais NON JOUÉ (pause du groupe) :
   `scratchpad/poste3-thf-23-09/diag2.py` — rejoue la boucle et capture le détail de l'écart
   dans SON PROPRE contexte (pas un script à part), rejoue l'écriture deux fois de suite sur
   les mêmes données (teste la reproductibilité intra-process), compare k/v de la boucle à
   ceux d'un tirage isolé (déjà confirmés identiques par construction, à re-vérifier par le
   code plutôt que par le raisonnement).
4. Pas exclu : un défaut PRÉEXISTANT (avant ce correctif) déjà présent dans le chemin V —
   le test C5-b (`test_kv_canal_c5b_carte.py:8`) tolérait déjà « V … à ± 1 code » comme un
   fait accepté, jamais expliqué non plus.

**Le correctif de code (division `__fdiv_rn`) reste posé** : il est correct par lecture (même
arithmétique que le jumeau et que le C5-b déjà juste) et TENU sur le test à un seul tirage ;
mais le test à cinq tirages n'est PAS vert, donc **la sortie par défaut ne doit PAS changer**
tant que ce désaccord n'est pas expliqué — aucune KL ni lot mêlé n'a été mesuré (jamais
atteint le tour de file).

## Reste, dans l'ordre, à la reprise

1. Jouer `diag2.py` (déjà écrit, jamais lancé) pour voir le détail de l'écart DANS son
   contexte reproductible.
2. Si confirmé lié à l'allocateur/état du process : isoler avec
   `torch.cuda.empty_cache()` entre les itérations de la boucle, ou par un test à processus
   séparé par graine (`subprocess`), pour trancher entre allocateur et autre chose.
3. Une fois expliqué et le test à 5 tirages vert : KL b=1 ≤ 0,74 et lot mêlé (REGLES § 1),
   seuils à écrire avant la mesure, en fin de file carte.
4. Tant que 1-3 ne sont pas faits : le correctif reste un candidat lu et partiellement
   prouvé, PAS un changement de défaut servi.

## diag2.py joué (23/09 22 h 43, reprise, ordre chef) — allocateur écarté, hypothèse changée

- instrument : `scratchpad/poste3-thf-23-09/diag2.py`, sur carte, `outils/carte.sh` tenu
- commit : `7000a101` (worktree `poste3`, fusion `a6826809` avant prise)
- régime : plein (compteur remis à zéro, REGLES § 1)
- scellé : aucun avant cette prise — étape de diagnostic, pas de mesure publiée
- mesuré : boucle des 5 graines rejouée deux fois de suite (même run) : rejeu identique
  déterministe intra-process (« deux écritures identiques : True ») ; MAIS l'écart change
  de graine d'un run à l'autre — le test unitaire d'hier situait l'écart en graine 103, ce
  rejeu le situe en graines 101 et 104 (103 est ici sans écart). Sens de l'erreur non
  systématique : cas 1 (t=2,h=0,d=103) `x/sc=-63,515…` → arrondi correct = -64, **obs=-63
  (faux)** ; cas 2 (t=14,h=0,d=108) `x/sc=63,517…` → arrondi correct = 64, **obs=64
  (correct), ref=63 (faux, côté Python/_quantize)**. Les deux cas sont à un code de la
  frontière d'arrondi (`x/sc` à ± 0,02 de la demi-unité), jamais ailleurs.
- verdict : PAS l'allocateur (hypothèse 1 écartée — rejeu intra-process reproductible au
  bit, donc pas un état mémoire non initialisé qui varierait entre deux écritures
  identiques). L'écart change d'un run à l'autre pour LA MÊME graine : ni un biais fixe du
  noyau CUDA (`__fdiv_rn`), ni un défaut de la référence Python seuls — les deux camps se
  trompent une fois chacun, toujours à la frontière d'arrondi. Hypothèse restante la plus
  probable, NON vérifiée : ordre de sommation/FMA non déterministe du noyau (registre vs
  mémoire, fusion multiply-add) selon l'état du JIT/cache entre runs — à distinguer d'un
  bruit de seed en isolant chaque graine dans un process séparé (`subprocess`, point 2 du
  plan déjà écrit). Le correctif de code reste posé, la sortie par défaut ne change pas
  (test à 5 tirages toujours rouge, cause non expliquée).
- durée : prévue 10 min, tenue ~8 min (une commande, un rejeu)

## diag3.py joué (24/09 02h08, ordre chef) — isolation par sous-processus : allocateur ET race JIT écartés

- instrument : `scratchpad/poste3-thf-23-09/diag3.py` (une graine par process, invoqué en boucle bash par
  `lancer_diag3.sh` — 5 processus python séparés, pas un seul boucle intra-process), sur carte, `outils/carte.sh`
  tenu deux fois (deux prises distinctes, cf. incident guet/.qui hors sujet, résolu par le chef — corrigé dans
  tests/test_gui_crochet_clic.py, sans rapport avec ce diagnostic).
- commit : d11f2dc4 (worktree `poste3`, fusion origin/main avant prise)
- régime : plein
- mesuré : 1er run — compilation de l'extension déclenchée (« verrou de compilation orphelin retiré »),
  puis graine=100 0 écart, graine=101 **1 écart** (t=2,h=0,d=103 : x/sc=-63,515 → arrondi correct -64, obs=-63),
  graine=102 0 écart, graine=103 0 écart, graine=104 **1 écart** (t=14,h=0,d=108 : x/sc=63,517 → arrondi correct
  64, obs=64 correct MAIS réf Python=63 fausse). 2e run (prise séparée, extension déjà compilée, pas de
  recompilation) : **résultat identique au bit** — mêmes deux graines (101, 104), mêmes t/h/d, mêmes valeurs
  observées et de référence.
- verdict : hypothèse 1 (allocateur CUDA) et hypothèse 2/3 (race JIT/flux CUDA entre process) **ÉCARTÉES** — des
  processus complètement séparés (nouveau contexte CUDA, nouvel import, memoire vierge) donnent exactement le
  même écart, aux mêmes positions. Ce qui varie n'est PAS le runtime : c'est le **binaire compilé** du noyau
  (`.so` figé une fois construit) qui fixe, de façon déterministe, lesquelles des 16×4×128 cases tombent du
  mauvais côté d'une frontière d'arrondi à ±0,02 de la demi-unité — cohérent avec la variabilité observée ENTRE
  sessions différentes (graine 103 dans le tout premier test, 101/104 dans le rejeu diag2 du 23/09 22h43 et ici :
  des BUILDS différents du même noyau, chacun déterministe en soi). Reste non expliqué : QUEL détail de
  compilation (ordre FMA, `--use_fast_math`, registre vs mémoire) fixe ce choix — piste suivante si le chef
  demande d'aller plus loin (comparer le PTX/SASS de deux builds qui divergent différemment). Le correctif de
  code reste posé, la sortie par défaut ne change toujours pas (toujours 1 cas sur 5 tirages, jamais 0/5).
- durée : 1er run ~7 min (dont 387 s d'attente carte), 2e run ~17,5 min (dont 1047 s d'attente carte,
  incident .qui hors sujet pendant la file) — deux prises séparées, chacune < 30 min.

## Prédiction écrite AVANT mesure (24/09, ordre chef, diag4.py — MÊME entrée × 5 process séparés)

- protocole : graine=101 FIXE (divergente en diag3), 5 processus python complètement séparés (même
  `.venv`, même `.so` déjà compilé, aucune recompilation attendue), chacun calcule K/V quantifiés et
  imprime le sha256 de (qv, sv, qk, sk). `set -euo pipefail`, à sec côté logique (pas d'aléa non
  contrôlé — seed fixée).
- prédiction : les 5 sha256 seront **IDENTIQUES** (5/5). Motif : diag3 (24/09 02h08) a déjà montré deux
  runs complets séparés produisant le MÊME écart aux mêmes positions pour les graines 101 et 104 — signe
  que le résultat est fixé par le binaire `.so` compilé, pas par un aléa d'exécution (allocateur, ordre
  de lancement, état de flux CUDA). Si les 5 sha256 sont identiques : CONFIRME hypothèse « fixé par le
  build », le test à 5 tirages restera 1/5 rouge de façon stable (pas un flake). Si un seul sha256
  diffère des 4 autres : RÉFUTE, il existe bien un aléa d'exécution malgré le même `.so` (à creuser :
  ordre de lancement de blocs, non-déterminisme atomique/registre du noyau).
- seuil de réfutation : 1 sha256 différent parmi 5 suffit à réfuter le déterminisme par build.

## Mesuré (24/09, ordre chef, carte.sh poste3-p109, tenue=5s, 1030s d'attente derrière poste4/poste6)

- 5/5 sha256 **IDENTIQUES**. Prédiction CONFIRMÉE.
- verdict : le résultat de la quantification V (graine=101, même entrée, même `.so`) est **déterministe au
  bit** entre 5 processus complètement séparés. Aucun aléa d'exécution (allocateur, ordre de lancement,
  flux CUDA) — cause définitivement fixée par le binaire compilé, pas par le runtime. Le test à 5 tirages
  restera stable à 1/5 rouge (pas un flake), tant que le `.so` ne change pas. Hypothèses 1, 2 et 3 du plan
  initial toutes ÉCARTÉES avec certitude désormais.
- reste (hors domaine de cette pièce, pour poste7/le chef) : identifier QUEL détail de compilation
  (ordre FMA, `--use_fast_math`, registre vs mémoire dans `__fdiv_rn`) fixe le choix d'arrondi à la
  frontière — nécessite lecture PTX/SASS, pas engagée ici (hors plan initial, à commander explicitement).
  Correctif de code toujours posé, sortie par défaut toujours inchangée (1/5 rouge stable et expliqué,
  pas une régression cachée).
- branche `poste3` reste NON fusionnée tant que le test est rouge (ordre chef).

## Localisation (24/09, ordre chef, diag5.py, carte.sh poste3-p109-localise, tenue=1s, 356s attente)

- même cas (graine=101, V, t=2 h=0 d=103), recomposé étape par étape en fp32 (précision exacte du
  noyau ET de `_quantize`, PAS le float64 des diagnostics précédents qui n'est l'arithmétique d'aucun
  des deux camps).
- **amax** = 16,375 (fp32 exact, hex `41830000`).
- **échelle** (amax/127, fp32, AVANT le cast fp16 de stockage) = 0,1289370059967041 (hex `3e040810`).
  Stockée en fp16 dans les deux camps : 0,12890625 — identique, ce n'est PAS là que ça diverge (le
  cast fp16 est une compression de stockage, la division de quantification utilise l'échelle fp32
  complète des deux côtés, lu au bit dans `kv_write_int8_kernel` : `sc = red[0]` avant tout cast).
- **ratio = x/échelle** (fp32, `_quantize`, torch) = **-63,5 EXACTEMENT** (hex `c27e0000`) — une vraie
  égalité à mi-chemin, pas une approximation de fin de calcul (confirmé : `ratio*2 == round(ratio*2)`).
  `torch.round()` (ties-to-even) tranche vers **-64** (pair) → `ref=-64`, cohérent avec IEEE 754.
- **le noyau rend -63.** Si son ratio interne était, lui aussi, exactement -63,5, `__float2int_rn`
  (ties-to-even par la norme PTX `cvt.rn.s32.f32`, même convention que torch) donnerait -64 aussi — pas
  -63. Le noyau ne peut donc PAS calculer exactement -63,5 en interne, malgré la même formule apparente
  (même amax, même division `x/échelle` en `__fdiv_rn`) : quelque chose déplace son ratio hors de
  l'égalité stricte, côté noyau seulement.
- **verdict : PAS « test trop strict ».** C'est une divergence réelle et reproductible à une égalité de
  mi-chemin exacte — le test a raison de la voir. La cause n'est PAS le seuil du test ni l'arrondi de
  `_quantize` (IEEE correct, cohérent). Reste à prouver en PTX/SASS (hors cette pièce, à commander) :
  la piste la plus probable est `--use_fast_math` contractant une FMA autour du `__fdiv_rn` ou de la
  réduction `fmaxf` d'amax malgré l'intrinsèque explicite (le commentaire du fichier `:4225-4229` déjà
  fixé UNE fois ce problème pour la division de quantification — peut-être un deuxième site non couvert,
  ou une contraction du compilateur qui déjoue l'intrinsèque localement).
