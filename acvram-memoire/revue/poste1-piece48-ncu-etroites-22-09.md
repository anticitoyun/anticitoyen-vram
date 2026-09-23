# Pièce 48 — où part le débit à M = 12 : diagnostic ncu préparé (à sec, prédictions avant la carte) — 22/09 (poste1)

## Pourquoi cette pièce existe

Deux mesures indépendantes disent la même chose et aucune ne l'explique :
la 42 (les projections nvfp4 à 0,45 To/s ne battent pas l'int8 sur deux fois
moins d'octets) et la 47 (retirer 384 lancements par pas ne rend que 0,21 ms
sur 0,47 attendus, parce que le GEMV reprend ce que la glue rend). Les deux
pointent la **bande et non les lancements** — mais « la bande » n'est pas un
diagnostic tant qu'on n'a pas dit *quelle* ressource sature. C'est ce que ncu
tranche, et rien d'autre ne le peut.

Repères mesurés, pour que les seuils ci-dessous aient un sens : plancher de
bande utile **1,52-1,55 To/s** (pièce 41) ; nos étroites int8 **0,71-0,82**,
nos étroites nvfp4 **0,44-0,46** (pièce 42), TRT-LLM **1,81** sur les mêmes
octets. Le pic matériel ≈ 1,79 To/s : **0,45 To/s ≈ 25-29 % de la bande.**
La bande n'est donc PAS saturée — ce qui sature est ailleurs, et c'est
exactement la question.

## Les trois bras (≤ 5 min de carte, noyaux en l'état, aucun code modifié)

| bras | noyau | forme servie | alias |
|---|---|---|---|
| **A** | `_dense_etroit_kernel` (nvfp4) | q [4096, 2048] grille 64×6, o [2048, 4096] grille 32×11 | alpha2 |
| **B** | `gemm_etroit` / `etroit_triton` (int8) | les mêmes | officiel |
| **C** | `nvfp4_gemv_marlin_kernel` avec et sans `xscale` | gate+up K=2048 N=768, down K=768 N=2048 | alpha2 puis officiel |

Métriques (ciblées, pas `--set full` : le replay coûte la fenêtre) —
`smsp__issue_active.avg.pct_of_peak_sustained_active`,
`dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.sum`,
`sm__warps_active.avg.pct_of_peak_sustained_active`,
`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` (attente DRAM),
`smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` (attente
shared/MIO), `sm__inst_executed.sum`, `gpu__time_duration.sum`.
Graphes CUDA : `--graph-profiling node` ; si ncu le refuse, le bras est rejoué
`ACVRAM_GRAPHES=0` et **le verdict le dit** (ce n'est plus le régime servi).

## Prédictions et issues, écrites AVANT la mesure

**Sur A et B, quatre issues, mutuellement exclusives par construction :**

| issue | condition ncu | ce qu'elle ouvre |
|---|---|---|
| **I1 — dépaquetage, émission saturée** | `issue_active` ≥ 70 % **et** `dram__throughput` < 50 % | MMA `mxf4nvf4` native + échelles swizzlées : gros chantier, et il ne sert que le nvfp4 |
| **I2 — latence mal couverte** | `issue_active` < 50 % **et** `long_scoreboard` ≥ 40 % | étages + `cp.async` : chantier moyen, **et il profite aussi à l'int8 servi** |
| **I3 — occupation / vagues partielles** | `warps_active` < 30 % du pic **et** blocs par SM < 3 | la forme, pas le noyau : split-K ou tuiles plus larges |
| **I4 — rien à gagner** | `dram__throughput` ≥ 80 % | la pièce 43 n'a pas de levier et il faut le dire : l'écart avec TRT-LLM serait dans les octets comptés, pas dans le noyau |

**Ma prédiction nominale, à laquelle je serai tenue : I2 sur A comme sur B**,
avec `issue_active` 25-45 % et `long_scoreboard` ≥ 45 % sur les deux ; je la
tiens parce qu'à 25-29 % de bande et sans lancement en cause (pièce 47), il
reste la latence. Ce qui la gênerait et que je nomme quand même : si A donne
I1 et B donne I2, alors **le format nvfp4 a bien un coût de dépaquetage
propre** — la 42 s'expliquerait par lui, et « exacte mais sans gain » serait un
verdict sur notre noyau et non sur le format. Et si A ou B donne I4, c'est ma
lecture de la bande depuis la 41 qui est fausse.

**Sur C — la relecture de la table `[E, K]`, explication que j'ai avancée en 47
sans la mesurer :**

* attendu : `dram__bytes` du GEMV gate+up **avec** `xscale` moins **sans**
  ≈ `(N/64) × K × 2 o` par paire, soit **+18 à +25 %** d'octets lus, et une
  durée en hausse du même ordre que les +0,369 ms/pas de la 47.
* **faux si Δoctets < 5 %** : alors la relecture n'est pas la cause du +0,369,
  mon explication de la pièce 47 tombe, et il faut chercher ailleurs (arrondi
  supplémentaire dans la boucle, pression sur les registres).
* ce bras ne coûte presque rien : deux lancements profilés, il se greffe sur A.

## Ce qui rendrait le diagnostic entier inutilisable

`issue_active` et `dram__throughput` tous deux < 30 % **sans** stall dominant :
le noyau serait alors trop court pour que ncu le mesure (10 µs), et il faudrait
un banc à lancements répétés plutôt que le service. Je le dirai au lieu de
choisir l'issue la plus commode.

## Reste

Le script de prise est écrit et versionné ; il ne demande que la carte, après
poste5. Aucun code du moteur n'est touché par cette pièce.

## Prise du 22/09 21 h 55 — **BLOQUÉE** : les compteurs ncu sont refusés à l'utilisateur

* instrument : `outils/gpu/mesure/ncu-etroites-p48.sh 7e5f6b1c`, sous
  `carte.sh` (`ACVRAM_NOM=poste1-p48-ncu`), journaux
  `scratchpad/poste1-p48-ncu/`. Quatre bras lancés, 21:55:47 → 22:00:39,
  **4 min 52 s de carte**. compute-apps début = fin (llama-server 8081 seul).
* **résultat : aucune métrique, sur aucun bras.** `ncu` se connecte au
  processus, le modèle charge, le régime s'imprime — puis :
  `ERR_NVGPUCTRPERM — The user does not have permission to access NVIDIA GPU
  Performance Counters on the target device 0`. Les quatre bras rendent
  `rc=1` et 22 lignes de journal sans une seule mesure.
* cause, vérifiée : `NVreg_RestrictProfilingToAdminUsers` n'apparaît pas dans
  `/proc/driver/nvidia/params` (donc à 1, la valeur par défaut) et
  `/etc/modprobe.d/99-optim-nvidia.conf` ne le pose pas. Le pilote réserve les
  compteurs à root ; **aucune des quatre issues de la pièce ne peut être lue**
  tant que c'est le cas. Ce n'est pas un défaut du script : `nsys` (qui n'a
  jamais eu besoin des compteurs) fonctionne, `ncu` seul est concerné.
* **première prise, 21 h 30, perdue autrement** : `ncu` n'est pas dans le PATH
  des sessions (`rc=127` quatre fois, zéro seconde de carte consommée). Corrigé
  par un chemin explicite et un refus si le binaire manque — et par un bras qui
  dit « BRAS VIDE » au lieu de laisser un CSV de zéro ligne passer pour un
  résultat.
* **correctif appliqué au script** : un micro-lancement `ncu` de trois secondes
  contrôle les compteurs **avant** de prendre la fenêtre, et refuse avec le
  code 77 en nommant le remède. Les cinq minutes perdues ce soir ne peuvent
  plus l'être.

## Ce qui débloque la pièce — décision de l'utilisateur, elle engage la machine

`NVreg_RestrictProfilingToAdminUsers=0` dans `/etc/modprobe.d/99-optim-nvidia.conf`,
puis `update-initramfs -u` et **un redémarrage** (le module `nvidia` ne se
recharge pas à chaud, les services 8081-8083 le tiennent). Alternative sans
redémarrage : lancer la prise sous `sudo`, ce qu'aucune session ne peut faire
seule. Tant que l'un ou l'autre n'est pas fait, la pièce 48 reste **ouverte et
non mesurable** ; les quatre issues et leurs seuils restent valides tels quels.

Ce que la pièce ne doit PAS devenir en attendant : un diagnostic reconstruit
depuis `nsys`. `nsys` donne des durées, pas des taux d'émission ni des raisons
de blocage — les quatre issues se distinguent précisément par ce que `nsys` ne
mesure pas. Un chiffre reconstruit pour combler ce trou serait pire que le trou.
