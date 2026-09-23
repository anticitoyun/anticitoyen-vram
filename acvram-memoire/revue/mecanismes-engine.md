# Mécanismes à une dimension dans `engine/` — recensement

Lu le 10/09/2026 au soir. Chaque entrée : où, quelle dimension est traitée,
lesquelles ne le sont pas, **et pourquoi** — le pourquoi demande de lire le
code, pas de l'inventorier, et c'est lui qui dit s'il faut boucher le trou.

Deux formes distinctes, à ne pas confondre :

* **ARRÊTÉ** — le mécanisme existe, complet, sur une dimension. Remède :
  étendre, après avoir su pourquoi il s'est arrêté.
* **JAMAIS POSÉ** — aucun mécanisme, et la dimension n'a jamais été
  formulée. Remède : poser. Beaucoup plus lourd, et invisible au recensement
  par `grep`, puisqu'il n'y a rien à trouver.

---

## 1. La clé de graphe `(b, ql, nblk, lb)` — 2 dimensions sur 4 arrondies

`graphs.py:226-239`

    nblk   bucket_blocks(...)     ARRONDI     puissances de 2 depuis 8
    lb     godet_mla(...)         ARRONDI     puissances de 2, hybrides SEULEMENT
    ql     query_lens[0]          non arrondi
    b      batch_size             NON ARRONDI   <- le cas type

**Pourquoi `b` ne l'est pas, et la réponse est bonne :** arrondir `b` exige
d'ajouter des lignes de rembourrage au lot — des séquences factices qui
traversent le modèle sans rien écrire. **Cela demandait une sentinelle que
le chemin d'écriture ne portait pas** : `slot < 0` était gardé par le noyau
CUDA int8 (`acvram_kernels.cu:1831`) et absent du repli PyTorch, où
l'indexation négative reboucle sur le dernier bloc. **Il y avait une raison,
et elle a été levée ce soir** (`kvcache.py`, commit `818d340`).

**Pourquoi `ql` ne l'est pas, et là aussi c'est fondé :** `ql` est déjà borné
par la longueur de spéculation, et les lots à longueurs mixtes sont refusés
d'entrée (`graphs.py:221-222`). L'arrondir ne réduirait pas le nombre de
formes de façon utile. **Ne pas étendre.**

**Pourquoi `lb` n'existe que pour les hybrides :** il vaut `0` partout
ailleurs, donc il ne multiplie pas les formes sur un modèle dense. Correct
par construction.

## 2. `_bind_hybrid` — lié au lot RÉEL, pas au godet

`graphs.py:325-334` : `sids = batch.seq_ids or range(batch.batch_size)`,
puis `for slot, sid in enumerate(sids)`.

Dimension traitée : les emplacements **réellement occupés**.
Dimension non traitée : les emplacements du **godet**, quand `b` sera arrondi.

### PRÉREQUIS BLOQUANT du chantier godets — établi par lecture, pas supposé

La chaîne complète, chaque maillon lu :

    graphs.py:325-334   `for slot, sid in enumerate(sids)`
                        seuls les emplacements REELS sont lies
    model.py:1104-1106  `b = h.shape[0] // q_len` puis `for i in range(b)`
                        le noyau lit `self.statics[i]` sur la taille du
                        TENSEUR — donc sur le GODET des que `b` est arrondi
    gdn.py:150-151      `st["conv"].copy_(conv)` — mutation EN PLACE
    kda.py:187, lfm2.py:63, mamba2.py:121   idem, les quatre familles
    model.py:1074-1075  `store[prev] = la.static_export(self.statics[slot])`
                        au changement de proprietaire, l etat du creneau est
                        EXPORTE vers le magasin sous l id du precedent

**La conséquence n'est donc pas « un état périmé est lu », elle est pire :
l'état d'un emplacement de rembourrage est AVANCÉ avec des entrées
factices, puis EXPORTÉ dans le magasin sous l'identifiant d'une séquence
réelle** — celle qui occupait le créneau avant. La corruption ne reste pas
dans la ligne jetée : elle rentre dans l'état d'une séquence vivante.

**Et elle ne toucherait que les hybrides.** Un essai sur un modèle dense
passerait au vert. Pas de plantage, pas d'exception : des jetons faux et
plausibles, sur les seuls modèles GDN, KDA, LFM2 et Mamba2.

**Avant d'arrondir `b`, l'un ou l'autre, et rien de moins :** lier
`range(godet)` au lieu de `sids`, ou une preuve écrite que `_la_decode` ne
touche jamais les créneaux au-delà de `len(sids)`. La lecture ci-dessus dit
que la seconde est fausse.

Le godet de `static_bind` est, lui, calculé sur `max_model_len` — dimension
traitée correctement.

## 3. La sentinelle `slot < 0` — les deux chemins d'écriture, vérifiés

    acvram_kernels.cu:1831   noyau fusionne int8      gardee (depuis toujours)
    kvcache.py               repli PyTorch            gardee le 10/09

**Recherche des autres consommateurs faite, pas supposée :** hors
`kvcache.py`, `slot_mapping` n'est que **construit** (`runner.py:596`,
`runner.py:840`, `speculative.py:224`, `speculative.py:439`), **copié**
(`graphs.py:355`), **transporté** (`model.py:132`) ou **tracé**
(`graphs.py:364`). Un seul site écrit dans le cache : `model.py:329`, qui
appelle `cache.write`. **Il n'y a pas de troisième chemin** — ni nvfp4 ni
processeur : le repli PyTorch les couvre tous.

`_quantize` (`kvcache.py:349-359`) énumère `int8` puis « tout le reste »
en fp8 — les deux membres de `quantized`, sans trou.

## 4. `graphs.run` refuse le prefill — dimension jamais posée

`graphs.py:219` : `if not self.enabled or batch.is_prefill: return None`.

**Ce n'est pas un mécanisme arrêté, c'est une dimension jamais formulée :**
un lot ne peut pas être moitié prefill moitié décodage parce que
`is_prefill` est **un booléen porté par le LOT** (`model.py:90`), là où
vLLM porte l'information **par séquence** dans `cu_seqlens_q`
(`/tmp/vllm/vllm/v1/attention/backends/flash_attn.py:1040`).

**À marquer à part : le remède n'est pas « étendre », c'est « poser ».** Et
la moitié existe déjà du côté le plus délicat — `model.py:420` fait déjà
`if all(ql == 1 for ql in batch.query_lens)`, donc le chemin de décodage
inspecte **déjà** les longueurs par séquence.

Lecteurs de `is_prefill` : 12 sites, dont **3 seulement décident**
(`graphs.py:219`, `model.py:331` via `is_decode`, `model.py:1428` pour le
MTP). Les propriétés de `ForwardBatch` ont été énumérées une par une —
`batch_size`, `positions_on`, `fixed_decode_views`, `slots_on`, `is_decode`
— une seule dérive de `is_prefill`, et aucun accès par `getattr`.

---

## Ce que ce recensement ne couvre pas

Les cas de forme « jamais posé » **ne se trouvent pas par recensement** :
il n'y a aucune ligne à lire. Le cas 4 n'a été trouvé que parce qu'une
mesure — le +50,7 % du prefill découpé — a demandé une explication. **Un
inventaire ne peut rendre que des mécanismes arrêtés ; les dimensions
absentes se découvrent par la mesure.**

## 5. `carte.sh` : `TYPE` a deux valeurs pour trois usages — JAMAIS POSÉ

`carte.sh:34` : `TYPE=${ACVRAM_TYPE:-mesure}`, qui refuse tout sauf
`mesure` ou `etat`.

    mesure    duree bornee, exclusivite necessaire, on attend derriere
    etat      instantane
    service   PERMANENT — la console, llama-server 8081, embeddings 8082,
              reranker 8083

**Le service n'a pas de valeur qui le décrive, donc il se déclare
`mesure`** — et paraît alors être une manche qui ne finit jamais. Un
arrivant ne peut pas distinguer « attends, ça va se libérer » de « n'attends
pas, c'est permanent ».

Forme : **jamais posé**, pas arrêté. `TYPE` a été écrit quand il n'y avait
que des mesures. Le remède est un ajout — un type `service` qui journalise
et apparaît dans `qui_tient`, mais **qu'un arrivant sait ne pas devoir
attendre** — pas une extension.

Constaté le 10/09 à 21h20 : la 3080 Ti tenue depuis 43 minutes par
`jerome-console-gui`, déclarée `mesure`, **et c'était voulu**.

## 6. Le verrou disait le présent, jamais le passé — POSÉ le 10/09

`carte.sh:125-126` écrivait `$INFO` par `>` et le supprimait au `trap EXIT`.
Le fichier répondait à « qui tient la carte MAINTENANT », jamais à « qui la
tenait à 17h53 » — **aucun `>>` dans tout le fichier.**

La question s'est posée parce que deux journaux jumeaux du chantier quota,
datés 17:53 et 17:54, portaient une étendue de 10,4 % là où la paire de
16:18 en portait 0,15 % : **une occupation non identifiée, dont le verrou ne
gardait aucune trace.**

Journal `>>` posé le 10/09 : une ligne par prise, une par restitution avec
la durée tenue.

**La règle que l'incident a produite :** *une mesure doit publier sa durée
ET exister en au moins deux exemplaires du même travail. Une seule des deux
conditions laisse l'indécidable intact.* Corollaire : **un témoin unique
n'est pas un témoin.**

## 7. SmoothQuant par canal casse l'échelle de bloc NVFP4 — MESURÉ le 13/09

Bead `anticitoyen-vram-brd` : lisser les activations avant fake-quant E2M1
(`s_j = max|X_j|^α / max|W_j|^(1-α)`, par CANAL d'entrée j) DÉGRADE le PPL
de façon MONOTONE avec α sur Llama-2-7B (α=0,50 → +4,98 %, α=0,65 →
+7,29 %, α=0,80 → +24,69 %) — **pire que l'absence totale de lissage**
(A4 nu : +2,58 %). Voir `revue/verdict-smoothquant-a4-sweep.md`.

**Le mécanisme (hypothèse, cohérente avec la monotonie observée, non
vérifiée noyau par noyau) :** le NVFP4 quantifie par BLOC DE 16 canaux
d'entrée consécutifs, avec UNE SEULE échelle E4M3 partagée par bloc — un
canal isolé dans ce bloc ne pénalise QUE ce bloc, à coût quasi nul pour le
reste du tenseur (c'est déjà la raison documentée du choix du bloc de 16,
`acvram/quant/nvfp4.py`). SmoothQuant calcule une échelle *par canal
individuel*, sans respecter cette frontière de bloc : un canal fortement
mis à l'échelle par `s_j` peut désormais dominer l'`amax` du bloc entier,
écrasant la résolution des 15 canaux voisins qui n'avaient pourtant pas
besoin d'être touchés. Plus α est grand, plus `s = max|X|^α` s'écarte
d'un canal à l'autre (l'activation varie fortement, le poids beaucoup
moins), et plus ce déséquilibre intra-bloc s'aggrave — d'où la
dégradation monotone.

**Ce que ça dit pour tout mécanisme futur qui touche à l'échelle d'un
format à bloc fin :** une transformation qui traite les canaux
individuellement (rotation, lissage, permutation) doit soit respecter la
frontière du bloc de quantification (une transformation PAR BLOC plutôt
que sur le tenseur entier), soit être évaluée avec la MÊME rigueur que
SmoothQuant ici — un gain démontré sur un format à groupe large (l'INT4-AWQ,
groupe 128, où la même rotation Hadamard AIDE : `piste-hadamard-downproj-refutee.md`)
ne se transporte PAS automatiquement à un format à bloc de 16.
