# Verdict — 184 : le pas ≈90 ms non attribué (181), poste2 25/09

* **instrument** : analyse directe de `srv-trace.jsonl` + `client-T.json` de la prise 181 (poste1, dossier de session
  hors git de la 181), sans relancer la carte (verrou libre) : script à sec sur la trace déjà capturée, aucune
  nouvelle prise nécessaire pour ce point.
* **commit** : worktree `poste2-p184` depuis `origin/main` 00ed964e · **régime** : lecture seule, pas de carte.sh.
* **scellé** : `scratchpad/poste2-p184-25-09/scelle.md` (avant lecture de la trace).
* **prédiction** : (1) capture de graphe CUDA pour nouvelle taille de lot ; (2) retrait de séquence ; (3) compilation
  Triton tardive ; (4) préfill tardif.
* **mesuré** : 3 pas ≈86-91 ms dans la trace (baseline décodage 19,2 ms/pas), répartis sur 2 des 4 lots mesurés
  (conforme à la 181) :
  - **lot 1** (t≈213132,35) et **lot 2** (t≈213146,78) : `run=8 att=0 fin=0`, en plein milieu d'une série de pas
    `run=8` déjà répétée 40 à 190 fois à 19,2 ms — PAS un changement de forme de lot (la forme run=8 est établie
    depuis longtemps quand le pas coûte cher). **Hypothèse (1) graphe CUDA RÉFUTÉE** : une capture n'arrive qu'à la
    première rencontre d'une forme, pas 49 puis 240 pas plus tard sur la même forme.
  - **lot 3/4** (t≈213160,87) : `run=0 att=0 fin=4` — 4 séquences finissent au même pas. Comparé au cas témoin d'une
    seule fin (`run=0 fin=1`, 14,7 ms, t=213125,29, HORS anomalie), le coût scale avec le nombre de fins : ≈19 ms par
    séquence qui finit. **Hypothèse (2) retrait de séquence CONFIRMÉE, mais seulement pour ce pas-là.**
* **reste, non attribué** : les 2 pas `run=8 att=0 fin=0` (lot 1 et 2) ne correspondent à AUCUNE des 4 hypothèses de
  chef — aucun changement d'état (run/att/fin identiques aux pas voisins à 19,2 ms), coût ×4,5 sans cause visible
  dans la trace disponible. Candidat non testé ici (hors instrument 181, qui n'horodate pas le GC ni l'allocateur) :
  pause côté CPU (GC Python périodique, ou trim de l'allocateur caching CUDA) — pas confirmé, demande un profileur
  (py-spy ou torch profiler) sur le process serveur, hors budget de cette pièce.
* **coût chiffré par lot** :
  - type `fin` (retrait, 1 lot/4) : +76 ms sur le lot (91,4 − 15 témoin), soit **1,2 % du lot** (6,29 s).
  - type non attribué (2 lots/4, un seul pas chacun) : +67 à +71 ms (87-90 − 19), soit **1,1 % du lot** chacun.
  - total sur les 4 lots mesurés de la 181 : ≈ 214 ms sur 25,1 s de fenêtre, **0,85 %** — cohérent avec le résidu de
    9,8 ms/lot en moyenne noté par la 181 (ces 3 pas isolés dominent le résidu, le reste des lots est propre).
* **levier** : le type `fin` scale avec le nombre de séquences qui finissent au même pas — la finalisation
  (détokénisation, vérif. chaîne d'arrêt, libération de slots KV) n'est pas vectorisée par lot ; à mesurer si le
  coût est linéaire au-delà de 4 fins simultanées (banc dédié, hors budget ici). Le type non attribué n'a pas de
  levier proposable sans profil CPU — pièce suivante à poser si jugé utile (sinon négligeable, < 1 % par occurrence).
* **issue qui me gênerait** : un des 3 pas se répétant à taille de lot déjà établie et fin=0 — c'est déjà le cas pour
  2 des 3 (lot 1 et lot 2), donc partiellement gênant : un mécanisme reste ouvert, prédiction (1)/(3)/(4) toutes
  réfutées ou sans objet, sans remplaçant confirmé.
* **durée** : prise à sec sur trace existante, < 10 min, aucune carte utilisée.

## Reprise (ordre chef) — prise réelle < 15 min de banc (attente carte hors budget, poste5 179b)

* **instrument** : instrument 181 (`serveur-trace.py`, inchangé) + `ACVRAM_TRACE_STEPS=1` (existant,
  `runner.py:1654-1659`). Rejeu identique (mixte, chat b=8, `-lgc 2700`), `poste2-p184-25-09/prise.sh`.
* **commit** : `poste2-p184` (worktree depuis `origin/main`). **scellé** : `scratchpad/poste2-p184-25-09/scelle.md`
  (prédiction avant). **durée mesurée** : 25,1 s de banc ; la file d'attente du verrou (poste5, 179b) a coûté
  ≈ 10 min hors budget de la pièce elle-même.
* **régime** : `pipeline=1` (`ACVRAM_PIPELINE`, décodage à un pas de retard), `graphes_n=0 captures=0 replays=0`
  — **aucun graphe CUDA engagé cette prise** : le chemin de décodage est `_plain_decode_pipeline`
  (`acvram/engine/pipeline.py:174`), pas `_plain_decode_sync`. Ma prédiction (surcoût dans `avant`/graphe,
  `runner.py:1740-1745`) était **FAUSSE de fait** : ce chemin n'est jamais emprunté sous pipeline actif ; aucun
  graphe n'a été capturé ni rejoué (`captures=0`), donc « rejeu ou capture de graphe » est sans objet ici.
* **rejeu de la même trace** : les 3 pas ≈86-91 ms réapparaissent aux mêmes positions relatives (63, 255,
  dernier), même écart 255−63 = 192 = 3 × 64.

### type `run=8 att=0 fin=0` (2 occurrences, idx 63 et 255, gap exact 192 = 3×64)

Chemin : lot stable → branche `_plain_decode_pipeline` sans recomposition, `acvram/engine/pipeline.py:208-222`
(`_pipeline_suite` + `event.synchronize()` + `_consommer`). Rien dans ce chemin ne dépend de run/att/fin.

**Candidat identifié par la périodicité (192 = 3 × 64)** : `runner.py:1665` (`self._repin_pass()`, appelé à la
FIN de chaque pas) avec cadence par défaut `ACVRAM_REPIN=64` jetons décodés (`runner.py:1677`, `1684`,
`cadence_atteinte` dans `memory/repin.py`) — un gap exactement multiple de 64 sur deux occurrences n'est
raisonnablement pas un hasard. Le corps (`runner.py:1693-1719`) parcourt toutes les couches MoE et fait un
rapatriement hôte (`m._usage_routage.detach().to("cpu").tolist()`, `runner.py:1702-1703`) — un vrai aller-retour
carte→hôte, périodique, indépendant de run/att/fin : cohérent avec le profil observé.
**Non confirmé au bit** : `self._pin` (`runner.py:721`) doit être non vide pour que le corps s'exécute
(`runner.py:1691`, sinon retour immédiat, quasi gratuit) ; `couches_exilées=0/64` dans la ligne de régime ne dit
pas si `_pin` est vide ou juste sans échange à faire. **Test qui tranche, non fait ici (hors budget)** :
`ACVRAM_REPIN=0` sur le même banc → si les deux pas ≈90 ms disparaissent, confirmé ; sinon, cadence 64
coïncidente, à rouvrir.

### type `run=0 att=0 fin>0` (confirmé, mécanisme identifié précisément)

Chemin exact : `acvram/engine/pipeline.py:198-206` — quand la composition du lot pendant change (ici les 4
séquences en vol finissent toutes, `roster_avant` devient vide), `_plain_decode_pipeline` ne fait pas UN pas
mais DEUX empaquetés dans le même `step()` : `pend["event"].synchronize()` + `_consommer(...)` (qui appelle
`_finish` par séquence finissante, `runner.py:1383-1401`, hash des blocs pleins + libération allocator) **PUIS**
`_pipeline_amorcer(...)` (`pipeline.py:108-136`, un décodage complet : `_build_batch`, `graphs.preparer`,
rejeu ou eager) pour amorcer le pas suivant. Le coût n'est donc pas dominé par le nombre de séquences qui
finissent (ma première lecture, 184-a) mais par la **compression de deux pas de décodage en un seul**,
mécanique de la pièce 14/09 (bead runner, doc en tête de fonction `pipeline.py:174-184`).

### chiffrage révisé

* type `run=8/fin=0` (repin, candidat) : ≈ +67 ms, 2 fois par 4 lots mesurés → ≈ 34 ms/lot en moyenne (0,5 %).
* type `fin>0` (double pas, confirmé mécanisme) : ≈ +72 ms, 1 fois par 4 lots (recomposition à la fin d'un lot,
  quand toutes les séquences finissent ensemble — le cas de ce banc, 8 requêtes synchrones) → ≈ 18 ms/lot en
  moyenne (0,3 %). **Ce n'est PAS "jusqu'à 150 ms par lot"** (hypothèse de chef) : un seul pas de recomposition
  par lot ici, pas un par séquence — le double-pas ne se répète pas par séquence finissante, seulement à la
  transition de composition.
* **total mesuré : ≈ 0,8 % du lot**, cohérent avec la 181 (9,8 ms/lot moyen sur 4 lots, dominé par ces 3 pas
  isolés) — **inférieur à tout le hors-calcul de la 181** (question de chef tranchée par la mesure : non).
* **levier** : pour le type confirmé, éviter le double-pas à la recomposition coûterait un redesign du
  pipeline (hors budget, gain < 0,3 %/lot) — non prioritaire. Pour le type candidat (repin), `ACVRAM_REPIN=0`
  l'élimine si confirmé, au prix de perdre le réajustement à chaud du pin (compromis, pas un gain net).
* **issue qui m'aurait gêné** : confirmée en partie — ma prédiction sur le graphe CUDA était fausse par
  construction du chemin (pipeline actif ⇒ jamais `_plain_decode_sync`), pas par une mesure qui l'aurait
  réfutée ; noté pour ne pas prédire sur un chemin sans vérifier au préalable lequel des deux (`sync` vs
  `pipeline`) est actif dans le régime servi.

## CORRECTIF (ordre chef) — « captures=0 » était une lecture fausse

**Erreur** : la ligne `[acvram] régime …` (`captures=0 replays=0`) est un **instantané au démarrage du
serveur**, imprimée UNE fois avant toute requête — je l'ai lue comme un compteur figé pendant tout le banc.
Le log complet (`srv-T.log`) montre 12 lignes `[graphe] capture clé (...)` PENDANT le banc : **les graphes CUDA
sont bien le défaut servi**, `pipeline=1` ne les coupe pas (`graphs.preparer()` reste appelé dans
`pipeline.py:123` et `159`, sous pipeline comme sans). Correction : « rejeu ou capture de graphe » n'est PAS
sans objet — c'est la cause, vérifiée au chiffre.

**Vérification directe, capture ↔ pas lent (même position dans le log)** :

| capture (ligne log) | clé (b, ql, nblk, lb) | durée capture | pas lent apparié |
|---|---|---|---|
| `srv-T.log:86` | (8, 1, 16, 256) | 85,6 ms | idx 63, JSONL : run=8 att=0 fin=0, **86,1 ms** |
| `srv-T.log:288` | (8, 1, 32, 512) | 87,0 ms | idx 255, JSONL : run=8 att=0 fin=0, **87,6 ms** |
| `srv-T.log:1173` | (2, 1, 32, 512) | 70,1 ms | idx 1103 (fin de trace), JSONL : run=0 fin=2, **91,1 ms** |

Écart capture/pas ≤ 0,5-1 ms sur les deux premiers (bruit de mesure), ≈21 ms sur le troisième (recomposition à
2 séquences EN PLUS de la capture, cohérent avec le mécanisme `pipeline.py:198-206` déjà identifié comme
compoundant). **Hypothèse (1) de chef CONFIRMÉE** : `nblk` (le nombre de blocs KV, `graphs.py:645-651`)
grossit avec la longueur de séquence par paliers (`bucket_blocks`, doublement), et la clé de graphe
`(b, ql, nblk, lb)` (`graphs.py:674`) change à chaque palier franchi — même taille de LOT (b=8 stable), mais
**nouvelle forme de TABLE DE BLOCS**, donc nouvelle capture (`graphs.py:684-703`, coût 63-88 ms selon la
`iso177`/observé ici, cohérent).

**Ce que ça change pour le levier** : ce n'est PAS 0,8 %/lot en régime établi — c'est un coût de **chauffe**,
borné par le nombre de paliers `nblk` distincts rencontrés (12 captures sur ce banc de 4 lots × 256 jetons ;
`MAX_GRAPHS=64` plafonne). Sur un service long-courant à contexte stable, ce coût s'amortit à zéro une fois
tous les paliers traversés une fois (cohérent avec « noyaux déjà compilés » de la 81). Le vrai levier, s'il y
en a un, est de PRÉ-CAPTURER les paliers `nblk` probables à l'admission plutôt qu'à la rencontre — pas mesuré
ici, hors budget.

**Retrait** : ma lecture précédente (« aucun graphe CUDA engagé », candidat repin `runner.py:1665`) est fausse,
gardée ci-dessus pour la trace mais remplacée par ce correctif. Le test `ACVRAM_REPIN=0` demandé par chef
n'apporte plus rien vu la correspondance directe capture↔pas (colonne ci-dessus) : je ne le lance pas sauf
contre-ordre.
