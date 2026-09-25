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
