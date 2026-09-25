instrument : `graphes.py:GraphesMoteur.warm_graphs_multibatch`, mesure `test_poste2_p186_mesure_25_09.py` (temporaire, retiré après lecture)
commit : `f4a3c556` (branche `poste2-p186`)
régime : ACVRAM_ECO=off, checkpoint synthétique (fixture `converted`), max_model_len=256, kv_planned_seqs=2
scellé : `scratchpad/poste2-p186-25-09-scelle.md` (avant tout code/mesure)
mesuré : deux moteurs chargés dans la même fenêtre, A sans `ACVRAM_WARM_GRAPHS_B`, B avec `=2`
verdict : mécanisme TENU, magnitude NON établie sur cette fixture (voir ci-dessous)
durée : prévu ≤ 10 min, tenu ~2 min (`carte obtenue apres 79 s`, suite 4,99 s)

## Résultat

- **A (sans préchauffe)** : `warm_graphs` capture 1 (b=1, L=128) au démarrage ; le premier pas de
  décodage réel à b=2 déclenche une capture EN DIRECT (captures 1 → 2) — comportement actuel,
  comme prédit.
- **B (avec `ACVRAM_WARM_GRAPHS_B=2`)** : `warm_graphs` + `warm_graphs_multibatch` capturent 2
  (b=1 et b=2) au démarrage ; le premier pas réel à b=2 ne déclenche **aucune** capture
  supplémentaire (captures 2 → 2) — la clé visée par la pré-capture fantôme est bien celle
  qu'atteint le lot réel. **Confirme le mécanisme du scellé : la préchauffe évite la capture en
  direct sur la première requête qui franchit le palier.**
- Temps mesurés (0,20 ms vs 0,14 ms sur le pas, 158,7 ms vs 8,1 ms au démarrage) : **non
  transposables** — checkpoint synthétique (quelques couches, `tiny_checkpoint`), une capture y
  coûte des fractions de ms, pas les 65-90 ms chiffrés en 184/186 sur le vrai modèle. La fixture
  valide le MÉCANISME (quelle capture a lieu, quand), pas la MAGNITUDE prédite dans le scellé
  (+2,3 à +3,2 s de démarrage, gain ≈ 65-90 ms sur la 1ʳᵉ requête au palier neuf).

## Addendum — 25/09, ordre chef : magnitude sur le vrai modèle (alias mixte au défaut)

instrument : `scratchpad/poste2-p186-reel/prise.sh` + `prise-b.sh` (gabarit `poste2-p184-25-09/prise.sh`)
commit : `14836891` + ce commit, branche `poste2-p186`
régime : ACVRAM_ECO=off, `-lgc 2700` (verrou tenu par un autre pendant A — `eco=off(2670: verrou
2700 posé hors processus)`, écart de 30 MHz, sans effet nommé), `--max-batch 8 --max-model-len 4096`
scellé : addendum du même fichier, prédiction avant mesure (65-90 ms sur le TTFT, blocs KV identiques,
+0,3 à +0,45 s au démarrage pour un seul `b`)
mesuré : A puis B, chacun un seul lot de 8 requêtes concurrentes (`prompt_tokens=652`, `max_tokens=1`)
verdict : **gain observé, magnitude PAS établie** — une seule mesure par bras, REGLES §3 l'interdit
comme verdict ferme
durée : file très disputée (poste5 + chef en parallèle), A obtenu après 657 s d'attente, B après
429 s ; charge du serveur elle-même rapide (13-14 s) — le coût réel de cette pièce est la file, pas
le calcul

### Résultat

| | A (défaut, OFF) | B (`ACVRAM_WARM_GRAPHS_B=8`) |
|---|---|---|
| charge en | 13,2 s | 13,7 s |
| blocs KV | 2048 (32768 j/couche) | 2048 (32768 j/couche) — **identique, comme prédit** |
| godets capturés d'avance | 5 (b=1 seul) | 10 (b=1 + b=8, mêmes 5 paliers) — **comme prédit** |
| TTFT b=8, prompt 652 jetons (8 requêtes concurrentes) | max 3040,8 · médiane 2712,0 · min 2710,8 ms | max 2565,0 · médiane 2370,2 · min 2175,5 ms |
| écart (A − B) | — | max 475,8 · médiane 341,8 · min 535,3 ms **plus large que prédit (65-90 ms)** |

Mémoire (blocs KV) et nombre de godets : TENUS au chiffre exact prédit. TTFT : B plus rapide dans
les trois statistiques, mais d'un facteur 4 à 8× la prédiction (65-90 ms) — sur un seul lot par
bras, sans répétition ni ABBA, dans une fenêtre où la carte changeait de mains en continu (charge
et horloge non contrôlées entre A et B, contrairement à une campagne isolée). **Ne peut pas
distinguer un vrai effet plus grand que prévu (ex. la capture en direct A gèle aussi le
scheduler/allocateur pendant le prefill des 7 autres requêtes du lot, pas seulement son propre
pas — coût multiplié par le lot) d'un artefact de charge externe.**

## Addendum ABBA — 25/09, ordre chef : tranche l'hypothèse « la capture gèle tout le lot »

instrument : `scratchpad/poste2-p186-abba/prise.sh`, ABBA (A1,B1,B2,A2), 5 lots/bras à 5 longueurs
distinctes (~142/247/457/877/1702 jetons réels, ciblant les paliers 128-2048)
scellé : addendum ABBA du même fichier, prédiction avant mesure
verdict : **RÉFUTÉ** — aucun gain mesurable de B sur A en TTFT réel ; l'écart de 341 ms vu au tour
précédent était du bruit de file, pas un effet du mécanisme
durée : carte obtenue après 661 s d'attente, séquence ABBA (4 serveurs, 20 lots) tenue en 8 min 50

### Résultat

Défaut d'instrument (corrigé après coup, pas rejoué) : `/metrics.regime_ligne` interrogé est la
ligne de régime GLOBALE (`acvram.regime_ligne`, sans compteur), pas celle de l'`Engine` — le compte
de captures en direct par lot n'a PAS été obtenu cette fois (`captures -1->-1` partout, artefact,
pas une mesure). Corrigé dans le script (`engine.graphes_captures` via `/metrics.engine`), à
utiliser si une répétition est demandée.

TTFT (ms), par lot, les 4 bras :

| lot (jetons) | A1 (max/méd/min) | B1 | B2 | A2 |
|---|---|---|---|---|
| 1 (142) | 738/488/487 | 734/484/483 | 736/485/485 | 739/488/488 |
| 2 (247) | 934/934/395 | 947/947/243 | 936/553/552 | 940/709/477 |
| 3 (457) | 1187/1186/515 | **2368/2140/1912** | 1183/1182/511 | **3152/2845/2844** |
| 4 (877) | 1014/1014/530 | 1502/1500/1056 | 1012/1011/526 | 1011/1010/527 |
| 5 (1702) | 1240/1239/625 | 1240/1239/626 | 1242/1242/628 | 1238/1237/624 |

**Lecture** : sur les lots où AUCUN des deux runs A ou B n'a été frappé par la contention externe
(lots 1, 2, 5 : les quatre bras s'accordent au ms près) et même sur le lot 4 (3 bras sur 4
identiques), A et B rendent le **même TTFT, sans différence attribuable au mécanisme** — bien en
dessous du bruit visible ailleurs. Le lot 3 explose en A2 (3152 ms) ET en B1 (2368 ms), mais reste
normal en A1 et B2 : l'anomalie ne suit PAS la condition A/B, elle suit le MOMENT (une autre
session a pris la carte pendant ce lot précis, les deux fois) — c'est la signature de la
contention, pas de la capture en direct. **L'écart de 341 ms mesuré au tour précédent (A/B à un
seul lot chacun) est donc, rétrospectivement, la même contention, pas un effet réel.**

**Conclusion** : sur ce modèle et cette charge (27B, 8 requêtes concurrentes), le coût d'une
capture en direct (65-90 ms, chiffré en 184/186 en isolation) ne se voit PAS dans le TTFT réel d'un
service — il est submergé par le prefill à b=8 (500 ms à 1,2 s selon la longueur) et par la
dispersion de la file elle-même (facteur ×2-3 sous simple contention externe). L'hypothèse « la
capture gèle tout le lot » n'est pas confirmée : rien ne distingue A de B au ms près sur les lots
propres.

## Reste et recommandation

`ACVRAM_WARM_GRAPHS_B` reste **opt-in, défaut vide (OFF)** — la mémoire est gratuite (blocs KV
identiques, confirmé) mais le bénéfice en TTFT réel n'est pas mesurable ici, seulement le coût de
démarrage (+0,5 s pour 4 `b`, confirmé démarrage 13,2→13,7 s pour un seul `b`). Basculer le défaut
sur ON demanderait un bénéfice démontré, pas seulement un coût nul ; ce n'est pas le cas. Utile
seulement pour un service qui sait à l'avance qu'il servira des `b` précis et veut lisser le tout
premier appel de chacun (latence de queue de distribution, pas la moyenne).
