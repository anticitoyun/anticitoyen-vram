# Verdict — 186 : combien de captures de graphe jusqu'à max_model_len, préchauffe actuelle, coût d'une préchauffe complète (poste2, 25/09, à sec)

* **instrument** : lecture de code seule (aucune carte, ordre chef « pas de code avant mon feu ») + réutilisation
  d'un chiffrage déjà fait (`acvram-memoire/revue/chiffrage-vram-max-graphs-64-13-09.md`, poste3 13/09, relu avant
  d'écrire — règle poste1 25/09) pour la formule mémoire par clé, adaptée au modèle réel de la 184
  (Qwen3.8-27B-unsloth-mixte-i8c, `hidden_size=5120`, `vocab_size=248320`, lu dans `config.json`).
* **commit** : worktree `poste2-p186` depuis `origin/main`. **régime** : aucun (à sec).

## (1) Combien de clés (b, ql, nblk, lb) jusqu'à `max_model_len`, mémoire, temps

**Les quatre dimensions de la clé** (`graphs.py:674`) :
* `b` = `godet_lot`/`godet_hybride` (`graphs.py:68-95`) : puissances de deux, plafonnées à `max_slots`
  (`plafond_hybride`, `graphs.py:259-272` : `ACVRAM_HYBRID_SLOTS` sinon `--max-batch`, défaut **16**,
  `cli.py:1278`). → **b ∈ {1, 2, 4, 8, 16}, 5 valeurs.**
* `ql` : **1 seule valeur** pour ce banc (`--speculative none`, pas de vérification spéculative — `ql>max_ql=1`
  refusé, `graphs.py:665-667`). Une spéculation active ajouterait `ql=spec_k+1` en plus (1 valeur de plus, pas
  un balayage).
* `nblk` : `bucket_blocks` (`kvcache.py:40-50`), doublement depuis 8, jusqu'à `ceil(max_model_len/16)`. Au
  contexte servi de la 181/184 (`kv_budget=32768`, régime lu dans les deux prises) : `ceil(32768/16)=2048` →
  buckets **8,16,32,64,128,256,512,1024,2048 = 9 valeurs.**
* `lb` (hybride seulement) : `godet_mla` (`mla.py:307-320`), doublement depuis `MLA_BUCKET=128`
  (`mla.py:30`, `ACVRAM_MLA_BUCKET`). **`nblk` et `lb` démarrent au même palier (128 jetons = 8 blocs × 16) et
  doublent à l'identique** : ils changent de valeur aux MÊMES longueurs de séquence, donc ne multiplient pas
  l'espace de clés entre eux — comptés comme **une seule dimension "palier de contexte", 9 valeurs.**

**Total théorique (cartésien b × palier)** : 5 × 9 = **45 clés distinctes**, sous le plafond `MAX_GRAPHS=64`
(`graphs.py:129`) — donc TOUTES capturables sans refus, si le trafic (ou une préchauffe) les présentait toutes.
En pratique un seul lot voit `nblk` et `lb` croître ENSEMBLE avec sa longueur (pas en croisé avec chaque `b`) :
le nombre de clés réellement rencontrées dépend de la diversité des tailles de lot servies, pas seulement de
la longueur.

**Mémoire, par la formule de la 13/09** (`graphs.py:498-506`, tampons `entry` PAR CLÉ, hors pool partagé) :
adaptée à ce modèle (`hidden=5120`, bf16), au pire palier (`b=16`, `nblk=2048`) :

    x          16 × 1 × 5120 × 2 octets   = 163 840 octets (160 KiB)
    tables     16 × 2048 × 8 octets       = 262 144 octets (256 KiB)
    positions/slots/seq_lens              ≈     400 octets (négligeable)
    ------------------------------------------------------------------
    pire clé (b=16, nblk=2048)            ≈ 426 KiB

Sommé sur les 45 clés (Σb=1+2+4+8+16=31, Σnblk=4088) :

    Σ tables = 8 × 31 × 4088                ≈  990 KiB
    Σ x      = 10 240 × 31 × 9              ≈ 2 856 KiB
    ------------------------------------------------------------------
    total tampons `entry`, 45 clés          ≈ 3,7 MiB

`entry["out"]` (sortie de `decode_fixed`) : sous le sampler-graphe **par défaut** (`graphs.py:294`,
`sampler_graphe_actif()`, verdict poste4 d145bf0d 22/09), la sortie capturée est `[2, b·ql]` **int64**
(ids + bits des logprobs empaquetés), PAS les logits pleins — quelques centaines d'octets par clé, négligeable
(contredit l'hypothèse haute de la 13/09 qui supposait des logits complets : elle datait d'avant ce défaut).
**Total mémoire des 45 clés ≈ quelques Mio — largement sous le seuil de 1 Gio de poste7**, cohérent avec la
conclusion du 13/09 (le pool partagé (b), pas les tampons (a), reste le vrai poste à surveiller, et il ne
scale pas avec le nombre de clés — `graphs.py:571-579`, `chiffrage-vram-max-graphs-64-13-09.md § 2-4`).

**Temps** : coût par capture observé 63-90 ms (184, `srv-T.log`), cohérent avec la fourchette documentée
« 40 à 130 ms » (`graphes.py:42`). **45 captures × ~75 ms (médiane) ≈ 3,4 s** au total si TOUTES étaient
capturées d'affilée (pas de recouvrement possible : capture = exclusive, `torch.cuda.graph(...)`).

## (2) Ce que fait le démarrage aujourd'hui — fichier:ligne

**Oui, il préchauffe — mais seulement `b=1`.** `cli.py:875-890` (commentaire « Ordre imposé (chef 21/09) :
plan → clamp → capture → chauffe ») appelle `engine.demarrer_service(...)` (`contexte.py:191-199`), qui appelle
`self.warm_graphs(warm_max_len)` (`contexte.py:199`) si `self.graphs is not None`.

`warm_graphs` (`acvram/engine/graphes.py:40-64`) : boucle `L = 128, 256, 512, 1024, 2048, …` tant que
`L ≤ min(max_len, max_model_len − 4)`, `max_len = ACVRAM_WARM_GRAPHS` (**défaut 2048**, `cli.py:882`) —
**PAS `max_model_len`** (32768 en production) : la préchauffe s'arrête à 2048 jetons par défaut, bien avant le
contexte réel servi. Pour chaque `L` : **un seul `self.generate([1]*(L-2), max_tokens=2)`** (`graphes.py:52-54`)
— **une séquence UNIQUE (`b_reel=1`)**, jamais `b>1`. Si hybride + spéculateur + `ACVRAM_WARM_SPEC≠0` (défaut
actif) : une seconde passe à `b=1` aussi, pour capturer la forme `ql=spec_k+1` (`graphes.py:55-62`).

**Conséquence directe, vérifiée sur la 184** : les captures `(1, 1, nblk, lb)` (6 clés, `srv-T.log:8-18`) sont
bien celles du démarrage (`L=128..4096`, avant toute requête). Les captures `(8, 1, nblk, lb)` (`srv-T.log:35,
86, 288`, celles qui coûtent la latence observée) arrivent **EN DIRECT, pendant le banc** — parce que
`warm_graphs` ne préchauffe jamais `b>1`. C'est la cause structurelle du problème que soulève chef : tout
`b≠1` rencontré pour la première fois à un palier de contexte donné paie sa capture pendant une vraie réponse.

## (3) Chiffrage d'une préchauffe complète (tous les paliers probables, tous les `b`)

**Temps de démarrage en plus** : aujourd'hui, 5-6 captures `b=1` (`L` jusqu'à 2048, `ACVRAM_WARM_GRAPHS`) ≈
5 × 75 ms ≈ **0,4 s** déjà payés. Étendre à `b ∈ {1,2,4,8,16}` × 9 paliers de contexte (jusqu'à `max_model_len`
réel, pas 2048) = 45 clés (§1) → **+40 captures ≈ +3,0 s au démarrage** (net, après retrait des 5 déjà
payées) — modeste dans l'absolu, mais chaque capture EXIGE une exécution réelle du modèle au palier visé
(`generate(...)`, pas une simple allocation), donc ce n'est pas 3 s de pur overhead fixe : c'est 3 s de calcul
carte en plus au démarrage, sur un modèle de 27B.

**Mémoire** : ≈ 3,7 MiB de tampons `entry` en plus (§1) — négligeable. Risque résiduel non chiffré ici : si un
des 40 paliers nouveaux capture une activation plus grande que jamais vue par le pool partagé
(`self._pool`, `graphs.py:576-578`), CE pool grandit — borné par la plus grande forme (`b=16, nblk=2048`),
pas mesuré ici (chiffrage 13/09 § 4 : « le risque existe mais n'est pas structurel »).

**Effet sur la capacité KV via le planificateur (pièce 146, `runner.py:1339-1361`) — LE VRAI COÛT, non
mémoire mais TRANSITOIRE** : préchauffer la clé `(b=16, nblk=2048)` exige de faire tourner **16 séquences
synthétiques simultanées, chacune à 32 768 jetons de contexte**, le temps de la capture — soit
`16 × 2048 = 32 768` blocs KV (524 288 jetons) retenus d'un coup par `_grow` (`runner.py:1374-1381`), rendus
juste après (`_finish`, `runner.py:1383-1401`, comme le fait déjà `warm_graphs` à `b=1` aujourd'hui — la même
mécanique, mais 16× plus large ET au palier le plus long). **Si le pool `BlockAllocator` (dimensionné par
`host_kv_gib`, `cli.py:874`) ne tient pas cette pointe transitoire**, `_grow` échoue et
`_finish_budget_epuise` (`runner.py:1339-1361`, pièce 146) **tronque silencieusement la séquence de
préchauffe** — la capture au palier maximal échouerait alors AVANT d'atteindre la forme visée, pas seulement
lentement : elle ne se ferait tout simplement pas, sans faire tomber le serveur (le code de la 146 est fait
pour ça), mais sans non plus livrer le graphe attendu. Aucun service réel ne sert normalement 16 requêtes
TOUTES à 32k jetons en même temps (la 181/177 en servent 8, contexte court) — la préchauffe au pire cas
demanderait donc une pointe de KV **plus grande que ce que le trafic réel utilise jamais**, rien qui garantit
qu'elle tienne dans le budget dimensionné pour ce trafic. **Ce chiffre-clé (le budget KV réel vs la pointe de
préchauffe) n'est pas mesuré ici** — dépend de `host_kv_gib` du déploiement, pas d'une constante du dépôt.

## Ce que je propose, sans coder (feu de chef requis)

* Préchauffer un sous-ensemble raisonnable plutôt que les 45 clés : les `b` réellement servis (pas toute
  puissance de deux jusqu'à 16 si le trafic ne dépasse jamais b=8) × les paliers courts (128-2048, déjà fait)
  serait quasi gratuit ; les paliers longs (4096-32768) à grand `b` sont ceux qui coûtent le plus en VRAM
  transitoire ET sont les moins probables en trafic réel (conversation longue ET très parallèle à la fois).
* Le point qui manque avant tout code : mesurer `host_kv_gib` effectif du déploiement contre la pointe
  transitoire du plus grand `b` visé — sans ce chiffre, impossible de dire si la préchauffe complète est même
  RÉALISABLE sans tronquer sa propre capture.
