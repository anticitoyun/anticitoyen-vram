# (b) soumis : ne plus épingler la source — les ~12 Gio non attribués

Écrit par poste2 le 8 septembre 2026 dans `/tmp/poste2-acvram`, **non poussé**,
en attente de relecture. Le témoin tourne : aucun test n'a été lancé.

## Ce que (b) change

`layers.py`, `StreamedWeight.__init__` : les tenseurs source sont passés à
`_emballer` **tels quels, paginables**, au lieu d'être épinglés un à un avant
l'emballage.

    - self.host = {k: (v.pin_memory() if not v.is_pinned() else v) ...}
    - self.plat, self.decoupe = _emballer(self.host)
    + self.plat, self.decoupe = _emballer(host_tensors)

`self.host = _decouper(self.plat, self.decoupe)` — le correctif (a) — est
conservé tel quel. Le seul `pin_memory()` restant est celui de `plat`, ligne
44 : c'est lui qui fait le DMA, il doit rester.

## Pourquoi ça vise les 12 Gio, et pas un candidat plausible

L'épinglage de la source était inutile deux fois : la copie vers `plat`
(`_emballer`, ligne 46) est **CPU→CPU** et n'exige rien de sa source ; et seul
`plat` sert au transfert.

Inutile, mais pas gratuit. **L'allocateur hôte épinglé de PyTorch ne rend
jamais au système** ce qu'il a pris — il le garde en cache pour réemploi. Les
tenseurs épinglés ligne 66, aussitôt déréférencés par le `_decouper` de (a),
restaient donc verrouillés pour la vie du processus. (a) avait supprimé la
double **référence** ; il laissait la double **épinglure transitoire**.

Chaîne d'élimination qui exclut les autres candidats : le témoin est une
perplexité, donc `evaluate.py:148` appelle `load_model` **seul**, sans Engine
ni Runner. Le cache KV hôte (`--host-kv-gib`, défaut **8 Gio**, épinglé), le
`_vers_hote` du runner et `bench.py` sont **hors chemin** — ce sont des objets
de `serve`. Les tampons de slots sont en VRAM, pas en `foll_pin`.

## Réserve 1 — la mesure avant, dans le même lancement

**Exigée par chef, et c'est ma propre règle : le calcul doit devenir un
relevé.** Avant d'accepter (b), relever dans le lancement du témoin :

    s = torch.cuda.host_memory_stats()
    s["reserved_bytes.all.current"] − s["allocated_bytes.all.current"]

**Prédiction écrite d'avance : 11-12 Gio.** `allocated` ≈ 34,75 Gio (les
`plat` des 30 couches + l'embed épinglé de `loader.py:273`), `reserved` ≈ 46
Gio (= le `foll_pin` mesuré).

**Issue nommée d'avance** : si l'écart est ailleurs — `allocated` > 35 Gio —
alors la source n'est pas celle-là, **(b) viserait faux**, et il faut chercher
quelle allocation détient les 12 Gio avant de l'appliquer.

## Réserve 2 — `_host_emptyCache` est un contournement, pas la correction

`torch._C._host_emptyCache()` existe (vérifié, torch 2.13.0+cu130) et rendrait
les 12 Gio **après** les avoir alloués. Il a sa place dans le protocole du
témoin pour gagner une soirée ; **jamais dans le moteur à la place de (b)**,
qui évite de les allouer.

## Ce que (b) ouvre au-delà du témoin

Si le cache hôte épinglé ne rend jamais rien, **tout modèle chargé puis
déchargé laisse un résidu épinglé pour la vie du processus**. Sur un serveur
qui change de modèle — ce que fait `serve` — le résidu s'accumule à chaque
chargement, invisible à tous les compteurs système. Ce n'est plus un problème
de mesure, c'est un problème de service.

`tests/test_epinglage_hote.py` (écrit, **non lancé**, exige la carte) garde les
deux propriétés :
* `test_la_source_n_est_pas_epinglee` — un poids exilé épingle son tampon plat
  et rien d'autre, à l'alignement près ;
* `test_l_epinglage_ne_s_accumule_pas_d_un_chargement_a_l_autre` — deux poids
  successifs ne doublent pas le verrouillage.

Le second est le test du problème de service, et il n'existait pas.

---

## RELEVÉ FAIT — (b) prouvé, et l'instrument prévu ne marchait pas

**L'instrument que nous avions tous deux validé était inexécutable.**
`torch.cuda.host_memory_stats()` :
* reste **vide** tant qu'aucun DMA n'a eu lieu — 200 Mio épinglés, dictionnaire
  vide ; elle ne se peuple qu'après un transfert réel ;
* **n'expose aucune clé `reserved_bytes`** ni `segment` — le
  `reserved − allocated` du protocole ne pouvait pas s'écrire ;
* ses `allocated_bytes.current` et `active_bytes.current` restent **égaux** et
  ne redescendent jamais : ils ne séparent pas ce qui est référencé de ce qui
  dort en cache.

Nous avions l'un et l'autre vérifié que la fonction **existe**, jamais qu'elle
**produise** ce qu'on lui demandait. Dix-septième cas, sur notre propre
protocole de validation.

**L'instrument juste est `foll_pin`**, et il est vérifié dans les trois sens :

    base                          0 Mio
    500 Mio épinglés            520 Mio   (+520 — vu SANS aucun DMA)
    après `del`                 520 Mio   (le cache ne rend rien : mesuré)
    après `_host_emptyCache()`    8 Mio   (le contournement marche : mesuré)

La ligne du milieu **prouve directement** le mécanisme avancé pour (b) : le
CachingHostAllocator ne rend jamais au système. Ce n'est plus une déduction.

**Résultat avant/après (b)**, sur un `StreamedWeight` de 256 Mio de poids,
cache vidé au départ :

| | épinglé | facteur | tests |
|---|---|---|---|
| avant (b) | 545 026 048 o | **2,03×** | 1 échec, 1 passé |
| après (b) | ≤ poids + alignement | **1×** | **2 passés** |

**Ce qui est prouvé** : le mécanisme, et que (b) le supprime.

**Ce qui ne l'est PAS, et je ne le prétends pas** : ma prédiction de 11-12 Gio
sur le témoin. Le facteur unitaire est 2,03 ; sur le témoin, 46 Gio pour 33,75
attendus donne 1,36. L'écart s'explique — 30 `StreamedWeight` successifs
laissent l'allocateur **réutiliser** les blocs libérés entre les couches, donc
le résidu ne double pas, il se stabilise plus bas — mais cette explication
reste une hypothèse. Le chiffre exact sur le témoin demanderait un chargement
complet avant/après ; il n'est pas nécessaire pour valider (b).
