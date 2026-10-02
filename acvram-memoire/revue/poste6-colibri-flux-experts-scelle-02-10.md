# bd 076 (3) — flux d'experts disque → RAM → VRAM de colibrì : ce qui s'appliquerait à acvram, sur NOS disques — scellé AVANT la mesure (poste6, 02/10)

Ordre : chef 02/10 (pièce sans carte ; lecture de `/mnt/AI_GENERATOR/colibri` fd93c41, débits réels de nos disques en
lecture seule, prédiction et seuils avant, verdict + INDEX, AUCUN code moteur). Écrit après la lecture du code, avant tout
octet lu sur un disque.

## Ce que colibrì fait (fichier:ligne, fd93c41)

1. **Trois niveaux pour les mêmes poids** (README:214, `docs/tuning.md` § Resource policy) : VRAM (chaud, fixé au démarrage),
   RAM (LRU par couche + experts épinglés), disque (« immutable recovery source »). Le placement ne change que la vitesse.
2. **Lecture d'expert** : les trois matrices d'un expert sont adjacentes, lues en UN `pread` (`c/colibri.c:2986-3021`) ;
   `DIRECT=1` = O_DIRECT aligné 4 K, **défaut OFF** (`:1260` : sur leur hôte le lecteur tamponné gagnait ; README:276 : +34 %
   à +65 % sur de vrais NVMe, neutre à négatif sur QLC / sans DRAM).
3. **Miroir multi-SSD** (`:2568-2700`) : copies en lecture seule, chaque expert routé vers un disque par un hachage
   DÉTERMINISTE pondéré par la bande mesurée (`expert_route`, `:2591`) ; sous O_DIRECT, un même expert est lu par bandes
   proportionnelles à la bande de chaque disque (`mir_stripe_plan`, `:2643` — 3 bandes égales avec un SATA : 12,6 ms au
   lieu de 2,0 ; pondérées : 1,9 ms) ; une erreur du miroir retombe sur le primaire.
4. **« Never wait for the disk twice »** (README:265) : pool de lectures asynchrone borné (`PIPE=1`, 8 fils, `:3549`) qui
   charge les experts manquants pendant que les résidents calculent ; union des experts d'un lot lue une fois ; prélecture
   par le routeur de la couche suivante (`PILOT=1`, `:1361`, 71,6 % du top-8 prédit une couche à l'avance sur GLM-5.2,
   « can be net negative on disk-saturated hosts »).
5. **Cache appris** : `.coli_usage` (chaleur de routage par usage) épingle les experts chauds au démarrage, réépinglage
   entre tours (`--repin`, hystérésis 25 %, 4 échanges).

## Ce qu'acvram fait aujourd'hui

* Exil = RAM hôte ÉPINGLÉE seulement (`loader.py:380-391`, arène) : tout poids exilé doit loger en RAM ; MLP dense exilé =
  un transfert PCIe par jeton (8,8 ms par MLP de gemma, `tiering.py:1131`), graphes coupés.
* Experts exilés : `ExpertPool` (`layers.py:335`), `n_slots` tampons par couche ; « rien n'est préchargé d'avance : on ne
  connaît les experts qu'après le routage » (`:348`, `moe.py:1809`). Profil de routage persistant déjà là
  (`expert_usage`, `loader.py:321`) : l'équivalent de `.coli_usage` existe, pour RAM ↔ VRAM.
* Chargement : `safetensors.safe_open(...).get_tensor` tenseur par tenseur (`loader.py`, `_ShardReader`) — mmap tamponné,
  un fil, puis copie vers la carte. Aucun O_DIRECT, aucune lecture parallèle.
* Aucun niveau disque : un modèle plus gros que RAM + VRAM ne se charge pas.

## Machine (relevé sans mesure)

RAM 93 Gio (81 disponibles), 8 cœurs logiques. NVMe PCIe 4.0 ×4 : FIKWOT FN960 4 To (`/mnt/AI_GENERATOR`, les modèles,
ext4), Lexar NM790 4 To (`/home`, xfs), Samsung 990 PRO 2 To (`/`), FIKWOT FN970 4 To (aucun fichier du projet : NON
mesurable en lecture seule, dit). Samsung 980 PRO 2 To en USB, SATA SSD, deux disques durs. Pas de `fio` : `dd` et un
script Python (`os.preadv` O_DIRECT, tampons alignés, fils). Expert de Qwen3-Coder-30B-A3B : 2,53 Mio (128 par couche,
top-8, 48 couches) → 384 experts = 0,95 Gio par jeton si tout est froid.

## Mesures prévues (lecture seule, fichiers du projet seulement, hors fenêtre `mesure` du verrou)

M1 séquentiel O_DIRECT 1 fil, blocs de 16 Mio, 4 Gio ; M2 O_DIRECT 4 et 8 fils sur régions disjointes ; M3 notre chemin :
cache de pages rendu (`posix_fadvise DONTNEED`), puis `safe_open().get_tensor` de tous les tenseurs d'un fragment de 4,1 Go
(et `dd` tamponné) ; M4 lectures aléatoires de la taille d'un expert (2,53 Mio et 19 Mio, celui de colibrì), 1 puis 8 en
parallèle ; M5 deux NVMe lus en même temps (FN960 + NM790).

## Prédictions et seuils

| | prédit | seuil / faux si |
|---|---|---|
| M1 FN960, O_DIRECT 1 fil | 3,5-6,5 Go/s | hors intervalle |
| M2 FN960, O_DIRECT 8 fils | 5,5-7,2 Go/s (plafond constructeur ~7) | < 5 |
| M3 notre chemin à froid (safe_open) | 1,2-2,5 Go/s | > 3,5 : notre chargeur est déjà proche du disque |
| M3 dd tamponné à froid | 2,0-3,5 Go/s | — |
| M1 990 PRO / NM790 (fichiers de 0,5 / 0,37 Go : indicatif) | 3-7 Go/s | — |
| M1 980 PRO en USB | 0,35-1,0 Go/s | > 1,2 |
| M1 disque dur | 0,10-0,25 Go/s | — |
| M4 expert 2,53 Mio, 1 à la fois, FN960 | 0,8-1,6 ms par expert (1,6-3,2 Go/s) | > 3 ms |
| M4 expert 2,53 Mio, 8 en parallèle | 3,5-6,5 Go/s agrégés | < 2,5 |
| M5 deux NVMe ensemble | ≥ 1,6 × le meilleur seul | < 1,3 × : borné par le processeur ou la mémoire, le miroir ne rend rien ici |

Décisions qui en découlent, seuils fixés maintenant :

* **(A) niveau disque pour les experts** — s'applique seulement à un modèle plus gros que RAM + VRAM (≈ 110 Gio utiles).
  Prédit : **0 modèle du parc acvram concerné** (le plus gros converti tient en RAM) ; n'ouvre que les MoE que bd 076 destine
  à colibrì lui-même. Utile si le pire cas tout-froid dépasse **1 jeton/s** sur un NVMe : pour un MoE de la classe
  Coder-30B (0,95 Gio/jeton) prédit 3,5-6 j/s tout froid à 8 lectures parallèles ; pour la classe GLM (11 Go/jeton, chiffre
  de colibrì) 0,5-0,65 j/s sur un NVMe, 1,0-1,3 sur deux — sous le seuil sur un seul disque.
* **(B) chargement à froid en O_DIRECT parallèle** — s'applique à TOUT modèle. Retenu si M2 ≥ 1,5 × M3 ET gain ≥ 5 s sur un
  modèle de 20 Go. Prédit : M2/M3 ≈ 2,5-4 ×, soit 5-11 s gagnées sur gemma-4-31B (19,5 Go) à froid ; **0 s à chaud**
  (cache de pages, 28 Gio aujourd'hui). Issue qui me gênerait : la part disque est petite devant la chauffe (220 s à
  65 536) — le gain existerait mais ne se verrait pas sur un démarrage de service.
* **(C) miroir / bandes multi-NVMe** — ne vaut que derrière (A) ou (B) ; retenu si M5 ≥ 1,6 ×.
* **(D) prélecture par le routeur (PILOT) pour notre exil RAM → VRAM** — non mesurable sans carte ; nommée, non chiffrée
  ici : il faudrait la prédictibilité L → L+1 de NOS modèles (trace de routage) et le recouvrement PCIe / calcul.
* **(E) O_DIRECT contre tamponné** : colibrì le laisse OFF par défaut ; prédit sur nos NVMe sans DRAM (FN960, NM790) :
  O_DIRECT 1 fil ≥ `dd` tamponné à froid de 1,2 × ; faux si < 1,0 × (alors (B) passe par des lectures tamponnées parallèles).

Contamination : le verrou `/tmp/acvram-carte-0.lock.qui` est relu avant chaque lot ; rien n'est lancé pendant une prise
`mesure` ; chaque lot ≤ 20 s, relevé de charge (`uptime`) avant et après. Un téléchargement vers `models_colibri` (même
NVMe) était en cours ce matin : vérifié arrêté ou dit.
