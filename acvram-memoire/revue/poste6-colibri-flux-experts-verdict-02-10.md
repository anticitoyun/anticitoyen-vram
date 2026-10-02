# bd 076 (3) — flux d'experts de colibrì contre l'exil d'acvram, sur nos disques : un seul levier s'applique aujourd'hui, le chargement à froid (0,62 → 3,2 Go/s, ≈ 22 s par modèle de 17 Go) ; le niveau disque ne concerne aucun modèle du parc

instrument : `scratchpad/poste6-076/mesure-disques.py` (O_DIRECT par `os.preadv`, tampons alignés, fils ; diagnostic, pas un instrument de cellule), recoupé par `dd iflag=direct` (2,5 Go/s pour 2,45) et `dd` tamponné (0,86 pour 0,87)
commit : poste6-gemma-anneau, scellé f5586e389 (poussé avant la première lecture), script au commit de ce verdict ; colibrì fd93c41 lu, non exécuté
régime : lecture seule, fichiers du projet seulement ; carte tenue en `service` par poste2 (llama.cpp, Qwen3.6-12B), aucune prise `mesure` ; charge 1,5-2,8 pendant les lots ; modèle témoin `Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4` (4 fragments de 4,1 Go, non servi)
scellé : `poste6-colibri-flux-experts-scelle-02-10.md`
mesuré : 9 lots, ≈ 2 min de lectures, 0 octet écrit
verdict : **(B) chargement à froid en O_DIRECT parallèle RETENU** (5,2 × notre chemin, seuil 1,5 ×) ; **(A) niveau disque : 0 modèle du parc concerné** et sous 1 jeton/s pour la classe GLM sur un disque ; **(C) miroir : les débits s'additionnent**, utile seulement derrière (A) ou (B) ; **(E) O_DIRECT 2,85 × le tamponné** ; (D) prélecture par le routeur non mesurée. Sept de mes douze prédictions de débit étaient FAUSSES, six par optimisme sur le NVMe des modèles, une par un seuil mal posé (M5).
durée : ≈ 1 h 10 de lecture de code et de mesure, 0 min de carte

## Prédit / mesuré

| | prédit | mesuré | |
|---|---|---|---|
| M1 FN960 (modèles), O_DIRECT 1 fil | 3,5-6,5 Go/s | **2,45** (2,03-2,50) | FAUX |
| M2 FN960, O_DIRECT 2 / 4 / 8 fils | 5,5-7,2 à 8 fils | 2,90 / 3,21 / **3,20** | FAUX : le disque plafonne à 3,2 |
| M3 notre chemin à froid (`safe_open().get_tensor`, lecture forcée) | 1,2-2,5 Go/s | **0,62** (4 Go en 6,9 s) ; à chaud 13,2 | FAUX, plus lent |
| M3 `dd` tamponné à froid | 2,0-3,5 Go/s | **0,86** | FAUX |
| tamponné parallèle à froid 2 / 4 / 8 fils (non scellé) | — | 1,46 / 1,20 / 1,80 | — |
| `fadvise WILLNEED` (non scellé) | — | sans effet (0 % en cache après 30 s) | — |
| M1 990 PRO (`/`) | 3-7 Go/s | **6,29** (0,5 Gio, indicatif) | tenu |
| M1 NM790 (`/home`) | 3-7 Go/s | 4,39 (0,34 Gio, indicatif) | tenu |
| M1 980 PRO en USB | 0,35-1,0 | 1,01 | tenu de justesse |
| M1 disque dur | 0,10-0,25 | 0,07 | FAUX, plus lent |
| M4 expert 2,53 Mio, 1 à la fois | 0,8-1,6 ms | **1,62 ms** (p95 2,46), 1,47 Go/s | FAUX d'un cheveu |
| M4 expert 2,53 Mio, 8 en parallèle | 3,5-6,5 Go/s | **2,71 Go/s** (6,8 ms par lecture) | tenu au seuil (≥ 2,5), sous la prédiction |
| M4 expert 19 Mio (colibrì), 1 / 8 | — | 7,75 ms, 2,33 Go/s / 3,16 Go/s | — |
| M5 FN960 + NM790 ensemble | ≥ 1,6 × le meilleur seul | 3,12 + 3,54 = 6,66 Go/s, **1,93 ×** | tenu |
| M5 FN960 + 990 PRO ensemble | ≥ 1,6 × | 2,49 + 5,67 = 8,17 Go/s, 1,51 × | FAUX au seuil écrit — mais chaque disque garde son débit seul : la somme est exacte, c'est mon seuil qui était mal posé pour deux disques inégaux |

Erreur d'instrument, dite : la première version de M3 lisait `get_tensor` sans copie — safetensors rend une VUE du mmap, rien
n'était lu (« 1,74 Go/s à froid », « 45 Go/s à chaud », 23,9 % du fichier en cache après une « lecture complète »). Corrigée
(`clone`), rejouée sur un autre fragment à 0 % de cache. FN970 : non mesuré (aucun fichier du projet, lecture seule).

## Ce qui s'applique à acvram

* **(B) chargement à froid — RETENU, à trancher par chef.** Notre chargeur lit par défauts de page du mmap, un fil,
  `read_ahead_kb = 128` : 0,62 Go/s. Le même disque rend 3,20 Go/s en O_DIRECT à 8 fils. Modèle de 16,8 Go (Coder-30B) :
  27 s → 5,3 s, **≈ 22 s gagnées** ; gemma-4-31B (19,5 Go) : 31 s → 6,1 s. **0 s à chaud** (cache de pages : 13,2 Go/s).
  Le parc compte 99 convertis, 14,8 Go en moyenne, pour 81 Gio de RAM disponible : deux à quatre modèles restent en cache,
  tout autre changement d'alias paie le froid. Trois façons, du moins au plus de code (aucune écrite) :
  (B3) `read_ahead_kb` du disque des modèles (sudo, utilisateur) — non mesurable sans sudo ;
  (B2) prélecture tamponnée des fragments par 8 fils avant `safe_open` : 1,80 Go/s mesurés (2,9 ×, ≈ 18 s gagnées sur
  27), dix lignes, le cache de pages reste chaud pour le rechargement ;
  (B1) lecture O_DIRECT parallèle des plages de tenseurs dans des tampons épinglés (la façon de colibrì) : 3,20 Go/s
  (5,2 ×), mais contourne le cache de pages — un rechargement relit le disque.
* **Hors code, le plus gros levier mesuré** : les modèles sont sur le NVMe le plus LENT de la machine (FN960 : 2,45 Go/s
  en un flux, 3,2 au plafond) ; le 990 PRO (`/`, 267 Go libres) rend 6,29 Go/s en un seul flux. Les alias les plus servis
  y liraient 2 × plus vite, O_DIRECT ou non. Décision de l'utilisatrice (disques).
* **(A) niveau disque pour les experts — ne s'applique à aucun modèle du parc** : le plus gros converti fait 59 Go, sous
  RAM + VRAM. Pour un MoE de la classe Coder-30B tout froid (384 experts de 2,53 Mio = 0,95 Gio par jeton) : 1,6 j/s à une
  lecture à la fois, **2,7 j/s** à 8 parallèles (prédit 3,5-6 : FAUX). Classe GLM (11 Go par jeton, chiffre de colibrì) :
  **0,29 j/s** sur le FN960, sous le seuil de 1 j/s ; ≈ 1,3 j/s en additionnant FN960 + NM790 + 990 PRO, à condition de
  trois copies. Ces modèles restent à colibrì (bd 076, parties 1-2).
* **(C) miroir multi-SSD** : les débits s'additionnent sans perte (6,66 et 8,17 Go/s mesurés à deux) — le hachage pondéré de
  colibrì (`expert_route`) serait la bonne façon ; sans objet tant que (A) n'existe pas ; pour (B), déplacer les modèles
  sur le disque rapide donne plus pour moins.
* **(E) O_DIRECT** : 2,45 contre 0,86 Go/s tamponné sur notre NVMe sans DRAM — le cas « neutre à négatif » que colibrì
  redoute ne se produit pas ici.
* **(D) prélecture par le routeur pour notre exil RAM → VRAM** : la seule idée de colibrì qui toucherait le DÉBIT d'acvram
  (experts exilés copiés pendant que la couche précédente calcule). Non mesurée : il faut la prédictibilité L → L+1 de nos
  modèles (trace de routage, sur carte) — pièce à part si chef la veut.

## Reste

Trancher (B) : B2 (prélecture tamponnée, petit) ou B1 (O_DIRECT, plus rapide, plus de code). Porter à l'utilisatrice le
placement des modèles sur le 990 PRO et `read_ahead_kb` (sudo). Rien d'autre de colibrì ne s'applique au parc actuel.
