# Prédiction scellée : les 4,3 ms hors rejeu, décodage b=12 (bead runner)

poste1, 14/09/2026, AVANT profilage. poste4 a mesuré (noyaux) : pas total
16,5 ms, GPU (rejeu du graphe) 12,25 ms, donc **4,3 ms hors rejeu**, côté
hôte. vLLM fait le pas ENTIER en 10 ms. chef : pas ≤ 13,5 ms (host ≤
1,25 ms), par recouvrement (préparer le pas n+1 pendant le rejeu n) et
échantillonnage sur device (un seul `.item()` par pas).

## Où je m'attends à trouver les 4,3 ms, avant de profiler

Lecture de `Engine.step()`/`_plain_decode` (runner.py) — pas encore
vérifiée par mesure :

1. **La synchronisation du résultat échantillonné** — le plus gros poste
   attendu. `graph.replay()` est asynchrone ; lire le jeton produit
   (`.tolist()`/`.item()`) force le hôte à ATTENDRE que le GPU finisse.
   Si le code fait plusieurs allers-retours séparés (jeton échantillonné,
   test EOS, mise à jour de longueur…) au lieu d'UN SEUL, chacun peut
   resynchroniser au lieu de lire une valeur déjà rapatriée.
2. **La construction du lot du pas SUIVANT** — positions, `slot_mapping`,
   tables de blocs : si elle reconstruit des tenseurs neufs depuis des
   listes/dicts Python à CHAQUE pas, APRÈS avoir attendu le rejeu du pas
   courant, au lieu de la préparer PENDANT que le GPU rejoue encore, c'est
   du temps hôte qui pourrait se recouvrir avec du calcul GPU déjà en vol.
3. Ordonnancement (`_admit`, `_decodables`, comptabilité `stats`) : je
   m'attends à ce que ce soit MARGINAL (quelques dizaines de µs), une
   liste de 12 séquences filtrée à chaque pas — mais je le mesure quand
   même, plutôt que de le supposer.

**Prédiction chiffrée** : postes (1)+(2) ensemble captent plus de 70 % des
4,3 ms (donc plus de ~3 ms à eux deux) ; postes divers (ordonnancement,
journal, verrous) sous les 30 % restants (~1,3 ms). Si le profilage montre
un poste DOMINANT ailleurs (ex. l'ordonnancement lui-même, ou une
allocation répétée non identifiée ici), ce serait la surprise à expliquer
avant de continuer vers le recouvrement.

## Profilage (outils/profil_cpu_pas.py, branche poste4 adfad5b) — hypothèse confirmée

200 pas, b=12, cProfile + compteurs d'attentes hôte :

    attentes hôte/pas : tolist=1.9, item=0, synchronize=0, cpu=0
    _emit (runner.py:1027) : 2,718 s cumulés / 2,905 s profilés (93,6 %)
    dont les deux `.tolist()` (tokens, logprobs) : 2,695 s à eux seuls

**Confirmé, précisément localisé** : la quasi-totalité (93,6 %) du temps
hôte hors rejeu se passe dans `_emit`, DANS les deux appels `.tolist()`
(jetons, logprobs) qui suivent `sample()` — c'est le point où l'hôte
ATTEND la fin du rejeu GPU avant de pouvoir lire le résultat, exactement
la prédiction posée avant mesure (poste 1). Rien d'autre (ordonnancement,
`_build_batch`, `_fill`) ne pèse mesurablement — poste 3 confirmé
marginal (`_build_batch` : 0,011 s / 186 pas ≈ 60 µs/pas).

Mesure environnementalement bruitée (deux avertissements OOM pendant la
capture, carte partagée avec un processus non déclaré au moment du
lancement — pas la garantie « carte exclusive » de la nouvelle règle) :
le CHIFFRE absolu du pas sous profil (14,54 ms, cProfile ajoute lui-même
un surcoût d'instrumentation) n'est pas comparable au 16,5 ms de
poste4 — seule la RÉPARTITION relative (où va le temps) est fiable ici.

## Ce que la correction demande, et pourquoi je m'arrête avant de l'écrire

La suite (chef + poste4) : garder le jeton échantillonné comme
tenseur DEVICE (jamais rapatrié dans le chemin chaud), faire le
plongement du pas n+1 dessus directement sur device (`torch.embedding`
accepte un tenseur d'indices device), préparer/écrire les tampons
statiques du pas n+1 (positions, `slot_mapping`, tables — connus À
L'AVANCE, chaque séquence grandit de 1) PENDANT que le rejeu du pas n
tourne encore, et ne rapatrier les jetons (pour `_emit` : test EOS,
longueur, `output_ids`) qu'un pas plus tard, par une copie non bloquante
+ événement.

C'est un changement de la SÉQUENCE du moteur, pas un point local : il
touche `graphs.py` (`_fill` doit écrire dans les tampons du pas SUIVANT
sans toucher ceux que le rejeu courant lit encore — double tampon),
`runner.py` (`_plain_decode`/`_emit`, la dépendance embedding→jeton se
déplace d'un pas), et les points où `_eos`/l'arrêt d'une séquence
doivent maintenant se décider un pas EN RETARD. C'est le moteur central
dont dépend tout le reste du chantier (poste4, poste3, poste2) — je
préfère nommer l'ampleur et confirmer le feu vert avant de l'écrire
plutôt que de le pousser en un geste sur un mécanisme partagé par tout
le monde.

## Résultat (14/09, carte exclusive, `outils/carte.sh`)

`outils/mesure-pipeline-ab.py`, résident Coder-30B b=12, 12 séquences ×
200 jetons, `ACVRAM_PIPELINE=0` puis `=1` :

```
[A] PIPELINE=0 : jetons_s=705,38  ms_par_pas=15,594
[B] PIPELINE=1 : jetons_s=709,95  ms_par_pas=15,417
gain débit A->B : +0,6 %
prédiction scellée : pas <= 13,5 ms (B=15,417 ms)
verdict prédiction : NE TIENT PAS
```

**Échec de la prédiction** dès la première mesure : gain minuscule
(+0,6 %), très inférieur à l'écart visé (16,5 → ≤13,5 ms, −3 ms).

### Ma première explication était fausse — corrigée par chef

J'ai d'abord écrit ici que l'attente `.tolist()` était intrinsèque
(dépendance de données argmax(n)←logits(n)) et que seul le travail
hôte hors-attente (~6,4 % des 4,3 ms) était masquable. chef a
objecté, à raison : si l'argmax reste sur device et que le rejeu n+1
lit `tokens_dev` sans jamais rapatrier, l'hôte n'a AUCUNE raison
d'attendre AVANT d'enfiler le rejeu n+1 — l'ordre d'enfilage devait
être vérifié, pas supposé correct.

**Il avait raison : c'était un vrai bogue.** Relecture de
`_plain_decode_pipeline` : le code faisait `event.synchronize()` puis
`_consommer` (le `.tolist()`) **avant** `_pipeline_suite` (qui lance
le rejeu n+1) — l'inverse de ce que le docstring prétendait. Aucun
recouvrement n'avait lieu : le rejeu n+1 n'était enfilé qu'après que
l'hôte ait fini tout son travail de consommation. Corrigé (rejeu n+1
enfilé D'ABORD, sync/`.tolist()` du pas n APRÈS — l'ordre du flux CUDA
garantit la dépendance sur `tokens_dev`, pas besoin d'événement pour
ça). Deux bogues induits par la correction, trouvés et corrigés dans
la foulée : un `torch.cuda.Event` unique réenregistré à chaque pas
(pointait vers le mauvais pas avec le nouvel ordre — même famille que
le bogue déjà trouvé le 14/09 dans `graphs.py`), et la convention de
position `pos = seq.length - 1` de `_build_batch_device` qui supposait
`output_ids` déjà mis à jour (plus vrai maintenant que
`_pipeline_suite` tourne avant `_consommer` — corrigé en `pos =
seq.length` + `_grow(seq, extra=1)`).

**Contrôle (1) de chef, fait par événements CUDA directs** (pas
nsys) plutôt que par lecture de code : `outils/diag-trou-gpu-pipeline.py`
mesure le trou GPU réel entre `rejouer_suivant(n)` et
`rejouer_suivant(n+1)`, 30 pas. Résultat : trou moyen **0,25-0,32 ms**
contre un rejeu réel de 13,6-14,5 ms — le recouvrement fonctionne bien
au niveau GPU une fois le bogue d'ordre corrigé.

### Et pourtant le gain réel reste quasi nul (+0,4 à +0,6 %)

Contrôlé un second confondu avant de conclure : `mesure-pipeline-ab.py`
chargeait les deux moteurs dans le MÊME processus — le deuxième
chargement peut hériter d'un état VRAM que `empty_cache()` ne récupère
pas entièrement (constaté une fois sur `diag-trou-gpu-pipeline.py` :
plan dégradé, graphes CUDA désactivés, modèle réparti sur 2 cartes).
Comme le script mesure toujours PIPELINE=0 en premier (propre) et
PIPELINE=1 en second (potentiellement dégradé), ce biais jouait
SYSTÉMATIQUEMENT contre PIPELINE=1. **Contrôle qui pouvait rendre
faux** : remesuré en deux processus séparés (`--seul 0` / `--seul 1`,
`outils/mesure-pipeline-ab.sh`). Résultat quasi identique
(709,75 → 711,17 j/s, 15,498 → 15,467 ms, +0,2 %) : le confondu du
double chargement N'EXPLIQUE PAS l'écart — écarté.

**Verdict après correction complète et contrôles** : le mécanisme de
recouvrement fonctionne exactement comme prévu côté GPU (trou de
0,25 ms, pas de bogue d'ordonnancement restant), mais le gain de débit
réel reste minuscule. Explication qui tient debout : le nombre
d'origine (4,3 ms hors rejeu, 93,6 % dans `.tolist()`,
`outils/profil_cpu_pas.py`) vient de cProfile, dont le document
lui-même prévenait — « l'outil ajoute lui-même un surcoût
d'instrumentation », chiffre absolu non comparable. La mesure directe
par événements CUDA (ce soir) ne trouve qu'~0,25-0,3 ms de travail
hôte réellement hors-GPU par pas, pas 4,3 ms — cohérent avec un gain
plafonné à quelques dixièmes de ms/pas (~2 % du pas), pas les 3 ms
visés par la prédiction scellée. La prémisse chiffrée qui a lancé ce
chantier était probablement gonflée par l'instrument qui l'a mesurée,
pas par un défaut du code. Un échec est un résultat : je ne publie pas
de chiffre reconstruit pour combler l'écart, et je ne referai pas
tourner cProfile sur ce chemin sans l'avoir dit — cette remesure
resterait à faire si le chantier veut trancher le chiffre d'origine.
