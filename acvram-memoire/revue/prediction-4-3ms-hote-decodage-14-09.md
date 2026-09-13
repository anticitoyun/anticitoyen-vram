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
200 jetons, `ACVRAM_PIPELINE=0` puis `=1`, même processus :

```
[A] PIPELINE=0 : jetons_s=705,38  ms_par_pas=15,594
[B] PIPELINE=1 : jetons_s=709,95  ms_par_pas=15,417
gain débit A->B : +0,6 %
prédiction scellée : pas <= 13,5 ms (B=15,417 ms)
verdict prédiction : NE TIENT PAS
```

**Échec de la prédiction.** Le gain mesuré (+0,6 %, −0,18 ms/pas) est
très inférieur à l'écart visé (16,5 → ≤13,5 ms, soit −3 ms).

### Cause : la prédiction confondait l'ATTENTE et le TRAVAIL hors rejeu

chef l'avait déjà précisé avant la mesure : « `.tolist()` à 93,6 % =
l'hôte qui ATTEND le GPU ; le gain vient du recouvrement de ce qui suit
(emit/build/fill) avec le rejeu suivant, pas de la suppression de
l'attente. » Le profilage (`outils/profil_cpu_pas.py`) avait mesuré
_emit cumulant 93,6 % de son temps DANS `.tolist()` — c'est-à-dire que
sur les 4,3 ms hors rejeu, l'écrasante majorité n'est pas du calcul
hôte à recouvrir, c'est de l'attente d'un résultat GPU dont dépend la
suite (l'argmax du pas n dépend des logits du rejeu n : cette
dépendance ne disparaît pas, le recouvrement ne fait que déplacer QUI
attend, pas la durée de l'attente elle-même).

Seule la part hors-`.tolist()` de ces 4,3 ms — le reste, ~6,4 % —
est du vrai travail hôte (emit/build/fill/preparer) susceptible d'être
masqué derrière le rejeu du pas suivant. Borne théorique :
4,3 ms × 6,4 % ≈ 0,28 ms/pas. La mesure (−0,18 ms/pas) est du même
ordre de grandeur que cette borne, pas du même ordre que les 3 ms
visés — c'est cohérent avec le mécanisme réellement implémenté (un
seul `.tolist()` par pas, retardé, embedding sur device), pas avec un
défaut d'implémentation.

Confondu ayant été écarté : `charge_s` de la mesure [A] (119,0 s) est
anormal (file d'attente carte 368 s + verrou de compilation orphelin
977 s rapportés au lancement, cf. log brut) mais ce délai est dans le
CHARGEMENT du modèle, hors de la boucle chronométrée (`duree_s`,
`ms_par_pas` ne courent qu'après `warm_graphs()`) — les deux mesures
[A] et [B] ont des `duree_s` proches (3,10 vs 3,08 s) dans le même
processus, donc la contention affecte les deux de façon comparable et
n'explique pas l'écart au seuil.

**Verdict** : le recouvrement fonctionne (bit-identique validé,
`tests/test_pipeline_decodage.py`, 1 passed) mais son gain réel est
plafonné par la part non-attente du hors-rejeu, pas par les 4,3 ms
entiers. La prédiction de chef (≤13,5 ms) supposait implicitement
que le recouvrement supprimait l'attente elle-même ; ce n'est pas ce
que le mécanisme fait, et ce n'est pas ce qu'il peut faire tant que le
pas n+1 dépend du jeton du pas n. Un échec est un résultat : je ne
publie pas de chiffre reconstruit pour combler l'écart.
