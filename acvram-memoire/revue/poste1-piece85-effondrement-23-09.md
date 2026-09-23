# Pièce 85 — effondrement du service à lots successifs (poste1, 23/09)

Constat (prise invalide de la 82, `scratchpad/poste1-p82-23-09/banc-mint-T.log`) : un serveur unique, paliers
b = 2, 4, 8, 12 : b=12 516 t/s, 139 W — mais `meilleure_passe_jetons_s` 1 918 sur les passes courtes (128 jetons)
du MÊME palier. L'effondrement porte sur les lots longs (1 024 jetons, contexte 257 → 1 280), pas sur le serveur
entier.

## Lecture à sec, avant la prise

`acvram/engine/graphs.py:614-618` : quand `len(self.graphs) >= MAX_GRAPHS` (16), la clé nouvelle rend `False`
SANS `self._eager(...)` : repli eager non compté (`repli_eager` reste 0), non nommé (la ligne n'imprime que sous
`ACVRAM_TRACE_STEPS`), définitif (aucune éviction ; `regime.py:125` dit à tort « éviction au-delà »).
Clé = (b godet ∈ {1,2,4,8,16}, ql, nblk ∈ {8..256}, lb). `warm_graphs` (`engine/graphes.py:40`) prend déjà 5 places
à b=1 (L = 128..2 048) + la confirmation à 2 304 (nblk 256) ; la spéculation n-gramme (lot_max 2) ajoute des ql > 1
à b ≤ 2. Les paliers 2, 4, 8 remplissent le reste ; à b=12 (godet 16) les nblk 64 et 128 n'ont plus de place.

## Hypothèse H1 et ce qui la réfuterait (écrit avant la prise)

H1 : plafond MAX_GRAPHS atteint, repli eager muet sur les clés (16, 1, 64|128, 0).
Prédit, bras A (défaut 16) : au palier 12, `graphes_nombre` = 16, `graphes_captures` figé, Δreplays / Δsteps
≤ 0,5 pendant les lots longs, `repli_eager` inchangé ; b=12 ≤ 900 t/s. Bras B (`ACVRAM_MAX_GRAPHS=64`, même
séquence) : b=12 ≥ 1 800 t/s, `graphes_nombre` ≤ 40, Δreplays/Δsteps ≥ 0,95.
**H1 réfutée si** : (a) `graphes_nombre` < 16 au palier effondré ; ou (b) Δreplays/Δsteps ≥ 0,9 pendant
l'effondrement (graphe rejoué, lenteur ailleurs) ; ou (c) le bras B s'effondre aussi (< 1 500 t/s à b=12).
Issues nommées : H2 spéculation n-gramme restée active ou garde (b ≤ 2 seulement, n'explique pas b=12) ;
H3 préemption/KV (lirait `preemptions` ou lot < 12 dans /metrics) ; H4 allocateur (pool fragmenté après
captures : B aggraverait au lieu de guérir). Alarme : si B et A donnent le même b=12, je mesure autre chose que
le plafond, et je le dis.

Instrument : `scratchpad/banc-llamacpp-16-09.py` (celui de la prise invalide, même paramètres), /metrics relevé
chaque seconde par `scratchpad/poste1-p85-23-09/releve-metrics.sh`. -lgc 2700. Prise ≤ 12 min, deux bras.

## Verdict

instrument : `scratchpad/banc-llamacpp-16-09.py` (HTTP/SSE, un serveur par bras, paliers 2→4→8→12 enchaînés) + `/metrics` toutes les 2 s, `scratchpad/poste1-p85-23-09/prise.sh`
commit : 1f8df367 (bras A/B), a383f677 (preuve D, défaut) ; alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c
régime : -lgc 2700 (horloge moyenne 2 636-2 678), plafond 400 W, max-batch 12, max-model-len 2 304
scellé : H1 réfutée si graphes_nombre < 16 au palier effondré, ou replays/pas ≥ 0,9 pendant l'effondrement, ou B < 1 500 t/s à b=12 (f43ed4c8, avant la prise)
mesuré : b=12 — A (16) 536,3 t/s, 135 W ; B (64) 1 802,4 ; D (défaut 64, sans variable) 1 742,0, 0,145 J/jeton net. b=8 — A 291,9 ; B 1 262,5 ; D 1 325,0
verdict : H1 TENUE. A : 16 places prises au palier 8, puis replays figés à 3 186 pendant que repli_eager monte de 0 à 3 009 ; B : 21 graphes, D : 23, repli_eager 0
durée : prévu ≤ 12 min / tenu 141 s (prise 1, contaminée) + 351 s + 149 s (journal `tenue=`)

Cause 1 (le constat de chef) : `acvram/engine/graphs.py:123` MAX_GRAPHS=16 sans éviction, et `graphs.py:614-618` refusait
sans `_eager` (repli muet, `repli_eager` = 0). Correctif : refus compté et nommé (1f8df367), défaut 64 (a383f677).
Cause 2, trouvée par ma première prise : `/metrics` → `Engine.regime()` (appelé 4 fois) → `_etat_eco()` → relecture
SOUS CHARGE (`eco.py:244-269`, matmuls + `synchronize` dans un fil) dans le processus de service. Un synchronize d'un
autre fil pendant une capture l'invalide : « operation failed due to a previous error during capture », graphes coupés
à vie (godet 16). Toute supervision qui interroge `/metrics` (tableau de bord, GUI) pouvait tuer les graphes. Correctif
`runner.py:_etat_eco` : relecture sous charge au premier `regime()` seulement (1f8df367). La prise 1 est donc invalide
(elle mesurait mon instrument) et le dit.
Tests qui cassaient : `tests/test_piece85_service_lots_successifs.py` (3 tests ; rouges sur l'ancien code ou avec
ACVRAM_MAX_GRAPHS=16).

Restes, nommés : (a) D b=12 1 742 reste à 8-12 % sous le serveur neuf du jour (1 940-1 995, pièce 77) : 429 pas hors
graphe sur 10 942 et les captures en cours de palier (23 × 40-130 ms) n'expliquent pas tout — à mesurer par poste2, serveur
neuf contre lots successifs, même séance. (b) b=2 anormal dans les trois bras (99,8 à 247,5 t/s, sous b=1 = 311) :
spéculation n-gramme (lot_max 2) soupçonnée, non prouvée — pièce à ouvrir. (c) Cellule publiable = poste2 (REGLES § 3,
l'auteur de la méthode ne produit pas la mesure qui la couronne).
