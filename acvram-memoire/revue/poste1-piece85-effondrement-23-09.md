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
