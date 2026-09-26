# Pièce 87 — service à lots successifs vs serveur neuf, b=12 (poste2, 23/09)

instrument : `scratchpad/banc-llamacpp-16-09.py` decode (HTTP/SSE, BANC_URL/BANC_MOTEUR=acvram sur serveur
  déjà lancé) + `/metrics` toutes les 2 s, `scratchpad/poste2-p87-23-09/prise.sh` (adapté de `poste1-p85-23-09`)
commit : 08ae6ac2 (bras L et N, fusion pièce 85 incluse)
régime : -lgc 2700 (horloge moyenne 2 671-2 669), plafond 400 W, max-batch 12, max-model-len 2 304,
  défaut MAX_GRAPHS=64 (a383f677), alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c
scellé : dernière cellule b=12 (bras L, fin de séquence 1→2→4→8→12) à ±3 % du b=12 du serveur neuf
  (bras N), même séance ; repli_eager=0 au palier 12 (scratchpad/poste2-p87-23-09/scelle.md, avant la prise)
mesuré : L (lots successifs 1,2,4,8,12) b=12 : 1 756,4 t/s, 0,1406 J/jeton net, graphes_nombre=22,
  repli_eager=0, bridage puissance actif (normal au plafond, REGLES §1). N (serveur neuf, b=12 seul) :
  1 883,0 t/s, 0,1288 J/jeton net, graphes_nombre=12, repli_eager=0, bridage puissance actif.
  Paliers L complets : b=1 385,4 · b=2 405,2 · b=4 711,4 · b=8 1 334,0 · b=12 1 756,4 t/s (aucune
  anomalie franche à b=2 cette fois — pas de conclusion, poste1 instruit ce palier séparément).
verdict : critère NON TENU — L est 6,72 % SOUS N en débit (1 756,4 vs 1 883,0) et 9,16 % plus cher en
  J/jeton net (0,1406 vs 0,1288), hors bande ±3 %. repli_eager=0 aux deux bras (partie du critère tenue).
  Le reste (a) de la pièce 85 est donc confirmé, chiffré : même sans plafond de graphes atteint et sans
  repli eager compté, un serveur qui a déjà servi des paliers plus petits reste 6-9 % sous un serveur neuf
  à b=12, même séance, même commit. Cause non instruite ici (hors scellé) : candidats nommés par la 85 —
  429 pas hors graphe et 23 captures en cours de palier ne l'expliquaient qu'en partie ; graphes_nombre
  22 (L) contre 12 (N) montre que L a capturé 10 formes de plus (paliers 1/2/4/8) sans que cela coûte du
  temps de capture pendant le palier 12 mesuré (aucune capture vue dans les 6 dernières lignes /metrics de
  L) — donc la différence n'est pas une capture en direct au palier 12 lui-même, plutôt un effet cumulatif
  (fragmentation allocateur, cache, ou contention d'un processus plus longtemps vivant) à instruire.
durée : prévu ≤ 12 min / tenu 373 s (carte obtenue après 142 s d'attente, poste1 p86-b2 devant)
