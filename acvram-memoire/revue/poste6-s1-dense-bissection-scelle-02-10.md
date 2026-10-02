# Pourquoi Devstral s'écarte de 10 × son témoin reprise sous les morceaux (0,0428 contre 0,00403) : scellé AVANT la bissection à sec (poste6, 02/10 15 h 3x, ordre chef)

Constat (carte, `poste6-s1-morceaux-verdict-ter-02-10.md`) : Devstral-24B, invite de 7 865 jetons, cache KV int8 dans tous
les bras ; morceaux de 4 096 (K/V bf16 transitoires) contre seul tenant : Δ 0,0428 à la position 0 ; même requête rejouée
(7 856 jetons relus du cache int8, une dizaine recalculés) : Δ 0,00403. Les bras A1/A2 sont au bit.

## Ce qui est déjà su avant de mesurer

* La reprise ne recalcule que la queue de l'invite : son erreur est celle de la relecture int8 des K/V pour une dizaine de
  lignes. Les morceaux, eux, changent la FORME de tous les appels (lignes M de chaque produit, longueur des clés de
  chaque attention) pour TOUTES les lignes de TOUTES les couches. Dix fois plus n'est pas absurde a priori.
* Sur processeur, le produit bf16 de torch est indépendant des lignes ; sur carte, la GEMM cuBLAS bf16 dépend de M
  (276i, 27/09 : `k_proj` couche 0, 1 ulp sur 38 % des éléments entre deux M). La bissection à sec ne peut donc PAS voir
  cette cause-là : elle peut seulement l'isoler par élimination.

## Instrument

Jouet de la CI (`converted`, cache int8, bf16), sur processeur : `forward` (seul tenant) contre `forward_tranches`
(morceaux, K/V transitoires), 20 séquences tirées au hasard de 400 jetons, morceaux [128, 256, 400] et [200, 400] ;
crochets sur chaque couche (q, k, v après projection et RoPE ; sortie du SDPA ; o_proj ; MLP ; résidu) ; et la grandeur du
comparateur S1 (écart des logits de la dernière position) contre le témoin reprise du jouet (deux passes, la seconde
relisant le cache int8). Script : `scratchpad/poste6-s1-dense/bissection.py`. Diagnostic, pas un instrument de cellule.

## Hypothèses et prédictions

| | hypothèse | prédit à sec | faux si |
|---|---|---|---|
| H1 | le SDPA rend un résultat qui dépend de la longueur des clés de l'appel (`layers.py:1063`, REGLES § 4) | premier écart = sortie du SDPA de la couche 0, avec q, k, v d'entrée ÉGAUX au bit ; reproduit hors modèle en appelant `attention()` sur les mêmes q, k, v à deux longueurs | premier écart ailleurs, ou q/k/v déjà différents |
| H2 | les K/V transitoires ne sont pas ceux du seul tenant | ÉCARTÉE : k, v égaux au bit couche 0 | k ou v différents à la couche 0 |
| H3 | le masque « bas-droite » des morceaux prend un autre chemin que le causal d'un seul tenant | ÉCARTÉE sur processeur : un morceau unique [400] rend 0 écart (déjà vu au diagnostic j4, graine 0) — confirmée sur 20 graines | un écart avec un seul morceau |
| H4 | GEMM cuBLAS bf16 dépendante de M sur les projections et le MLP | INVISIBLE à sec : sur processeur, projections et MLP égaux au bit dès que leur entrée l'est | un écart de projection à entrée égale sur processeur |
| G | grandeur du comparateur sur le jouet | écart des logits morceaux ≤ celui du témoin reprise (rapport 0,3-1,5) — l'inverse de Devstral sur carte (10,6) | rapport ≥ 5 sur processeur : la cause serait visible à sec |
| P | propagation | l'écart maximal des états cachés croît de la couche 0 à la dernière (≥ 1,5 ×) | décroît |

Conclusion attendue, écrite avant : à sec on trouve H1 seule (fichier:ligne `layers.py:1063`), d'une ampleur comparable
au témoin ; le facteur 10 de Devstral ne s'y reproduit pas → il vient de ce que seule la carte ajoute, H4 (cuBLAS selon M
sur ~7 projections × 40 couches × toutes les lignes), à confirmer par une micro-prise de 30 s (un `F.linear` bf16 de la
couche 0 de Devstral à M = 7 865 contre 4 096 + 3 769). Si G est faux (rapport ≥ 5 à sec), H4 n'est pas nécessaire et la
cause est lisible sur processeur — je la nommerai.
