# GEMV groupée des experts v2 — à sec (poste4, 18/09) ; porte = micro-banc de poste3

Commande : poste7-lecture-profils-coder-17-09 § 2 (priorité finale du circuit) :
décodage Coder b=12, GEMV d'experts 6,6 ms à 68 % de bande (8,1 Go/pas),
plancher 4,5 ms ; scellé **6,6 → ≤ 5,3 ms**, b=12 nu 1 114 → **≥ 1 300 t/s**
(faux < 1 200).

## Ce que fait v1 (`nvfp4_gemv_grouped_gateup_kernel`, `_warp_kernel`)

Un bloc par PAIRE (expert, jeton) et par tranche de 8 lignes : les poids de
l'expert sont relus une fois par paire — 96 paires pour ~69 experts
distincts à b=12 (× 1,39), les paires arrivant dans l'ordre des jetons (un
expert repris 3 jetons plus loin peut être déjà sorti du L2 : 183 Mo par
couche pour 96 Mo de L2) ; l'activation est étagée en mémoire partagée à
chaque bloc (4 Kio lus par bloc pour 16 Kio de poids).

## v2 (`acvram_kernels.cu` : `nvfp4_gemv_grouped_v2`, `_gateup_v2`)

- Les paires sont TRIÉES par expert côté hôte (`torch.argsort(eid,
  stable=True)`, déterministe, sous graphe) ; le noyau reçoit
  `eid_s`, `tok_s` et `ordre` (place d'origine de chaque paire) et ÉCRIT
  chaque sortie à sa place d'origine : act, down, `moe_reduce` inchangés.
- Bloc « meneur » : au début d'un segment d'expert, ou à un multiple de TPB
  depuis ce début (fil 0 remonte le segment, diffuse par la mémoire
  partagée) ; il sert jusqu'à TPB jetons (4 pour K ≤ 2 976, 2 jusqu'à
  5 952, 1 au-delà — l'étage tient en 48 Kio) : les poids de la ligne sont
  chargés UNE fois (uint4 par voie, comme v1) et dottés contre les TPB
  activations étagées ; les blocs non meneurs sortent aussitôt (coût : un
  bloc vide). Créneaux fantômes (e < 0) : zéro écrit, aucun poids lu.
- **Même arithmétique, même ordre** : `nvfp4_row_dot_warp_multi<TPB>` est
  `nvfp4_row_dot_warp` avec la boucle interne dupliquée par jeton (mêmes
  a0/a1/c0/c1, même `(…) * gscale`, même réduction par shuffles) → sortie
  identique AU BIT à v1 — le juge.
- Côté hôte : `ACVRAM_MOE_GEMV = v1 (défaut jusqu'au scellé) | v2`
  (regime.py, cli.py) ; `MoEBlock._forward_grouped` trie et appelle gateup
  v2 puis `_grouped(…, tri=)` pour down. Les chemins table/AWQ/distinct
  restent v1.
- Compilation contrôlée à sec : `nvcc -c` (C++20, sm_86 + sm_120) sans
  erreur ; pas de carte ici pour l'exécuter.

## Juge : `tests/test_gemv_experts_v2.py` (carte requise, 9 tests skippés à sec)

v2 = v1 `torch.equal` sur gate/up (K 2048, M 768) et down (K 768, M 2048),
routage aléatoire Coder, un expert qui reçoit les 12 jetons (sous-segments
> TPB), fantômes (zéros) ; gate/up fusionné v2 = v1 ; **bras cassant** :
`ordre` décalé d'un rang → différent ; tri stable vérifié.

## Porte : `outils/banc-gemv-experts-18-09.py` (poste3, ~5 min)

Coder b=12, 20 routages aléatoires (~69 experts distincts), gate/up + down,
rejeu de graphe, octets = experts distincts × (gate+up+down) ; chaque
routage jugé identique au bit ; JSON avec ms/pas (× 48 couches) et verdict :
**v2 ≤ 5,3 ms/pas ET identique** OUVRE.

Prédiction scellée : v1 ≈ 0,135-0,145 ms/couche (6,5-7 ms/pas, ≈ 1,2 To/s,
le 68 % de poste7) ; v2 ≈ 0,095-0,110 ms/couche (**4,6-5,3 ms/pas, 1,55-1,8
To/s**) — le gain vient des relectures évitées (÷ 1,39) et de l'étage
d'activation amorti ; faux si v2 > 5,3 (alors les blocs non meneurs coûtent
ou le segment remonté sérialise : mesurer TPB 2 vs 4 par
`ACVRAM_GROUPED_RPW` et un routage sans répétition, où v2 = v1 attendu ±
3 %). En situ ensuite (poste3, 20 min) : `ACVRAM_MOE_GEMV=v2` sur Coder
b=12 : ≥ 1 300 t/s nu (faux < 1 200), ppl-decode-kv identique (bit).
