instrument : outils/gpu/mesure/familles-b1.py, familles-comparees.py, nsys-tete-sampler.sh (Qwen3.5-35B-A3B, b=1 et b=8)
commit : branche poste2-p215 depuis origin/main 1b56f68e9
régime : plein (ordre chef, reprise post-pause)
scellé : voir prédiction ci-dessous, écrite avant toute prise
mesuré : —
verdict : —
durée : —

## Prédiction (avant mesure)

Profil par famille Qwen3.5-35B-A3B (b=1, b=8) : lacune de la 199 (aucun profil), à combler par nsys +
`familles-b1.py`. Attendu : experts (MoEBlock, non-nommable) dominent b=8 comme sur les autres MoE de la série
(51-58 % du pas, cf. 199) ; à b=1 part experts plus faible en proportion (moins de recouvrement mémoire/calcul),
GDN + attention + glue montent en part relative.

Explication de la régression −3,13 % (200, b=1, GDN Marlin portée globale) : 5 projections GDN passent en NVFP4
Marlin (`qkv`, `gate`, `alpha`, `beta_proj`, `out_proj`, cf. `acvram/engine/gdn.py:181-182`). Seuls `gdn.out` et
`shared_expert.{gate,up,down}` sont nommés par `role_marlin` (kernels/__init__.py:1189) donc éligibles à la
disposition doublée — mais le doublage ne change rien à la régression (pièce doubles, 25/09) : les grosses
formes `qkv` (8192×1024) et `gate` (4096×1024) restent en Marlin seul quelle que soit la disposition. À M=1, un
GEMV est mémoire-lié par construction (comme la 99 : 35-40 pJ/bit, int8 Triton = Marlin nvfp4) — la perte n'est
pas un défaut d'arithmétique mais le surcoût par-tuile du noyau Marlin (dequant groupé, ordonnancement pensé
pour M≥16), non amortie à M=1, motif déjà vu en 156 (Marlin dense neutre/négatif à b=1).

**3 postes nommés, par poids approximatif (lignes × colonnes, proportion du footprint GDN-Marlin hors
alpha/beta, négligeables en octets)** :
1. `qkv` (8192×1024) — ≈ 50 % du footprint GDN-Marlin → gain max prédit si le surcoût par-tuile tombait à zéro :
   ≈ −1,5 à −1,6 pt sur les −3,13 % mesurés.
2. `gate` (4096×1024) — ≈ 25 % → gain max prédit ≈ −0,8 pt.
3. `out_proj` (nommé `gdn.out`, taille du même ordre que `gate`) — ≈ 25 % → gain max prédit ≈ −0,8 pt.
`alpha`/`beta_proj` (porte par tête, colonnes ≪ 1024) : part négligeable, non retenus comme poste.

Seuil de falsification : si nsys attribue à `qkv`+`gate`+`out_proj` (famille `proj_dense`/`gdn`, durée noyau
seule) moins de 60 % du delta µs/pas entre acvram-nvfp4-global et le témoin sans Marlin GDN, la prédiction est
FAUSSE — chercher ailleurs (glue, dequant hôte, ordonnancement de piles).

## Addendum 26/09 02 h — b=1 capturé, taxonomie fausse sur les GEMV GDN

Prise b=1 (nsys, portée globale, `scratchpad/poste2-p215-25-09/b1/`) : `noyaux_us_pas` 3532,7, famille
`experts` 1508,6 (43 %), `autres` 638,3 (18 %) — 2ᵉ poste. Défaut trouvé dans `familles-b1.py:12` : motif
`nvfp4_gemv\b` ne matche pas `nvfp4_gemv_kernel<...>` (le `_` après `gemv` casse la frontière de mot `\b`) → tous
les GEMV étroits nvfp4 (dont les projections GDN `qkv`/`gate`/`alpha`/`beta_proj`, non nommables par
`role_marlin`) tombent dans « autres » au lieu de « projections ». `fused_recurrent_gated_delta_rule_fwd_kernel`
(30×/pas = exactement les 30 couches `linear_attention`, cohérent) y est aussi, à raison — c'est la récurrence,
pas une projection.

Limite trouvée : le nom de noyau + dimensions de grille (`<256x1x1>`, `<512x1x1>`, `<8x1x1>`…) ne distingue PAS
`qkv` de `gate` de `alpha`/`beta_proj` — plusieurs projections de tailles différentes partagent le même template
générique. Attribution fine par poste (les 3 nommés dans la prédiction) impossible sans marquage NVTX par
couche/projection, absent de l'instrumentation actuelle. Recommandation : soit corriger le regex (poste séparé,
mineur, 1 ligne) pour au moins sortir ces noyaux d'« autres » vers « projections », soit ajouter des plages NVTX
nommées dans `gdn.py:_projections` si le partage par poste (qkv vs gate vs alpha/beta vs out) est exigé par le
chef. Sans ça, le verdict final ne pourra confirmer/infirmer la prédiction qu'au niveau « projections GDN
globales », pas poste par poste.

b=8 lancé (`ACVRAM_NOM=poste2-p215-b8`), sortie `scratchpad/poste2-p215-25-09/b8/`.

## Blocage avant prise

`outils/carte-libre.sh` (25/09, avant réservation) : carte 0 occupée par PID 85799, **ni verrou (mesure) ni
journal (service)** — `python -m acvram.cli serve /mnt/AI...`, intrus au sens du script. Pas de prise lancée
tant que ce process n'est pas identifié/arrêté par son propriétaire ou que carte-libre.sh ne rend pas franc.
Pointeur envoyé à chef plutôt qu'action unilatérale (leçon `.qui orphelin après kill -9`, `pkill -f tue son
shell`).
