instrument : outils/gpu/mesure/familles-b1.py (corrigé d599b0ceb), nsys profile+stats, CSV brut regroupé par grille
commit : poste2-p215 d08cb4f27, prise-familles.sh (scratchpad/poste2-p215-25-09/)
régime : plein (ordre chef, reprise post-pause)
scellé : 3 postes GDN nommés avant mesure (qkv/gate/out_proj) ; falsificateur : <60 % du delta attribué → faux
mesuré : b=1 (350,1 pas), b=8 (349,9 pas), portée Marlin globale, Qwen3.5-35B-A3B-srcQ4_K_M-nvfp4
verdict : qkv+gate CONFIRMÉS (Marlin, pas plain), out_proj NON TRANCHÉ sans NVTX — voir ## Verdict
durée : b=1 ~20 min (file + capture + post-traitement), b=8 ~35 min (file + capture)

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

## Addendum 26/09 02 h 30 — reconstruction par N (ordre chef), regex corrigé

Regex corrigé (1 ligne, `outils/gpu/mesure/familles-b1.py:12`, commit d599b0ceb). Reconstruction par N à partir
du CSV brut (b=1, `graphe_cuda_gpu_trace.csv`, groupé par (Nom, GrdX, GrdY, GrdZ, BlkX/Y/Z), N connus par
`acvram_manifest.json` couche 0) :

**Famille « autres » (nvfp4_gemv_kernel plain, hors Marlin)** — AUCUNE des trois shapes N=8192 (qkv) ou N=4096
(gate) n'y apparaît. Trois groupes seulement :
- grille (8,1,1), 21 060 lancements = 60,1/pas → `linear_attn.alpha` (N=32) **et** `linear_attn.beta` (N=32) :
  **même N, indiscernables par la grille** (le cas prédit par chef) — combiné 37,7 µs/pas, part négligeable ;
- grille (256,1,1), 40,1/pas → N=256 = `mlp.gate` (routeur, N=256, fire sur les 40 couches) — PAS un poste GDN ;
- grille (512,1,1), 40,1/pas → N=512 = `shared_expert.gate_proj` **et** `up_proj` (N=512 chacun, tie aussi, mais
  un seul groupe pour 40x/pas au lieu de 80x attendu si séparés → probablement fusionnés en une multi-projection
  à l'exécution, disposition non confirmée sans NVTX).

**qkv et gate ne sont PAS dans « autres » : ils tournent en Marlin** (confirmé « en Marlin seul quoi qu'il
arrive », doubles 25/09), donc comptés dans « experts » (1508,6 µs/pas). Isolés dans `marlin_only.csv` par grille :
- `nvfp4_gemv_marlin2_kernel<…>` grille (128,1,4), **30,1/pas** (= exactement les 30 couches `linear_attention`) :
  245,2 µs/pas total, 8,18 µs/lancement → **candidat `qkv`** (N=8192, le plus gros, le plus lent/lancement) ;
- `nvfp4_gemv_marlin2_kernel<…>` grille (64,1,8), **30,1/pas** : 157,5 µs/pas, 5,25 µs/lancement → **candidat
  `gate`** (N=4096, moitié du poids de qkv, durée cohérente memoire-liée ≈ proportionnelle aux octets) ;
- `out_proj` (N=2048, GDN-only) **non isolé** : aucun 3ᵉ groupe à ~30/pas trouvé côté Marlin ni côté plain — nommé
  par `role_marlin` (`gdn.out`), probablement piloté dans la pile doublée avec `shared_expert.down_proj` (tie de
  disposition, pas de N) → seul poste des 3 nommés qui reste non tranché sans NVTX (2ᵉ cas prévu par chef).

**Révision de la prédiction du 25/09** : qkv+gate confirmés comme postes dominants (402,7 µs/pas à eux deux sur
3532,7 noyaux_us_pas ≈ 11,4 % du temps noyau total, b=1) — accord qualitatif avec la prédiction (qkv > gate en
coût), mais le mécanisme nommé était faux (« Marlin seul » ⇒ ils étaient déjà dans « experts », pas « autres » :
la prédiction pointait la bonne cause, la mauvaise famille). `out_proj` reste à trancher : NVTX en opt-in dans
`gdn.py:_projections` si le chef le juge nécessaire, sinon le verdict se limite à qkv+gate, confirmé, out_proj
non isolé.

## Verdict — b=1 vs b=8, familles et postes GDN

b=1 (noyaux_us_pas 3532,7) : experts 1508,6 (43 %), autres+projections après correctif 341,1+297,2 = 638,3 inchangé
en somme (le correctif reclasse, ne change pas le total), glue_torch 816,3, attention 210,8.
b=8 (noyaux_us_pas 5699,9, +61 % vs b=1) : experts 3013,9 (53 %, part MONTANTE avec b — attendu, cf. 199 : 51-58 %
sur les autres MoE), autres 1031,3, projections 616,6, experts_glue 172,1 (quasi nul à b=1, 35,8 — le
regroupement d'experts routés pèse à b>1). **À b=8, le noyau dominant change de nature** : `marlin_moe_wna16::Marlin`
(grille [510x1x1], 1883,0 µs/pas, 72,8×) et `marlin::Marlin` générique ([170x1x1], 1054,0 µs/pas, 106,2×)
remplacent les `nvfp4_gemv_marlin2_kernel` étroits de b=1 — confirme le motif connu (156) : Marlin bascule vers
des noyaux tuilés/groupés amortis dès M>1, ce qui explique le +4,52 % TENU à b=8 (200) pendant que M=1 reste sur
la voie étroite pénalisée.

**Postes nommés dans la prédiction, verdict par poste** :
- `qkv` (N=8192) — CONFIRMÉ à b=1 : `nvfp4_gemv_marlin2_kernel` grille (128,1,4), 30,1/pas (= couches
  `linear_attention` exactement), 245,2 µs/pas, 8,18 µs/lancement. Poste le plus coûteux des deux isolés.
- `gate` (N=4096) — CONFIRMÉ à b=1 : grille (64,1,8), 30,1/pas, 157,5 µs/pas, 5,25 µs/lancement — moitié du poids
  de `qkv`, cohérent avec un noyau mémoire-lié (bande passante ∝ octets de poids, N moitié).
- `out_proj` (N=2048) — NON TRANCHÉ : aucun 3ᵉ groupe à ~30/pas isolé (ni Marlin narrow, ni plain), nommé par
  `role_marlin` (`gdn.out`) donc probablement piloté avec `shared_expert.down_proj` (tie de disposition, pas de
  N distinct côté grille) dans la pile doublée. Reste ouvert.
- `alpha`/`beta_proj` (N=32 chacun) — tie confirmé (grille 8,1,1, 60,1/pas = 2×30), part négligeable (37,7 µs/pas
  à b=1), non retenus comme poste dominant (conforme à la prédiction).

qkv+gate ensemble : 402,7 µs/pas sur 3532,7 (11,4 % du temps noyau total, b=1) — poste significatif, cohérent
avec le sens de la régression −3,13 % (200) même si le mécanisme prédit initialement (« projections plain mal
classées ») était erroné : elles sont bien en Marlin, simplement sur la voie étroite non amortie à M=1. Seuil de
falsification de la prédiction (§ ci-dessus, ≥ 60 % du delta) non évalué faute d'une capture nsys du témoin
« sans Marlin GDN » (portée denses) — hors périmètre de cette pièce, à demander si le chef veut la borne
quantitative complète plutôt que le mécanisme qualitatif rendu ici.

Repos sur 215 : verdict qualitatif rendu (qkv+gate confirmés, out_proj ouvert), 2 postes sur 3 tranchés sans
NVTX comme demandé.

## Blocage avant prise

`outils/carte-libre.sh` (25/09, avant réservation) : carte 0 occupée par PID 85799, **ni verrou (mesure) ni
journal (service)** — `python -m acvram.cli serve /mnt/AI...`, intrus au sens du script. Pas de prise lancée
tant que ce process n'est pas identifié/arrêté par son propriétaire ou que carte-libre.sh ne rend pas franc.
Pointeur envoyé à chef plutôt qu'action unilatérale (leçon `.qui orphelin après kill -9`, `pkill -f tue son
shell`).
