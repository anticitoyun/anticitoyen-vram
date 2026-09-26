instrument : `kl-decode-200.py`/`compare-200.py` (en processus, gabarit kl2 de la 195) + `abba-200.sh` (ABAB×4, banc chat servi, gabarit cellule-190), tous sous `outils/carte.sh`, `ACVRAM_ATTENTE=5400`
commit : origin/main (be837ca1 + 196/198/199), branche poste2-p200, un seul worktree (portée = env au chargement, pas un code différent)
régime : mesure, plein (aucun repli réserve rencontré)
scellé : `scratchpad/poste2-p200-25-09/scelle.md`, écrit avant toute prise — prédiction vitesse +2 à +5 %, KL ≤ 2×max(T1,T2) à échantillon égal, falsificateurs B≤+1 %/B≥+8 %/B<A
mesuré : voir le détail par alias ci-dessous
verdict : Coder-30B no-op (bilan_marlin vide des deux côtés) ; Qwen3.5-35B qualité TENUE, vitesse b=8 TENUE dans la fourchette, b=1 falsificateur B<A DÉCLENCHÉ (régression nette, pas un bogue neuf)
durée : ~2 h dont ~50 min de file (carte très disputée ce tour, plusieurs prises > 30 min d'attente)

## Qwen3-Coder-30B-A3B(-nvfp4) : portée globale est un NO-OP, établi sans logits

`bilan_marlin` (imprimé directement depuis `model.proj_marlin_bilan`, `acvram/kernels/__init__.py:1112-1127`) est
**identique et vide des deux côtés** : `{doubles:0, seuls:0, exclus:0}` sous `denses` (`portee: denses:moe-exclu`)
ET sous `global` (`/tmp/p200-kl-coder3-B.log`). Cause trouvée dans le manifeste
(`/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4/acvram_manifest.json`, `weight_map` de
`layers.0.self_attn.{q,k,v,o}_proj`) : ces projections sont **déjà en INT8** (`qweight`/`scales`/`zeros`), pas en
NVFP4 — Marlin (nvfp4 uniquement) n'a **aucun candidat** hors experts sur cet alias, quelle que soit la portée.
Testé sur les deux variantes d'alias disponibles (`Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c` et `-nvfp4` natif) : même
résultat. **Aucune ABBA de vitesse jouée** — le résultat est déterministe (mêmes poids, même code), une mesure
n'aurait rien pu falsifier de plus que le bilan lui-même. La partie « Coder-30B » de la 199/200 ne s'applique donc
qu'aux alias MoE dont l'attention est encore en NVFP4 natif ; à vérifier au cas par cas avant toute autre pièce.

## Qwen3.5-35B-A3B-srcQ4_K_M-nvfp4 : candidats réels (GDN `linear_attn`, hors `MoEBlock`)

`bilan_marlin` sous `global` : `{doubles:0, seuls:111, exclus:60, replis:0}` (`/tmp/p200-kl-qwen35-B.log`) — la
portée a bien pris, 111 piles/poids NVFP4 (qkv/alpha/beta/gate/out de la récurrence linéaire GDN, `shared_expert`)
reçoivent la disposition dense. `role_marlin` (`:1132-1141`) ne nomme que `mlp.gate_up/down` et `gdn.out` : le
reste (qkv/alpha/beta/gate GDN) passe en piles « seules », pas en piles composées — gain probablement sous-optimal
par rapport à un `role_marlin` complété pour ces rôles (piste non explorée ici, hors périmètre de la 200).

**Qualité** (`compare-200.py` sur `dumps/qwen35-{A,B}-logits.pt`, 8 séq. × 78+32 pas, T1 = 8×b=1, T2 = rejeu de A,
comme la kl2 de la 195) — **TENU** : KL_AB_max **0,1101** ≤ seuil 2×max(T1,T2)=**0,6949** (T1_max 0,3474, T2=0,0) ;
argmax_AB **0,9414** ≥ argmax_T1_A−0,005 = 0,9208 ; PPL_A 15,539 vs PPL_B 15,472 (écart 0,4 %), ΔNLL fenêtre max
0,124. Aucun falsificateur de qualité déclenché.

**Vitesse b=8** (ABAB×4, `/tmp/p200-abba-qwen35-b8.log`) — **TENU, dans la fourchette prédite** : débit médian A
1 137,3 → B 1 188,75 t/s, **+4,52 %** (prédit +2 à +5 %) ; J/jeton net 0,2076 → 0,1771, **−14,72 %** (plus fort que
le débit seul ne l'annonçait : watts moyens 306,7 → 282,0, bridage « puissance » → « aucun » — sous `global` le
service ne heurte plus le plafond 400 W sur cette fenêtre, gain composé). 4/4 paires B > 4/4 paires A, aucun
chevauchement.

**Vitesse b=1** (ABAB×4, `/tmp/p200-abba-qwen35-b1.log`) — **FALSIFICATEUR B<A DÉCLENCHÉ** : débit médian A 252,0 →
B 244,1 t/s, **−3,13 %** (régression nette, 4/4 A à 252,0 ± 0,1, 4/4 B à 244,1 ± 0,1, aucun chevauchement). J/jeton
net légèrement meilleur (0,5705 → 0,5531, −3,05 %, watts 217,6 → 208,0) — la régression n'est PAS un bogue neuf :
elle reproduit le motif déjà connu de la 156 (Marlin dense « b=1 inchangé, 0,979 à 0,996 » sur les modèles purement
denses, `CHANGELOG.md:85-86`) — à M=1 le GEMV Marlin n'apporte rien et son surcoût de dispatch (plus de branches,
`_marlin_dense`/`_marlin_gemv_seul`) coûte net sur les petites projections GDN. **Pas de repli 157 déclenché sur
CE point** (la régression est arithmétique/dispatch, pas une exclusion de justesse).

## Repli 157 nommé (chef : « si l'un tombe en repli, dis-le »)

Sur `Qwen3.5-35B-A3B`, aux DEUX portées (`denses` et `global`, donc indépendant de cette pièce) : `[acvram]
disposition Marlin refusée : down_proj : 31 échelles sous-normales non représentables en Marlin (S0E5M3) — pile
naturelle gardée`. C'est le mécanisme de la pièce 157 (échelles NVFP4 sous-normales dans un poids d'expert), pas un
effet de la portée globale — présent avant cette pièce, sans conséquence sur les chiffres ci-dessus (il concerne un
poids d'expert, hors du périmètre « denses non-experts »).

## Recommandation

Ne pas basculer le défaut à `global` sans discrimination par b : le gain (+4,5 % débit, −15 % J/jeton) n'existe
qu'à b≥8 ; à b=1 c'est une perte nette de 3 %. Si chef veut ce gain, il faudrait soit une garde par b (comme
d'autres opt-in du parc), soit accepter la perte à b=1 comme prix d'une bascule simple. Sur Coder-30B (et
vraisemblablement toute la famille dont l'attention est déjà convertie en INT8), la question est sans objet :
aucun changement possible avec le mécanisme actuel.

## Fichiers hors git (sha256)

* `scratchpad/poste2-p200-25-09/dumps/qwen35-A-logits.pt` : `6b9176ff4d740942f66e892771ab06bffa7421171002e12815a36ce2ca1e9f11`
* `scratchpad/poste2-p200-25-09/dumps/qwen35-B-logits.pt` : `2c7968cb9de045018a740e51b4fa3aaff5808970aa04a850eb36674f36816c3e`
* `scratchpad/poste2-p200-25-09/dumps/coder-A-logits.pt` : `564e2a1a72d4d05bf4b272a705efca2e5528ac031a905626df678e284e14014c`
* `scratchpad/poste2-p200-25-09/dumps/coder-B-logits.pt` : `4c79d6132c75e99ad4f081697b3051c5d29a0b22b3283c2c79b234df588c79b0`
* `scratchpad/poste2-p200-25-09/dumps/coder2-A-logits.pt` : `5d3c6c8c47ef84a472640ac9e77125022df3bf71a74ee0913e1af73a3cbe57d4`
* `scratchpad/poste2-p200-25-09/dumps/coder2-B-logits.pt` : `0f654800063120ac3c5b29e4bb112f6545e5c49a701a8b233bb7565de39dc761`
* `scratchpad/poste2-p200-25-09/dumps/coder3-B-logits.pt` : `5a4db0ba5923d287e684bbef483d76643a4c7e57a509d1ca3d18a92c086100cc`
