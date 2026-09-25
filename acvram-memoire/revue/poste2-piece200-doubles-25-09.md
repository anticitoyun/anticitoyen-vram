instrument : lecture de code (`acvram/kernels/__init__.py:1040-1046,1132-1141,1216-1236`, `acvram/engine/loader.py:1301-1334`,
`acvram/engine/gdn.py:181-182`) + manifeste/config du modèle (`safetensors` header, `config.json`) — aucune prise
commit : origin/main (be837ca1 + 200), branche poste2-p200
régime : à sec, 0 min de carte
scellé : aucun — question de chef, chiffrage avant toute mesure future
mesuré : rien de nouveau ; extrapolation à partir de la 129 (Qwen3.8-27B) et des shapes réelles du 35B
verdict : la disposition « doubles » (156/129) s'applique à une PETITE PART du 35B (gdn.out + MLP partagé), coût
mémoire faible (~0,24 Gio), gain attendu marginal — elle ne corrige PAS la régression b=1 mesurée en 200
durée : ~20 min de lecture

## La question : le 35B a-t-il droit à « doubles » comme le 129 l'a posé pour Qwen3.8 ?

`_PROJ_MARLIN_DOUBLES` (`acvram/kernels/__init__.py:1040-1042`) est un ensemble de RÔLES (pas de tenseurs), lu depuis
`ACVRAM_PROJ_MARLIN_DOUBLES` (défaut vide = aucun doublé, tout en disposition unique quand Marlin est pris). Un poids
n'est éligible au statut « doublé » QUE si `role_marlin(module, attr)` lui donne un nom
(`:1132-1141`) : seuls `mlp.gate_up`, `mlp.down` (classe `MLP`) et `gdn.out` (classe `GatedDeltaNet`, attribut
`out_proj`, `acvram/engine/gdn.py:182`) sont nommés. **Tout le reste garde `role=""` et ne peut JAMAIS être doublé,
quelle que soit la valeur de la variable** — disposition unique (Marlin seul) ou naturelle seule, jamais les deux.

## Ce qui, sur le 35B, est nommable — et ce qui ne l'est pas

Sur `Qwen3.5-35B-A3B-srcQ4_K_M-nvfp4` (40 couches, hybride GDN + MoE, shapes lues dans le safetensors) :

| tenseur (par couche) | rôle | nommable → doublable | shape (N, K empaqueté) |
|---|---|---|---|
| `linear_attn.out` | `gdn.out` | **oui** | 2048 × 2048 |
| `mlp.shared_expert.gate_proj` | `mlp.gate_up` | **oui** | 512 × 1024 |
| `mlp.shared_expert.up_proj` | `mlp.gate_up` | **oui** | 512 × 1024 |
| `mlp.shared_expert.down_proj` | `mlp.down` | **oui** | 2048 × 256 |
| `linear_attn.qkv` | *(aucun)* | **non** | 8192 × 1024 |
| `linear_attn.gate` | *(aucun)* | **non** | 4096 × 1024 |
| `linear_attn.alpha`, `.beta` | *(aucun)* | **non** | 32 × 1024 chacun |

**Point central** : les deux plus grosses formes du GDN (`qkv` 8192×1024, `gate` 4096×1024 — 2 à 4× plus grosses que
`out`) ne sont PAS nommées par `role_marlin` et resteraient donc en disposition UNIQUE (Marlin seul) même avec
`ACVRAM_PROJ_MARLIN_DOUBLES` posé. Le mécanisme actuel ne peut doubler que `out` + le petit MLP partagé.

## Coût mémoire (estimation, formule du code : `qweight.numel() + block_scale.numel()`, `:1225`)

* `gdn.out` : qweight 2048×2048 = 4,00 Mio + block_scale ≈ 0,50 Mio ≈ **4,50 Mio/couche**
* `mlp.shared_expert.{gate,up}_proj` : 2 × (512×1024 = 0,50 Mio + ≈0,06) ≈ **1,12 Mio/couche**
* `mlp.shared_expert.down_proj` : 2048×256 = 0,50 Mio + ≈0,06 ≈ **0,56 Mio/couche**
* **Total ≈ 6,18 Mio/couche × 40 = 247 Mio ≈ 0,24 Gio.**

À comparer à la 129 sur Qwen3.8-27B (denses purs, gate‖up 34 816×5 120 et down 5 120×17 408, 48 couches) :
**10,6 Go** pour les trois mêmes catégories de rôle (`revue/verdict-129-1-gemv-marlin-m1-24-09.md:29-30`) — **44×
moins** sur le 35B, parce que la partie « MoE partagée » de ce modèle (`shared_expert_intermediate_size=512`) est
minuscule comparée au MLP dense complet d'un modèle purement dense ; le gros du calcul MoE (256 experts routés,
`moe_intermediate_size=512` chacun) n'est ni nommable ni éligible Marlin par construction (`sous_moe`, pièce 156).

## Effet sur la capacité KV

0,24 Gio sur une carte à 32 Go, avec ce modèle déjà chargé à `kv_budget=32768/8` sans tension mémoire visible dans
la prise 200 (`bilan_marlin` avec `capacite_kv=8192`... par slot, aucune invalidation mémoire) : l'impact attendu
est de l'ordre de **quelques centaines de jetons de capacité KV, pas un pourcentage à deux chiffres** — sans commune
mesure avec le −41 % mesuré sur Qwen3.8-27B (`verdict-129-2-memoire-kl-24-09.md:17-18`, où gate‖up+down+GDN out
pesaient 10,6 Go sur un budget bien plus serré). `_verifier_memoire_marlin` (`loader.py:1301-1334`) refuserait au
chargement, nommé, si ce n'était pas le cas — donc pas d'OOM silencieux à craindre, seulement un refus visible si
l'estimation ci-dessus est fausse.

## Gain attendu, b=1 et b=8 — et pourquoi il ne règle pas la régression de la 200

À l'échelle de la 129 (perte b=1 en % du GEMV concerné, gain b=8 du même GEMV, `verdict-129-1:18-27`), les FORMES
nommables du 35B (`out` 2048×2048, MLP partagé 512×1024/2048×256) sont **petites** — largement plus petites que
`qkv` (8192×1024) et `gate` (4096×1024), qui dominent probablement le temps GEMV de la couche GDN et qui, eux,
**resteraient en Marlin seul, doubles ou pas**. La régression de −3,13 % mesurée à b=1 dans la pièce 200 vient très
vraisemblablement de `qkv`/`gate` (les grosses formes non nommables), PAS de `out`/MLP partagé. **Poser
`ACVRAM_PROJ_MARLIN_DOUBLES=gdn.out,mlp.gate_up,mlp.down` récupérerait une fraction marginale de cette perte** (les
formes concernées sont petites), **pas la régression observée** — et n'apporterait qu'un gain b=8 tout aussi
marginal sur ces mêmes petites formes.

## Recommandation

« doubles » est un droit théorique pour le 35B (mécanisme compatible, coût mémoire négligeable, aucun refus attendu
au chargement), mais un mauvais levier pour le problème réel : la régression b=1 de la 200 est portée par des
formes que `role_marlin` ne sait pas nommer. Pour la corriger vraiment, il faudrait soit étendre `role_marlin` pour
nommer `linear_attn.qkv`/`.gate` (code neuf, hors périmètre d'une simple variable d'environnement), soit accepter la
perte à b=1 comme prix du gain b=8 (déjà la position prise dans `revue/poste2-piece200-25-09.md`). Poser `doubles`
sur les seules formes nommables aujourd'hui n'est pas nuisible mais ne vaut probablement pas une pièce dédiée avant
d'avoir mesuré la part réelle de `out`/MLP-partagé dans le pas — hors périmètre de cette page (à sec, sans prise).
