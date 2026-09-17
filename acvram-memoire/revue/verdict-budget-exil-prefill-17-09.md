# Verdict — le budget d'exil ne réservait pas les activations du préfill : Llama-3.3-70B-nvfp4 charge DÉGRADÉ puis OOM 448 Mio au premier préfill (poste3, verdict-palier1-bloc6-17-09)

poste4, 17/09, à sec, sur ordre de poste7 relayé par chef — un commit, test qui casse si le terme est retiré.

## Bogue : fichier:ligne (main bcba240)

`acvram/engine/loader.py:1268-1270`, dans `_reajuster_plan` : `marge = max(2 * 2**30, int(0.07 * capacite))` puis `while utilise() > capacite - marge:` — `utilise()` (l. 1213-1226) somme KV + embed + tête + attention et MLP résidents ; la marge de 7 % (2,3 Gio sur 32) est censée couvrir « contexte CUDA, activations, piles d'experts ». Même défaut sur le budget KV, `loader.py:1088-1090` (`_borner_kv_par_la_vram` : `borne = libre − poids − marge`, marge `max(1,5 Gio, 5 %)`).

Ce que le 70B demande au premier préfill et que ni l'une ni l'autre ne comptent : pour 2 048 jetons (H 8 192, I 28 672, 64 têtes q / 8 kv, D 128), ≈ 0,71 Gio d'activations par couche traversée (flux résiduel, ligne normée, sorties, q/k/v, gate/up/act bf16 + act fp32 : 371 Kio par jeton) **plus** la matrice déquantifiée en bf16 que le chemin W4A16 du préfill matérialise pour chaque projection au-delà du seuil GEMV (`kernels/__init__.py nvfp4_matmul` → `nvfp4_dequant`) : 28 672 × 8 192 × 2 = **470 Mio** — l'allocation de 448 Mio qui échoue. Total ≈ 1,15 Gio à 2 048 jetons, 3,3 Gio à 8 192, là où le plan avait rempli la carte jusqu'à la marge (arène hôte de 11,2 Gio à part, épinglée, pas de la VRAM).

Ligne de fiche pour poste3 : **acvram Llama-3.3-70B-nvfp4 = « refus : OOM au préfill en exil, bogue engine/loader.py:1268-1270 (marge d'exil sans activations de préfill) »**.

## Correctif (commit sur poste4)

- `engine/config.py` `ModelSpec.activations_prefill_bytes(n_jetons)` : par jeton 4·H + (HQ + 2·HKV)·D + 3·I (bf16) + I (fp32) — MoE : sur les `top_k` lignes routées (`moe_intermediate_size`) + expert partagé — plus, une fois, la plus grosse matrice déquantifiée (dense : max(I·H, qkv·H) ; experts : E·I_moe·H, la pile `_pile_bf16`) × 2 octets. Pas de logits (le moteur ne les calcule que pour les positions échantillonnées) ; un préfill groupé de plusieurs invites (`ACVRAM_PREFILL_BATCH`) n'est pas couvert, dit dans la docstring.
- `engine/loader.py` `_reserve_prefill(spec, max_model_len, manifest)` (ctx = max_model_len, sinon `kv_max_tokens` du manifeste, sinon 8 192 ; 0 sans spec) ; `_reajuster_plan(…, reserve=)` l. 1275 : `marge += reserve` ; `_borner_kv_par_la_vram(…, reserve=)` l. 1089 idem ; les deux journaux impriment la réserve ; appelants `_plan_from_manifest` (deux chemins) et `load_model`.
- `tests/test_budget_exil_prefill.py` : (1) l'estimation donne 0,9-1,5 Gio pour le 70B à 2 048 jetons, contient la déquant, croît avec la longueur ; (2) plan synthétique posé EXACTEMENT sous `capacité − marge` : `reserve=0` n'exile rien (l'ancien comportement → OOM), `reserve=activations(2 048)` exile 1-3 couches — retirer le terme fait rouge.

Prédiction pour le rejeu du 70B (bloc à part) : ≈ 34-36/80 couches exilées à 2 048 de contexte (1,15 Gio ≈ 3 MLP de 350 Mio), préfill qui passe, b=1 ≤ 5 t/s ; à `max_model_len` 8 192 la réserve monte à 3,3 Gio (≈ 9 MLP de plus) — le contexte demandé décide du nombre de couches exilées, c'est voulu.

Aussi dans ce commit (pas ce bogue) : `MoEBlock.forward` lit `_usage_routage` par `getattr` — un bloc factice des tests (`test_improvements`, `Faux(MoEBlock)` sans `__init__`) cassait la suite depuis que `ACVRAM_ROUTE_PREP=2` est le défaut.

## Suite (17/09, après 1aa767b fusionné) : l'OOM persiste — la cause était ailleurs

poste3, Llama-3.3-70B-nvfp4 après 1aa767b : 52/80 couches exilées, 31,0 Gio
occupés, même OOM 448 Mio au premier préfill. La réserve ci-dessus était juste
mais ne pouvait pas suffire : **exiler une couche dense ne libérait pas sa
VRAM, il la doublait.**

Fichier:ligne (main 428c653) : `acvram/engine/layers.py:188` —
`StreamedWeight.__init__(host_tensors, device, n_buffers=2, pool=None)` ; `:233-244`
`_ensure()` alloue `n_buffers` copies GPU de la couche (`torch.empty_like(self.plat,
device=self.device)`), jamais rendues, une paire PAR POIDS. Le pool partagé
n'existait que pour les experts MoE (`loader.py` : `ExpertPool(mlp_dev, 2·top_k+2)`
passé à `MoEBlock`) ; les linéaires denses exilés passaient par `lin()`/`mlin()`
→ `QuantLinear.to_device(d, streamed=True)` sans `pool`, donc deux tampons privés
chacun. Sur le 70B : 52 couches × (attn + mlp ≈ 0,36 Gio nvfp4) × 2 = **37 Gio
épinglés sur la carte pour des poids « exilés »** — plus que la carte. Le
`while utilise() > capacite − marge` (`loader.py:1270`) ne les compte pas
(`utilise()` somme les RÉSIDENTS), donc chaque itération d'exil aggravait ce
qu'elle croyait réduire ; le préfill trouvait 448 Mio de moins que rien.
Invisible sur les MoE (pool) et sur les denses qui tiennent (rien d'exilé).

Correctif (poste4) : `loader.py::_pool_dense(device)` — UN `ExpertPool(device,
_DENSE_SLOTS=4, dense=True)` par appareil, passé à `lin()`/`mlin()` quand
`streamed` ; `layers.py::ExpertPool(dense=)` ; `QuantLinear.prefetch()` précharge
depuis un pool dense (pas depuis un pool d'experts) ; `_reserve_prefill(…, plan)`
ajoute `_DENSE_SLOTS × max(attn+mlp)` (1,7 Gio sur le 70B) puisque ces tampons
sont désormais comptés une fois pour toutes. `ACVRAM_DENSE_SLOTS` hors régime
(regime.py, cli.py). Juge : `tests/test_pool_dense_exil.py` — réserve = 4 × plus
grosse couche, prefetch dense/experts, et sur carte : 12 poids d'une forme dans
le pool ≤ 4 tampons, un poids sans pool ≥ 2 copies (le bras qui doit différer).
Suite 772 passed.

Prédiction scellée pour l'essai du 70B (poste3, 2 048 ctx, budget par défaut) :
exil 34-40/80 couches ; VRAM de tampons dense ≤ 2 Gio ; préfill 2 048 jetons
passe ; décodage b=1 ≤ 5 j/s (lié à la bande PCIe : 40 couches × 0,36 Gio par
pas ≈ 14 Gio à 21 Go/s ≈ 0,7 s/pas ⇒ ~1,5 j/s plus probable). Faux si : OOM
encore (alors une troisième cause, à chercher avec `torch.cuda.memory_summary`
au moment de l'échec) ou exil > 52 (la réserve mange trop).
