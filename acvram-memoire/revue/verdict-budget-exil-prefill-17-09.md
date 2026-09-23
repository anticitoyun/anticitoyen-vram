# Verdict — le budget d'exil ne réservait pas les activations du préfill : Llama-3.3-70B-nvfp4 charge DÉGRADÉ puis OOM 448 Mio au premier préfill (Laure, verdict-palier1-bloc6-17-09)

Laurine, 17/09, à sec, sur ordre de Sage relayé par Jérôme — un commit, test qui casse si le terme est retiré.

## Bogue : fichier:ligne (main bcba240)

`acvram/engine/loader.py:1268-1270`, dans `_reajuster_plan` : `marge = max(2 * 2**30, int(0.07 * capacite))` puis `while utilise() > capacite - marge:` — `utilise()` (l. 1213-1226) somme KV + embed + tête + attention et MLP résidents ; la marge de 7 % (2,3 Gio sur 32) est censée couvrir « contexte CUDA, activations, piles d'experts ». Même défaut sur le budget KV, `loader.py:1088-1090` (`_borner_kv_par_la_vram` : `borne = libre − poids − marge`, marge `max(1,5 Gio, 5 %)`).

Ce que le 70B demande au premier préfill et que ni l'une ni l'autre ne comptent : pour 2 048 jetons (H 8 192, I 28 672, 64 têtes q / 8 kv, D 128), ≈ 0,71 Gio d'activations par couche traversée (flux résiduel, ligne normée, sorties, q/k/v, gate/up/act bf16 + act fp32 : 371 Kio par jeton) **plus** la matrice déquantifiée en bf16 que le chemin W4A16 du préfill matérialise pour chaque projection au-delà du seuil GEMV (`kernels/__init__.py nvfp4_matmul` → `nvfp4_dequant`) : 28 672 × 8 192 × 2 = **470 Mio** — l'allocation de 448 Mio qui échoue. Total ≈ 1,15 Gio à 2 048 jetons, 3,3 Gio à 8 192, là où le plan avait rempli la carte jusqu'à la marge (arène hôte de 11,2 Gio à part, épinglée, pas de la VRAM).

Ligne de fiche pour Laure : **acvram Llama-3.3-70B-nvfp4 = « refus : OOM au préfill en exil, bogue engine/loader.py:1268-1270 (marge d'exil sans activations de préfill) »**.

## Correctif (commit sur laurine)

- `engine/config.py` `ModelSpec.activations_prefill_bytes(n_jetons)` : par jeton 4·H + (HQ + 2·HKV)·D + 3·I (bf16) + I (fp32) — MoE : sur les `top_k` lignes routées (`moe_intermediate_size`) + expert partagé — plus, une fois, la plus grosse matrice déquantifiée (dense : max(I·H, qkv·H) ; experts : E·I_moe·H, la pile `_pile_bf16`) × 2 octets. Pas de logits (le moteur ne les calcule que pour les positions échantillonnées) ; un préfill groupé de plusieurs invites (`ACVRAM_PREFILL_BATCH`) n'est pas couvert, dit dans la docstring.
- `engine/loader.py` `_reserve_prefill(spec, max_model_len, manifest)` (ctx = max_model_len, sinon `kv_max_tokens` du manifeste, sinon 8 192 ; 0 sans spec) ; `_reajuster_plan(…, reserve=)` l. 1275 : `marge += reserve` ; `_borner_kv_par_la_vram(…, reserve=)` l. 1089 idem ; les deux journaux impriment la réserve ; appelants `_plan_from_manifest` (deux chemins) et `load_model`.
- `tests/test_budget_exil_prefill.py` : (1) l'estimation donne 0,9-1,5 Gio pour le 70B à 2 048 jetons, contient la déquant, croît avec la longueur ; (2) plan synthétique posé EXACTEMENT sous `capacité − marge` : `reserve=0` n'exile rien (l'ancien comportement → OOM), `reserve=activations(2 048)` exile 1-3 couches — retirer le terme fait rouge.

Prédiction pour le rejeu du 70B (bloc à part) : ≈ 34-36/80 couches exilées à 2 048 de contexte (1,15 Gio ≈ 3 MLP de 350 Mio), préfill qui passe, b=1 ≤ 5 t/s ; à `max_model_len` 8 192 la réserve monte à 3,3 Gio (≈ 9 MLP de plus) — le contexte demandé décide du nombre de couches exilées, c'est voulu.

Aussi dans ce commit (pas ce bogue) : `MoEBlock.forward` lit `_usage_routage` par `getattr` — un bloc factice des tests (`test_improvements`, `Faux(MoEBlock)` sans `__init__`) cassait la suite depuis que `ACVRAM_ROUTE_PREP=2` est le défaut.

## Suite (17/09, après 1aa767b fusionné) : l'OOM persiste — la cause était ailleurs

Laure, Llama-3.3-70B-nvfp4 après 1aa767b : 52/80 couches exilées, 31,0 Gio
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

Correctif (laurine) : `loader.py::_pool_dense(device)` — UN `ExpertPool(device,
_DENSE_SLOTS=4, dense=True)` par appareil, passé à `lin()`/`mlin()` quand
`streamed` ; `layers.py::ExpertPool(dense=)` ; `QuantLinear.prefetch()` précharge
depuis un pool dense (pas depuis un pool d'experts) ; `_reserve_prefill(…, plan)`
ajoute `_DENSE_SLOTS × max(attn+mlp)` (1,7 Gio sur le 70B) puisque ces tampons
sont désormais comptés une fois pour toutes. `ACVRAM_DENSE_SLOTS` hors régime
(regime.py, cli.py). Juge : `tests/test_pool_dense_exil.py` — réserve = 4 × plus
grosse couche, prefetch dense/experts, et sur carte : 12 poids d'une forme dans
le pool ≤ 4 tampons, un poids sans pool ≥ 2 copies (le bras qui doit différer).
Suite 772 passed.

Prédiction scellée pour l'essai du 70B (Laure, 2 048 ctx, budget par défaut) :
exil 34-40/80 couches ; VRAM de tampons dense ≤ 2 Gio ; préfill 2 048 jetons
passe ; décodage b=1 ≤ 5 j/s (lié à la bande PCIe : 40 couches × 0,36 Gio par
pas ≈ 14 Gio à 21 Go/s ≈ 0,7 s/pas ⇒ ~1,5 j/s plus probable). Faux si : OOM
encore (alors une troisième cause, à chercher avec `torch.cuda.memory_summary`
au moment de l'échec) ou exil > 52 (la réserve mange trop).

### Essai (Laure, 09:09-09:41, ee69a7b = main ce4014e)

- Instrument PPL, 3 tranches : exil **57/80** (ma borne « faux si > 52 » est
  dépassée : prédiction fausse sur ce montage — l'instrument charge sans
  godets b=1 ni HYBRID_SLOTS=1, sa réserve est plus grosse), arène 20,45 Gio,
  aucun OOM, PPL ≈ 22 (GGUF source à comparer). L'OOM du préfill est levé :
  c'était bien les deux copies privées.
- Rondes b=1 (certifie-b12, HYBRID_SLOTS=1, ctx 2048) : DÉGRADÉ 37/80 (dans
  34-40), arène 13,07 Gio, préfill passé, puis **blocage** : 33 min sans une
  ligne, GPU 0 %, 17-22 W ; fil principal en `poll` (attente CUDA bloquante),
  36 fils python en futex (pool torch au repos), cuda-EvtHandlr en poll.
  Aucun verrou Python dans le moteur (runner.py:382 seul) : le flux de calcul
  attend un événement du flux de copie du pool, ou l'inverse. Pile Python
  illisible (ptrace_scope=1, processus sous setsid, pas de sudo). Tué 09:41.
- Témoin lancé 09:42 : même commande avec `ACVRAM_POOL_SYNC=1` (layers.py:365 :
  copies sur le flux de calcul, sans flux annexe ni événement). Prédiction
  scellée avant : décode (~1 j/s) ⇒ cause dans l'ordonnancement événements/flux
  du pool appliqué aux denses ; bloque aussi ⇒ pas le pool, « refus » publié.

### Cause du « blocage » (Laure, essai TRACE_COUCHES 70082d4, dmon 10 min)

Aucune ligne de couche, pas même la 0 ; rxpci 0-3 Mo/s, sm 0-2 % : **le
modèle n'est jamais appelé.** La ligne présente dans les trois essais :
« budget KV de cuda:0 borné par la VRAM libre : 0,32 → 0,00 Gio (libre 28,7,
poids 24,5, marge 4,5 dont préfill 2,94) » (`loader.py:1117`). Zéro octet
de KV → `_kv_blocks_per_device` rend `max(1, …)` = 1 bloc de 16 jetons ;
l'invite de 256 jetons (17 blocs) n'est jamais admise (`runner.py:_admit`
: `need > num_free → break`, sans fin) ; le moteur tourne à vide, fil
principal en poll, fils en futex. Ni le pool, ni un cycle CUDA : une
requête inadmissible gardée en file. Le POOL_SYNC et la trace ne pouvaient
rien voir — le premier pas n'a jamais existé.

Deux mécanismes se contredisaient : la boucle d'exil (`_reajuster_plan`,
marge 2 Gio + 7 % + réserve, VRAM libre lue AVANT) s'arrêtait satisfaite à
37/80 ; la borne KV (`_borner_kv_par_la_vram`, marge 1,5 Gio + 5 % + réserve,
VRAM libre lue APRÈS le JIT) trouvait 2 Gio de moins et prenait sur le KV
— jusqu'à zéro, sans refuser. Correctif (laurine) :
- `loader._kv_plancher` : octets KV d'UNE séquence de `max_model_len`
  jetons sur l'appareil (0,33 Gio sur le 70B à 2 048) ;
  `_borner_kv_avec_exil` : borne, puis si le budget est sous le plancher,
  exil supplémentaire (`_reajuster_plan` avec `reserve + manque`) et
  re-borne, 4 tours ; encore sous le plancher → **`RuntimeError` « refus :
  budget KV insuffisant »** avec les chiffres (un chargement qui ne peut
  servir une requête s'arrête, il n'attend pas).
- `runner._admit` : `need > allocator.num_blocks` → la requête est finie
  `finish_reason="refus"`, sortie rendue au pas suivant, ligne
  `[acvram] requête refusée : N blocs KV nécessaires, M en tout`.
- Juge : `tests/test_kv_plancher_exil.py` — plancher = 166 400 × (2 048+16) ;
  montage qui reproduit le KV à zéro (bras témoin) puis exil supplémentaire
  jusqu'au plancher ; refus explicite quand rien ne suffit ; moteur CPU à
  1 bloc : la requête de 40 jetons sort « refus » au premier pas, une courte
  passe ensuite. Suite 776 passed.

Prédiction pour l'essai suivant (Laure, même certifie b=1, ctx 2 048) : exil
≈ 40-44/80 (une à deux couches de plus que 37, pour 0,33 Gio de KV + l'écart
de 2 Gio entre les deux lectures de VRAM), budget KV ≥ 0,33 Gio, la première
ligne CERT sort en moins de 5 min, décodage b=1 ≈ 1-1,5 j/s (PCIe : 40 ×
0,36 Gio par pas à ~21 Go/s ≈ 0,7 s). Faux si : RuntimeError « refus »
(alors la borne mange plus que 4 tours ne rattrapent : lire les chiffres),
ou moteur à vide encore (alors une troisième file d'attente, à chercher
avec `ACVRAM_TRACE_STEPS=1`).

### Essai a803254 (Laure) : refus en 7 s, « 3 Mio manquants » × 4 tours — ma faute

Le plancher comptait `max_model_len + 16` jetons (un bloc de marge) : 3 Mio
au-dessus de la cible du planificateur (0,32 Gio = 166 400 × 2 048), et la
borne ne fait que RÉDUIRE une cible — quatre tours d'exil (41 MLP au lieu
de 37) laissaient le budget à 0,32, toujours 3 Mio sous un plancher
inatteignable. Prédiction fausse par construction, pas par la carte.
Correctif (laurine) : plancher = exactement `kv_bytes_per_token ×
max_model_len` (la convention d'`auto_plan`) ; et la cible est d'abord
relevée au plancher si le plan lui accordait moins, la borne fait le reste.
Test ajouté : cible 100 Mio sous le plancher → relevée puis bornée au
plancher. Suite 777 passed. Même scellé que ci-dessus pour l'essai suivant,
plus : le journal doit montrer un seul tour d'exil supplémentaire (le
premier suffit : 0,32 Gio revenaient dès le tour 1 chez Laure).

### Exil total 36/36 : « ExpertPool saturé : 4 emplacements » (Laure 02b316d) — corrigé

`ExpertPool.copier` prenait l'emplacement du TOUR DE RÔLE et refusait s'il
était en vol, sans regarder les autres. Séquence du forward (`model.py`,
précharge i+1 puis exécute i) quand la couche 0 est elle-même en flux :
précharge(1) prend s0,s1 ; la couche 0, jamais préchargée, copie à la
demande s2 puis s3 et les rend ; précharge(2) tombe sur s0, en vol, alors
que s2 et s3 sont libres. En exil partiel, la première couche en flux est
préchargée par sa voisine résidente et l'ordre des rendus reste celui des
prises. Correctif : premier emplacement LIBRE à partir du tour de rôle ; le
refus reste quand tout est en vol. Juge : `test_pool_dense_exil.py::
test_le_pool_prend_un_emplacement_libre_pas_le_tour_de_role` (la séquence
ci-dessus, puis saturation réelle → RuntimeError). Suite 778 passed.

### Régression de la réserve (Laure, fla-17-09) : 13,04 Gio réservés sans `max_model_len` — corrigée

`_reserve_prefill` prenait, faute de `max_model_len`, `kv_max_tokens` du
manifeste : la CAPACITÉ KV planifiée toutes séquences (56 401 jetons sur
Qwen3.8-27B), pas une longueur d'invite → 13,04 Gio d'activations réservées,
19/64 couches exilées d'un modèle de 13,1 Gio, instrument de préfill refusé
(« régime dégradé ») ; mêmes refus sur Kimi (147 experts) et Nemotron au
palier 2. Correctif : `ctx = max_model_len or 8192` (le défaut du moteur),
jamais `kv_max_tokens` ; test : 56 401 planifiés → réserve de 8 192, et de
4 096 quand `max_model_len` le dit. Sur Qwen3.8 à 4 096 : 1,05 Gio
d'activations + 0,8 Gio de tampons denses = 1,85 Gio.
