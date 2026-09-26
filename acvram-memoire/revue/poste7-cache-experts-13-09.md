# poste7 — cache d'experts apprenant et double banque : conception pour acvram (13/09/2026)

Réponse à chef. Rien lancé, rien commité. Sources lues : `externes/colibri/c/`
(`expert_store.h`, `route_trace.h`, `deepseek_v4_bank_pair.h`, `colibri.c:8259-8300`
et `11214-11262`, `deepseek_v4.c:10501-10560`), notre `engine/model.py`,
`engine/layers.py`, `engine/graphs.py`, `engine/loader.py`, `memory/tiering.py`,
`memory/trace_routage.py`, `kernels/acvram_kernels.cu`, les deux manifestes cibles.

## 0. En cinq lignes

1. **Le compteur existe déjà et n'a jamais tourné** : `trace_routage.py:147`
   (`taux_de_succes`, LRU/LFU) — aucun fichier de trace sur les disques au 13/09.
   Première mesure : zéro ligne de moteur, une heure de carte.
2. **La double banque de colibrì est un mécanisme de PREFILL** (en-tête de
   `deepseek_v4_bank_pair.h:1`). Au décodage elle vaudrait 340 Mo par couche et par
   pas : impossible. Ne pas la transposer au décodage.
3. La bonne structure chez nous n'est pas un cache posé à côté des piles : c'est
   **le placement par expert au lieu du placement par couche**, avec une **table
   d'adresses** lue par le noyau groupé. Elle est compatible graphes par
   construction, et « placement = vitesse » y est vrai par construction.
4. **3080 Ti : banque calculante ou rien.** Une banque passive est impossible sans
   pair-à-pair. La calculante bat l'exil PCIe d'un facteur 8-20 sur le papier,
   avec trois mesures qui peuvent la tuer.
5. **Avant toute décision : deux bandes contradictoires dans nos propres
   documents** (18,7 contre 52,8 Go/s vers la 5090). Tout ce qui suit est chiffré
   dans les deux régimes ; le facteur 2,8 entre eux change les verdicts.

## 1. Les chiffres qui cadrent

| grandeur | Qwen3-Coder-30B-A3B | Qwen3.6-35B-A3B | source |
|---|---|---|---|
| couches / experts / actifs | 48 / 128 / 8 | 40 / 256 / 8 | `config.json` |
| octets d'un expert NVFP4 | 2,655 Mo | 1,778 Mo | `mlp_bytes / E`, manifeste |
| experts actifs par jeton et par couche | 21,38 Mo | 16,22 Mo | `mlp_active_bytes` |
| experts, total | 16,3 Go | 18,2 Go | 48 × 340 Mo ; 40 × 455 Mo |
| plan actuel | 48/48 résidentes | 40/40 résidentes | `plan.layers`, manifeste |
| pas de décodage b = 1 | ≈ 5,0-5,7 ms (174-201 t/s) | non mesuré | `FEUILLE-DE-ROUTE.md:1760,1800` |
| experts distincts par pas à b = 12, routage uniforme | 69 sur 128 (183 Mo) | 81 sur 256 (144 Mo) | 128·(1−(1−8/128)¹²) |

Bande PCIe vers la 5090 — **deux valeurs mesurées qui se contredisent** :

| valeur | où | quand | note |
|---|---|---|---|
| **18,7 Go/s** (x8 négocié, 59 % du théorique) | `docs/FEUILLE-DE-ROUTE.md:2403` ; c'est ce que portent les tiers des deux manifestes | 7/09 | 3080 Ti : 11,4 |
| **52,8 Go/s** h2d | `docs/MATERIEL.md:233` | non daté | 3080 Ti : 6,6 |
| 5,7 Go/s effectifs | `layers.py` (commentaire de `StreamedWeight`) | 8/09 | borné par la latence, 90 copies/couche — corrigé depuis (une copie plate) |

Coût de l'exil de 12 couches entières, par jeton (b = 1) : 12 × 21,38 Mo = 257 Mo →
**13,7 ms à 18,7 Go/s (pas ×3,7)**, 4,9 ms à 52,8 (pas ×2). Le facteur 3-4 mesuré
(`docs/TEST-EXIL.md:97-102`) est celui du régime 18,7 : c'est lui que le moteur
voit aujourd'hui. À b = 12 : 69 experts × 12 couches = 2,2 Go par pas → **118 ms à
18,7** (pas ×5-6 contre ~19-25 ms résident), 42 ms à 52,8.

**Contredit un acquis** : `estimer_cout_exil` (`tiering.py:1003`) chiffre avec les
bandes DÉCLARÉES des tiers — lien 18,7 et lecture VRAM **1792** — alors que la
lecture mesurée vaut ~1050 (`REGLES.md §9`). `t_pas_resident` est sous-estimé
×1,7, donc le `ratio` surestimé ×1,7 au moins, ×4,8 si le lien vaut 52,8. Le seuil
0,20 sonne trop tôt. À corriger avant toute décision fondée sur ce ratio (bead jt5,
item 1 — c'est le même geste).

## 2. Ce que colibrì fait exactement, lu dans le code

* `expert_store.h:16-38` : clé `(layer, expert)`, vue louée (`lookup`/`release`
  exactement une fois), `prefetch` **consultatif**, et une structure `stats` avec
  `hits / misses / prefetched / prefetch_hits / bytes_read` — **c'est le compteur
  qui nous manque, posé dans l'interface**, pas dans un outil à côté.
* `colibri.c:11214-11262` — `PIN=auto` : l'historique vivant `.coli_usage`
  (accumulé entre sessions, par `(couche, expert)`) bat le profil figé ; au
  chargement les plus chauds sont épinglés sur **la moitié du budget d'experts**,
  la LRU prend le reste ; `AUTOPIN` exige ≥ 5000 sélections d'historique avant de
  décider (quota proportionnel à la confiance).
* `colibri.c:8259-8270` — `REPIN=n` : entre deux tours, au point sûr, échange les
  pins les plus froids contre les non-pinnés les plus chauds ; **hystérésis 25 % + 4,
  au plus 4 échanges par passe**, carte de chaleur **décroissante** distincte de
  l'historique persistant.
* `deepseek_v4_bank_pair.h` + `deepseek_v4.c:10501` : pendant que la couche L
  calcule ses **morceaux de prefill**, un fil charge **les 256 experts** de L+1 dans
  la seconde banque sur un flux annexe ; décision pure `(double_on, other_layer,
  incoming_layer)` testée sans GPU ; un prefetch partiel vaut quand même (carte de
  validité par expert). Le transfert part **avant** de connaître le routage — c'est
  ce que paie le chargement de la couche entière.

Ce qui ne se transporte pas : la banque pleine couche au décodage. Chez nous une
couche de Coder-30B pèse 340 Mo : **18 ms par couche et par pas à 18,7 Go/s**, soit
0,87 s par jeton sur 48 couches. Au prefill d'une couche exilée le transfert de
340 Mo (18 ms) domine le calcul (512 jetons × 8 experts : ≈ 39 GFLOP, < 1 ms) :
le recouvrement cache ≤ 5 % du temps. **Pas de double banque chez nous**, ni au
décodage ni au prefill. Ce que colibrì appelle « refill route-aware on demand »
est ce que fait déjà `ExpertPool` (`layers.py:308`).

## 3. Ce que nous avons déjà (fichier, ligne) — et où chaque mécanisme s'arrête

| mécanisme | où | traité | non traité |
|---|---|---|---|
| trace de routage | `trace_routage.py`, appelée de `model.py:857` (tous les chemins passent par `_route`) | ordre d'émission, `.tolist()` assumé | **jamais prise** ; incompatible capture (sync) → à prendre en `ACVRAM_DISABLE_CUDA_GRAPHS=1` |
| `taux_de_succes(chemin, capacite, lru|lfu)` | `trace_routage.py:147-200` | LRU/LFU, capacité **globale** | pas de politique « épinglage statique », pas de capacité **par couche**, pas de « distincts par pas » à un lot donné |
| `cached_expert_fraction` | `tiering.py:596-622` | rapport de capacité (borne haute du coût, commenté honnêtement) | n'entre que dans `decode_tok_s`, **ne décide pas le plan** ; `expert_cache_bytes` réservé, rien ne l'implémente (`FEUILLE-DE-ROUTE.md:259`) |
| `ExpertPool` | `layers.py:308-400`, créé `loader.py:358` | `2k+2` emplacements, copie après routage, événements `pret/libre` | rien de préchargé ; décision **hôte** → impossible sous graphe |
| `StreamedWeight` | `layers.py:187` | double tampon dense = la « double banque » des poids denses, déjà là | experts : non |
| graphes | `graphs.py:143-225` (`_eligible`), clé `(b, ql, nblk, lb)`, `MAX_GRAPHS=16` | nomme le poids exilé qui coupe | **un seul poids en flux coupe les graphes du modèle entier** (`:205`) — aucune capture par segment |
| noyau groupé | `acvram_kernels.cu:632-650` : `qe = qw + e·M·half_k` | pile contiguë `[E, M, K/2]` | **l'adresse de l'expert est arithmétique** : le placement est cuit dans la pile ; idem `:712` (int4), `:1480`, `:1515` (gate·up fusionné), `:1599` (GEMM prefill) |
| instrument d'adresses | `graphs.py:78` `_empreinte_adresses` (`ACVRAM_TRACE_PTRS`) | relève toute adresse figée | c'est **le test** d'un cache sous graphe : zéro adresse changée entre capture et rejeu |

## 4. Structure minimale (question 1)

**Principe** : l'unité de placement devient l'expert, plus la couche. Une couche
« partiellement résidente » possède une pile VRAM de `C` emplacements (les chauds)
et une pile hôte épinglée de `E` experts (tous, mêmes octets), et **une table
d'adresses `[E]` en int64** — tenseur statique, contenu mutable — que le noyau lit
à la place de l'arithmétique `qw + e·stride`. Le noyau ne change pas de
mathématiques ; il change de pointeur. C'est ce qui rend « placement = vitesse,
jamais sémantique » **vrai par construction** et testable par identité bit à bit.

Les experts froids sont lus par le **même noyau** à travers l'adresse hôte
(mémoire épinglée mappée, `cudaHostGetDevicePointer`, lecture zéro-copie par le
PCIe) : pas de décision hôte, pas d'emplacement, pas d'événement, capturable dans
un graphe. Chaque octet d'un expert n'est lu qu'une fois par pas dans un GEMV
mémoire-borné : la copie explicite (`ExpertPool`) faisait PCIe + VRAM, la lecture
directe fait PCIe seul.

| où | quoi | taille |
|---|---|---|
| `kernels/acvram_kernels.cu:632-650` (+ `:712`, `:1480`, `:1515`) | argument `const long long *table` (nul = pile contiguë) ; `qe`, `be`, `gscale` pris de `table[e]` | ~40 lignes |
| `engine/model.py:612` `_try_build_stacks` | la pile devient `(format, qw_vram[C], bs, gs, k, m, table)` ; `_grouped` (`:665`) et `_forward_grouped` (`:771`) passent `table` | ~30 lignes |
| `engine/model.py:827` `_route` (ou `ext.moe_route`) | **compteur de chaleur sur la carte** : `atomicAdd` dans `compteurs[couche, E]` (int32 statique) — pas de synchronisation, capturable ; coût 1 lancement/couche exilée ≈ 3 µs, ou zéro si intégré à `moe_route` | ~20 lignes CUDA |
| `engine/loader.py:358` | pour une couche `experts_residents < E` : pile hôte plate (réutiliser `_emballer`), pile VRAM `[C]`, table initiale depuis le profil d'usage ; `routeur` reste résident comme aujourd'hui | ~60 lignes |
| `memory/tiering.py:108` `LayerPlan` | `experts_residents: int`, `taux_succes: Optional[float]` (**None = non mesuré → le plan garde `C/E`**, la borne haute d'aujourd'hui) ; `_estimate:660` et `estimer_cout_exil:1003` chiffrent `8·(1−h)·octets_expert / lien_mesuré` | ~40 lignes |
| `engine/runner.py` (entre deux pas) | **REPIN** : tous les N = 64 jetons, lecture des compteurs (48 × 128 × 4 o = 24 Ko), ≤ 4 échanges par couche et par passe, hystérésis 25 % + 4 comme colibrì ; copie hôte→VRAM **sur le flux de calcul**, puis écriture de la table — ordonné avant le rejeu suivant par le flux lui-même | ~60 lignes |
| `<dossier du modèle>/usage-experts.json` | profil persistant `(couche, expert) → compte`, écrit à la fin de chaque requête ; **jamais dans le manifeste** (le manifeste est la conversion, l'usage est celui de l'utilisateur) ; `PIN=auto` au chargement, quota proportionnel à l'historique (≥ 5000 sélections) | ~30 lignes |
| `graphs.py:205` | accepter une couche à table (résidente + zéro-copie) : elle n'est pas « en flux » | 5 lignes |

Ce que cette structure **interdit** : changer `C` en cours de service (les graphes
ont figé la pile), un format différent entre chaud et froid (INT2 au transit à la
HOBBIT = autres octets = autre sémantique : si un jour, c'est une décision de
conversion avec sa PPL, pas de placement), et une promotion int8 décidée par le
budget de VRAM — la promotion par SNR reste à la conversion, où elle est déjà.

Coût total estimé : 3-5 jours, dont un noyau. **Ne pas l'engager avant M1.**

## 5. Ordre des mesures qui réfutent (question 2) — prédictions et seuils posés ici

**M0 — la bande du lien, dans le moteur** (10 min carte, jt5 item 1).
`acvram bench --what bandwidth` + le débit effectif d'une copie plate de 21 Mo
depuis `StreamedWeight.plat`. Prédiction : **18,7** tient (x8 négocié) ; si 52,8,
tout coût d'exil ci-dessous se divise par 2,8. Écrire la valeur mesurée dans les
tiers (18,7 → mesure ; 1792 → 1050). Ne tue rien : met tout à l'échelle.

**M1 — taux de succès d'experts, sans rien changer** (1 h carte, 1 h script).
`ACVRAM_TRACE_ROUTAGE=trace-Qwen3-Coder-30B-decode-b1.txt ACVRAM_DISABLE_CUDA_GRAPHS=1`
(le nom porte le régime), ≥ 20 requêtes de code réelles, ≥ 50 000 jetons décodés,
puis la même chose à b = 12. Étendre `taux_de_succes` de trois choses : politique
`pin` (top-`C` par couche appris sur la **première moitié** de la trace, évalué sur
la **seconde** — jamais sur la moitié qui l'a appris), capacité **par couche**,
et « experts distincts par pas » au lot servi (pas = lignes de même base, une
par couche). Sortie : par couche, `h_pin(C)`, `h_lru(C)`, `C/E`, pour
`C ∈ {16, 32, 48, 64, 96}`, et distincts/pas à b = 1, 4, 12.

Prédictions (les miennes, datées 13/09, à me reprocher) :

| grandeur | prédiction | uniforme |
|---|---|---|
| Coder-30B, b = 1, `h_pin(64)` couches 8-40 | 0,72 ± 0,08 | 0,50 |
| Coder-30B, b = 1, `h_lru(64)` | 0,78 ± 0,08 | 0,50 |
| Coder-30B, b = 12, distincts par pas | 55-65 | 69 |
| Qwen3.6-35B, b = 1, `h_pin(128)` | 0,70 ± 0,10 | 0,50 |

**Seuil, fixé avant de voir** : le cache apprenant **ne peut pas payer** si
`Δh = h_pin(E/2) − 0,5 < 0,10` sur les couches candidates à l'exil. Arithmétique :
gain par couche exilée et par jeton = `8 · Δh · 2,655 Mo / 18,7 Go/s` = 0,11 ms à
Δh = 0,10 ; pour 12 couches, 1,4 ms sur un pas de 11,8 ms (12 %) — sous la
dispersion de nos bancs (26-37 % le 10/09), donc **indémontrable**, donc non payé.
À Δh = 0,22 (ma prédiction) : 3,0 ms, 25 % ; à 0,35 : 4,8 ms, 41 %. Les issues :

* `Δh < 0,10` : routage quasi uniforme → **abandonner le cache apprenant** ; l'exil
  se décide par couche sur `T_transfert` (jt5) et la 3080 Ti devient la seule
  piste (§6).
* `0,10 ≤ Δh < 0,25` : gain 12-28 %, à condition que ≥ 8 couches soient exilées ;
  sinon ne pas engager.
* `Δh ≥ 0,25` : engager §4.
* `h_lru − h_pin ≥ 0,10` : la localité temporelle domine → REPIN à N court (16),
  pas un pin figé au chargement.
* **L'issue qui gêne la question** : ces deux modèles tiennent en VRAM (17 et 19 Go
  sur 32) tant que services, KV et activations laissent ≥ ~24 Go — l'exil n'a lieu
  qu'à cause de ce qui est déjà sur la carte. Relever `nvidia-smi
  --query-compute-apps` services compris avant M1 : si l'exil n'existe pas pour
  ces deux modèles, le cache ne les concerne pas ; il concerne Qwen3-Coder-Next
  (512 experts) et nemotron-lightning (17,8 Gio / 32, celui des 12 couches).

**M2 — lecture zéro-copie** (30 min carte). Le noyau groupé lisant 21 Mo à travers
des pointeurs hôte : Go/s contre `cudaMemcpyAsync`. Prédiction : 70-85 % de la
copie (13-16 Go/s à 18,7). **Tue** : < 60 % → renoncer à la lecture directe, garder
la copie par noyau de rassemblement (ids prédits, tenseur statique) — et alors les
graphes exigent la segmentation (M4).

**M3 — prototype sur UNE couche** (2 h). (a) Identité : logits `torch.equal` entre
pile contiguë et pile à table, même noyau, 128 jetons — **tout écart = bogue**, pas
un seuil. (b) Temps : 12 couches à moitié résidentes, `pin` (top-64 du profil)
contre `aveugle` (ids 0-63), même VRAM. Prédiction : octets PCIe × (1−h)/(1−0,5) =
0,56 ; pas 8,8 ms contre 11,8 ms. **Tue** : gain < 15 % du pas à VRAM égale.
Jumelles obligatoires (`REGLES.md §4`).

**M4 — graphes par segment** (indépendant du cache, 1-2 jours). Capturer
`[0, i)` / eager couche exilée / `[i+1, L)`. Ordre de grandeur du gain : 11,9 %
mesuré sur un hybride pour la perte des graphes (`graphs.py:147`), et
`model.py:889` mesurait 2,7 ms de processeur pour 1,2 ms de carte par couche MoE en
eager. **Tue** : gain < 5 % du pas avec une seule couche exilée.

**M5 — joules** (`outils/gpu/mesure/energie.py`, compteur NVML, les deux cartes).
Chaque bras publie J/jeton, pas seulement ms. Prédiction : l'attente PCIe coûte
la puissance de repos de la 5090 pendant 13,7 ms/jeton — de l'ordre de 1,5 J/jeton
ajoutés (**non mesuré** : la puissance en attente n'est relevée nulle part).

## 6. La 3080 Ti (question 3) : banque calculante ou rien

Faits : 12 Go, sm_86 (ni FP4, ni conversion E2M1 : `REGLES.md §9`), lien mesuré
11,4 Go/s (7/09, x8) ou 6,6 (`MATERIEL.md`), pas de pair-à-pair, **elle héberge
déjà llama-server 8081 et TabbyAPI** (`ETABLI.md:1531`) — sa VRAM libre n'est pas
connue, à relever.

* **Banque passive** (octets sur la 3080 Ti, lus par la 5090) : chaque octet
  transite d2h 3080 Ti (6,4-11 Go/s) puis h2d 5090 — **pire que la RAM hôte**, qui
  est déjà le second saut. Rien.
* **Banque calculante** (style Fiddler) : les experts exilés vivent sur la 3080 Ti
  **avec les mêmes octets NVFP4**, déquantifiés par table de 16 entrées (le ×25 de
  `REGLES.md §9` concerne l'instruction `cvt` ; une LUT dans un GEMV
  mémoire-borné est gratuite), l'état caché traverse deux fois le PCIe par couche
  exilée (4 Ko à b = 1, 48 Ko à b = 12).

| régime | résident 5090 | exil RAM à 18,7 | 3080 Ti (prédit) |
|---|---|---|---|
| b = 1, par couche exilée | 0,03 ms | 1,14 ms | 21,4 Mo / ~550 Go/s = 0,04 + 2 sauts ~0,06 + lancements ~0,03 = **~0,13 ms** (÷9) |
| b = 12, par couche exilée | ~0,25 ms | 9,8 ms | 183 Mo / 550 = 0,33 + 0,09 = **~0,42 ms** (÷23) |
| capacité à 10 Go libres | — | — | 29 couches Coder-30B, 22 couches Qwen3.6 : **tout exil absorbé** |

Coûts : un noyau sm_86 (`nvfp4_gemv_grouped` avec déquant par LUT, ~200 lignes,
seconde cible dans `kernels/__init__.py:99`) ; un saut inter-cartes **par couche
exilée**, ce que l'en-tête de `tiering.py:29-33` interdit aujourd'hui (« une seule
frontière ») — dimension nouvelle, à poser une fois ; graphes : multi-appareils =
eager (`graphs.py:170`), donc M4 d'abord, capture multi-appareils ensuite.
Équivalence : deux noyaux n'accumulent pas dans le même ordre → **pas d'identité
bit à bit possible entre sm_86 et sm_120** ; le test est celui de `TEST-EXIL`
(PPL au dix-millième + accord glouton sur 3 × 256 jetons), et il doit tomber si
l'on falsifie une entrée de la LUT.

Mesures qui tuent, dans l'ordre : (i) aller-retour 5090 → hôte → 3080 Ti → hôte →
5090 d'un tenseur de 4 Ko : **> 0,5 ms** → le gain à b = 1 disparaît (12 × 0,5 =
6 ms ≈ exil RAM à 52,8) ; (ii) GEMV groupé sur la 3080 Ti **< 300 Go/s** effectifs ;
(iii) VRAM libre 3080 Ti services compris **< 4 Go** (banque < 12 couches) ;
(iv) `energie.py` : J/jeton du bras 3080 Ti ≥ bras exil RAM (elle est bridée
275-375 W, `ETABLI.md:167`, et son repos est déjà payé par 8081).

Verdict : la seule piste qui bat l'exil RAM d'un ordre de grandeur à b = 12 ; à
engager **après** M1 si l'exil est réel et **après** (i)-(iii), qui coûtent une
heure.

## 7. Pièges connus chez nous (question 4)

* **Graphes** : la clé `(b, ql, nblk, lb)` ne gagne aucune dimension — bien, tant
  que `C` est fixé au chargement pour la vie du serveur. La table d'adresses est un
  tenseur statique ; REPIN copie sur le flux de calcul avant le rejeu, jamais sur
  un flux annexe non joint. **Test** : `_empreinte_adresses` avant/après un REPIN
  → zéro adresse de pile changée ; une seule = accès fantôme.
* **`ExpertPool` sous graphe est impossible** : ses décisions sont hôte. Le chemin
  froid doit être zéro-copie (M2) ou un noyau de rassemblement à ids prédits.
* **La trace synchronise** (`trace_routage.py:171`) : sous capture elle échoue
  (`cudaErrorStreamCaptureUnsupported`) — prendre M1 en eager, et le compteur de
  chaleur de §4 ne doit jamais faire `.item()`.
* **« Placement = vitesse, jamais sémantique »** : chaud et froid = mêmes octets ;
  pas d'INT2 au transit ; la promotion int8 par SNR reste à la conversion, où
  elle est déjà documentée — le cache ne doit pas ouvrir un **second** endroit où la
  précision dépend du budget. Test d'équivalence **dans le même commit** : tout
  résident contre 12 couches à moitié résidentes, mêmes jetons gloutons sur
  3 × 256, PPL au dix-millième ; identité bit à bit quand le même noyau sert les
  deux (table seule).
* **Mécanismes arrêtés à une dimension** (`MECANISMES.md`) — le cache doit couvrir
  les cinq chemins qui lisent des experts : groupé décodage (`t ≤ 32`,
  `model.py:875`), boucle par expert (`_stack_state == "non"`,
  `ACVRAM_MOE_DECODE_MASQUES`, `:882`), prefill groupé (`_pile_bf16:675`
  déquantifie **toute** la pile : une pile à moitié hôte doit rassembler ou
  streamer la couche entière), processeur, hybride (Qwen3.6 : couches
  `linear_attention`, graphes à `b = 1`). Écrire la fraction couverte, pas la liste.
* **Bandes déclarées ≠ mesurées** (§1) : le ratio de `estimer_cout_exil` est faux
  ×1,7 à ×4,8. Corriger avant tout seuil.
* **Activations non provisionnées** : +3,06 Gio à 16 séquences
  (`chantier-activations-moe.md`). Un cache dimensionné sur « VRAM libre au
  chargement » sera chassé par l'OOM à b = 16 : dimensionner **après** KV et
  provision d'activations.
* **Les services** : quelle carte porte 8081/8082/8083 et combien — relever, pas
  supposer ; le « 12 Go libres » de la 3080 Ti est une supposition.
* **Le nom porte le régime** : `trace-<modèle>-<prefill|decode>-b<lot>.txt`,
  colonnes `h_pin`, `h_lru`, `C_sur_E`. Une trace à b = 1 ne prédit pas b = 12.

## 8. Ordre recommandé

| # | geste | gain attendu | coût | tué par |
|---|---|---|---|---|
| M0 | bande du lien mesurée dans le moteur, tiers corrigés (18,7 / 1050) | ratio d'exil juste (×1,7-4,8) | 10 min | rien ; met à l'échelle |
| M1 | trace + `taux_de_succes` étendu (`pin`, par couche, distincts/pas) | **décide tout** | 1 h carte + 1 h | Δh < 0,10 → pas de cache |
| M2 | zéro-copie | chemin froid sans hôte, sous graphe | 30 min | < 60 % de memcpy |
| §4 | placement par expert + table + PIN=auto + REPIN | pas b = 1 : 11,8 → 8,8 ms à Δh = 0,22 (25 %) | 3-5 j | M3 < 15 % à VRAM égale |
| M4 | graphes par segment | ordre de 12 % (hybride) ; à mesurer sur MoE | 1-2 j | < 5 % |
| §6 | 3080 Ti calculante, mêmes octets, LUT | ÷9 (b = 1), ÷23 (b = 12) par couche exilée | 3-5 j | saut > 0,5 ms ; < 300 Go/s ; < 4 Go ; joules |
| — | prefetch prédictif inter-couche (Fate, routeur l+1 sur h_l) | ≤ 25 % du temps des couches exilées (recouvre le calcul de l, ~0,1 ms sur 1,14) | élevé | précision < 90 % |
| — | double banque pleine couche | ≤ 5 % au prefill, impossible au décodage | — | déjà mort au chiffre |

## 9. Non vérifié, dit tel quel

* Puissance de la 5090 en attente PCIe : jamais relevée (M5).
* Bande effective du GEMV sur la 3080 Ti : 550 Go/s est 60 % de la plaque,
  supposé ; latence d'un saut inter-cartes : supposée ~30 µs.
* Le pas à b = 12 d'un MoE résident : le seul chiffre disponible est GLM-42B
  (replay 18,5 ms, `campagne-chrono-sync-11-09.md`) — pas Coder-30B.
* La laquelle des deux bandes (18,7 / 52,8) est celle de la machine d'aujourd'hui,
  après le changement d'OS du 12/09.
