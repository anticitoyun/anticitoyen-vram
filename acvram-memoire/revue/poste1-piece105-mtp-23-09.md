# Pièce 105 — dossier : spéculation MTP sur la famille Qwen3.5 (Qwen3.8-27B-nvfp4) — à sec, aucun code — 23/09 (poste1)

Références : `revue/veille-ninfer-23-09.md` (NInfer, RTX 5090, MTP3 : Qwen3.8-27B nvfp4 143,8 t/s à C=1, acceptation
48,9 % ; 766,6 t/s à C=8) ; `verdict-mtp-exact-19-09.md` (MTP sur hybride GDN NON MESURABLE le 19/09) ;
`poste2-piece102-scelle-ninfer-23-09.md` (notre b=1 sans spéculation prédit 70-85 t/s, b=8 300-420 t/s).

## 1. Les têtes MTP sont-elles dans nos conversions ? OUI — mais le chargeur ne les voit pas

* **Source** `Qwen3.8-27B-bf16` : 15 tenseurs `mtp.*` (config : `mtp_num_hidden_layers: 1`,
  `mtp_use_dedicated_embeddings: false`, donc plongement et `lm_head` partagés).
* **Conversions** `Qwen3.8-27B-nvfp4`, `…-calibA`, `…-srcexl3_6_00bpw-nvfp4` : les 15 sont **présents**, sous leurs
  noms HF.
  * `mtp.fc.weight` [5 120 × 10 240] en bf16 ;
  * `mtp.layers.0.self_attn.{q [12 288 × 5 120 : q + porte], k, v, o}_proj` et `mlp.{gate, up, down}_proj` en nvfp4 ;
  * `mtp.layers.0.{input, post_attention}_layernorm`, `self_attn.{q, k}_norm`, `mtp.norm` et
    `mtp.pre_fc_norm_{embedding, hidden}` en bf16.
* **Pourquoi aucune tête n'est chargée** :
  * `engine/mtp.py:66-75` (`cles_mtp`) ne reconnaît que le préfixe `model.mtp.<n>.`, la convention du GGUF (`quant/gguf.py:298-316` y renomme les couches `nextn`) et de DeepSeek.
  * `engine/loader.py:991-1031` (`_charger_mtp`) lit ensuite `eh_proj`, `enorm`, `hnorm` et `shared_head_norm`/`norm` sous ce préfixe. Les noms Qwen3.5 (`mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm`, `mtp.layers.0.*`) ne correspondent à aucun.
  * Résultat : `model.mtp` vaut None, et `--speculative auto` (`cli.py:815-817`) retombe sur n-gram, sans le dire.
* **Défaut de conversion trouvé au passage (normes zéro-centrées)** :
  * `quant/convert.py:824-826` (`_NORMES_ZERO_CENTREES`) ajoute +1 par suffixe. Les normes de `mtp.layers.0.*` (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`) sont donc décalées, mais **pas** `mtp.norm`, `mtp.pre_fc_norm_embedding` ni `mtp.pre_fc_norm_hidden`, qui ne finissent par aucun suffixe listé.
  * Avec notre `RMSNorm` (x·w), ces trois normes vaudraient ≈ 0 : la tête produirait du bruit.
  * Le correctif se fait au chargement (+1 sur ces trois noms pour `_QWEN35_HF`, sans reconversion), ou à la conversion, qui exige alors une reconversion.
* **Déjà juste** :
  * l'attention à porte de la couche MTP (`loader.py:1017-1020`, `output_gate=spec.attn_output_gate`) ;
  * l'ordre `fc([norm_e(e) ; norm_h(h)])`, plongement d'abord (`mtp.py:59-63`, identique à vLLM `qwen3_5_mtp.py:159-162`).

## 2. Points d'insertion (fichier:ligne), face au n-gram

| rôle | n-gram (servi) | MTP (existant) | à faire pour Qwen3.5 |
|---|---|---|---|
| choix du proposeur | `cli.py:818-820` (`NGramProposer`) ; défaut `--speculative ngram` (`cli.py:1239`) | `cli.py:831-835` ; `auto` → mtp si `model.mtp` | rien |
| brouillon | `speculative.py:133-220` (`NGramProposer`, recherche dans l'historique, sans passe de modèle) | `speculative.py:464-594` (`MTPProposer`) : k passes eager d'un bloc + `lm_head` ; **une séquence à la fois** (tampon d'état caché unique, docstring) | lot > 1 : état caché par séquence |
| garde de lot | `speculative.py:59-104` (`lot_max`, fenêtre, gain_min) : au-dessus de `lot_max`, jamais éligible | même garde | b=8 : `lot_max` ≥ 8 et garde par temps |
| état caché de la cible | — | `model.py:118-120`, `:201-207` (préfill), `:353-407` (tampon) ; `runner.py:1712`, `:1793` (`_mtp_hidden_n`) | rien |
| chargement | — | `loader.py:954-970`, `:974-1031` ; `mtp.py:66-75` | **noms Qwen3.5 + trois normes +1** (§ 1) |
| vérification (q_len = k+1) | commune : `attention.py:540-556` (noyau paginé, q_len uniforme) ; hybrides GDN : `couches.py:300-316` (`ensure_hist`, tuple corrigé après le 19/09) | commune | preuve de bout en bout sur hybride : **jamais faite** (19/09 NON MESURABLE, défaut corrigé depuis, jamais rejoué) |
| graphes | `graphs.py:634-639` (q_len uniforme exigé, `paged_ok`) | `graphs.py:1071-1075` (tampon d'état caché réservé avant capture) | rien ; brouillon hors graphe (eager) |

## 3. Coût en graphes CUDA pour q_len > 1

* **Clés** : une capture par (godet b, godet nblk, q_len). MTP3 donne q_len = 4, clés nouvelles à côté de q_len = 1.
  * Capture paresseuse ≈ 0,11 s par clé neuve (p91) ; `MAX_GRAPHS` = 64 (p85).
  * À b=1 : ≈ 2 × les clés actuelles, sans risque de plafond. À b=8 : idem, par godet.
* **Hybride GDN (Qwen3.8 : 48 couches linéaires sur 64)** : `ensure_hist(q_len)` (`couches.py:311-315`) alloue, par couche GDN, q_len copies de chaque état statique du godet.
  * État récurrent par séquence : 48 têtes V × 128 × 128 (`linear_*` de la config), soit 3,1 Mio en fp32 ; conv. négligeable.
  * **Hypothèse à vérifier** : dtype fp32 de l'état.
  * Soit q_len × b × 3,1 Mio × 48 : **b=1 ≈ 0,6 Gio, b=8 ≈ 4,8 Gio** pour q_len = 4.
  * C'est acceptable à b=1. À b=8, c'est le poste qui borne le lot sur 32 Go, avec des poids de ≈ 15 Go.
* **Le brouillon n'est pas capturé** (`MTPProposer.propose` : k passes eager d'une couche + `lm_head` [248 320 × 5 120] nvfp4, 0,64 Go lu par jeton brouillon).
  * Coût estimé par jeton brouillon : ≈ 0,5 ms de lecture (lm_head + bloc, ≈ 0,9 Go à ≈ 1,6 To/s) + ≈ 0,5-1 ms de lancements eager.
  * Pour k = 3 : **≈ 3-4,5 ms par pas**.

## 4. Gain prédit (écrit avant toute mesure) et ce qui le réfuterait

Modèle : jetons par pas = 1 + k·a (a = acceptation par jeton proposé), coût du pas = vérification (≈ 1,05-1,15 × le
pas simple à b=1, borne mémoire) + brouillon (§ 3).

* **b=1, MTP3, APRÈS le correctif du chargeur (§ 1)** :
  * a = 40-55 % (NInfer : 48,9 % avec les mêmes têtes), soit 2,2-2,65 jetons par pas.
  * Pas simple ≈ 12-14 ms (70-85 t/s, scellé de la 102) ; pas spéculatif ≈ 16-20 ms.
  * **Prédit : +35 à +75 % de t/s** (≈ 100-135 t/s), sous NInfer (143,8), faute de brouillon capturé.
* **b=8** :
  * **avec le code actuel : 0** (proposeur à séquence unique, `lot_max`).
  * Après un proposeur par lot (état caché par séquence), prédit **+10 à +30 %** sur 300-420 t/s. La vérification devient plus coûteuse (32 lignes), et la VRAM de `ensure_hist` (≈ 4,8 Gio) peut forcer un godet plus petit.
* **Réfutations** (b=1) :
  * a < 30 % : têtes mal lues (noms, normes, ordre) → pas de gain, correctif faux.
  * gain < +15 % à a ≥ 40 % : le coût du brouillon eager est sous-estimé → capturer le brouillon avant de conclure.
  * Sorties gloutonnes ≠ sans spéculation (sha256, protocole de `verdict-mtp-exact-19-09`) : **défaut de justesse**, arrêt.
  * Chute au premier pas spéculatif (hybride GDN) : le correctif du 19/09 ne suffit pas.
* **Premier contrôle, 15 min de carte** : charger `Qwen3.8-27B-nvfp4` avec la table de noms (§ 1) et mesurer a en forçage enseignant sur 5 invites, avant tout débit.
  * Si a < 30 %, rien d'autre ne se mesure.

## 5. Coût de mise en œuvre (sur ta décision)
* Noms Qwen3.5 et trois normes +1 au chargement, avec un test à sec qui casse sans le +1 : **1-2 h**.
* Contrôle de l'acceptation, puis b=1 greedy sha et t/s : **≈ 30 min de carte**.
* Proposeur par lot pour b=8 : **4-6 h**, puis ABBA.
* Capture du brouillon : **3-5 h**, à ne faire que si la réfutation « coût eager » tombe.
