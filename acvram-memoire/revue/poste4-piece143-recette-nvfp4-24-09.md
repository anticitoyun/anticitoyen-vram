# Pièce 143 — la recette explique-t-elle nos +11,0 % (à sec, recherche + code, aucune carte)

## (1) Notre recette, fichier:ligne

* **Format des poids** : `acvram/quant/nvfp4.py:259-322` (`quantize_nvfp4`) — bloc de 16 éléments
  le long de K, E2M1 4 bits, échelle de bloc E4M3, échelle globale FP32.
* **Échelle globale** : `nvfp4.py:283-291` — `amax = wb.abs().amax()` (l'amax du POIDS LUI-MÊME,
  aucune activation) ; `g = amax/(E2M1_MAX*E4M3_MAX)`. Aucune donnée d'activation n'entre dans ce
  calcul.
* **Échelle de bloc** : `nvfp4.py:292-314` — deux schémas, `max6` (classique, amax/6, **défaut du
  module** `_echelle_defaut`) et `4sur6` (candidat amax/4 retenu si son MSE de bloc est meilleur,
  `--echelle 4sur6`, **opt-in**, cite `arXiv:2512.02010` en commentaire — *le même papier que la
  recherche externe ci-dessous, déjà dans notre code, jamais activé pour ce modèle*).
* **Arrondi** : `nvfp4.py:62-79` — au pair le plus proche sur la grille E2M1, standard.
* **Calibrage/AWQ** : `acvram/quant/calibrate.py:1-27` décrit AWQ (mise à l'échelle par canal
  guidée par les activations) et Hadamard comme disponibles pour NVFP4 **et** INT4, mais
  **`wants_hadamard` (`convert.py:553-565`) : en mode `auto`, `return fmt == "int4_awq" and ...`
  — le Hadamard n'est JAMAIS appliqué au format nvfp4**, seulement à int4_awq. Et le manifeste
  RÉEL de notre conversion (`/mnt/AI_GENERATOR/models_acvram/Qwen3.8-27B-nvfp4/acvram_manifest.json`,
  section `options`) : **`"calibrate": false, "calib_tokens": 0, "calib_seqs": 0, "awq": false`**
  — **arrondi au plus proche PUR, sans aucune calibration, sans AWQ, sans Hadamard**, pour ce
  modèle précisément (pas une affirmation générale sur tout le parc).
* **Couches gardées en clair** : `SENSITIVE_SUFFIXES` (`convert.py:321-335`) — layernorms,
  `embed_tokens.weight`, `k_b_proj`/`v_b_proj` (MLA), routeur MoE, et pour les hybrides GDN :
  `conv1d.weight`, `a_log.weight`, `dt_bias.weight` (les PARAMÈTRES SCALAIRES des portes, pas
  leurs projections). **Les projections `in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b` (portes
  GDN) et `self_attn.q/k/v/o_proj` NE sont PAS exemptées** : quantifiées en nvfp4 comme le MLP,
  par défaut.

## (2) Recettes externes, chiffres publiés

* **NVIDIA ModelOpt** (`NVIDIA/Model-Optimizer`, `examples/hf_ptq/README.md`) : `mtq.quantize(model,
  mtq.NVFP4_DEFAULT_CFG, forward_loop)` — **la calibration est OBLIGATOIRE** dans l'API elle-même
  (un `forward_loop` sur un `calib_set` réel, 128-512 échantillons typiques, mélange
  `cnn_dailymail`+`Nemotron-Post-Training-Dataset-v2` par défaut). Pour la MEILLEURE précision
  NVFP4, ModelOpt recommande explicitement `NVFP4_MLP_ONLY_CFG`/`NVFP4_OMLP_ONLY_CFG` : **garder
  l'attention QKV en précision plus haute**, ne quantifier que MLP (+ `o_proj` en option).
* **llm-compressor** (vLLM/neuralmagic) : recette par défaut `QuantizationModifier(targets="Linear",
  scheme="NVFP4", ignore=["lm_head"])`, échelles d'activation calibrées par `static_minmax` sur un
  échantillon (20 dans les exemples) — là aussi, calibration par défaut, jamais du RTN pur.
* **Kozyrev, S. & Maiboroda, D., « Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for
  the Recurrent Half of a Hybrid 27B LLM », `arXiv:2609.04098v1 [cs.AI]`, 3 sept. 2026,
  https://arxiv.org/abs/2609.04098 (PDF relu en entier : https://arxiv.org/pdf/2609.04098, `pdftotext
  -layout`, sha256 du PDF téléchargé non conservé — texte relu directement, reproduit ci-dessous) —
  **directement sur Qwen3.8-27B, 48 couches GDN + 16 attention, le MÊME modèle que le nôtre**. Table
  1, ligne recopiée telle quelle (p. 3, « PPL @4K / @32K ↓ ») :

  > `PPL @4K / @32K ↓    6.95 / 10.35    7.67 / 10.84    7.16 / 9.91    7.35 / 9.95`
  > (colonnes : BF16, Minima, Unsloth, RadixArk — légende Table 1 : « Four models, one regime (FP8
  > KV, vLLM 0.27.1, TP=1, one RTX PRO 6000)… PPL@32K is measured inside a 32K request. »)

  soit, en écart relatif au BF16 :

  | modèle | PPL@4K | écart vs BF16 |
  |---|---|---|
  | BF16 | 6,95 | — |
  | **Minima** (NVFP4 W4A4, TOUT quantifié y compris les portes GDN, **calibré**) | 7,67 | **+10,4 %** |
  | Unsloth (communauté, protège le bloc GDN en 8/16 bits) | 7,16 | +3,0 % |
  | RadixArk (idem, protège GDN) | 7,35 | +5,8 % |

  Point clé : **même calibré avec soin (précision de tâche = BF16 à bruit de graine près sur 6
  suites), quantifier TOUT (y compris les portes GDN) coûte +10,4 % de PPL — presque notre
  +11,0 %.** Les recettes communautaires qui PROTÈGENT le bloc GDN (comme Unsloth, dont NInfer
  dérive son A16Only d'après `scelle-102bis-source.md`) tombent à +3,0/+5,8 % — c'est LÀ que
  vivent les « ~1-3 % » publiés que vous citiez, pas dans les recettes « tout nvfp4 ». Le papier
  montre aussi que la PPL est un indicateur PLUS sensible que la précision de tâche à ce choix
  (§4, "Perplexity is the honest residual").
* **Écart non expliqué à noter** : notre NInfer mesuré (pièce 138 quater) est à **+13,87 %**, pas
  aux +3,0 % d'Unsloth que le papier rapporte — alors que NInfer dérive justement d'un checkpoint
  Unsloth (`scelle-102bis-source.md`). Corpus différent (le nôtre : Austen/essai, `wiki.test.raw`
  eux), protocole de fenêtre différent (`min_context=0` chez nous, inconnu chez eux), référence
  HF potentiellement différente — écart à ne pas fusionner avec la question de calibration tant
  qu'il n'est pas expliqué séparément.

## (3) Écarts nommés, gain prédit, mesure possible SANS carte

Trois leviers, chacun mesurable en Python pur sur processeur avec `acvram/quant/nvfp4.py` tel
quel (le module est explicitement conçu pour tourner sur CPU, `nvfp4.py:22-24`) sur les VRAIS
poids bf16 déjà sur disque (`/mnt/4TO_SATACMR_2022/Modeles/models/Qwen3.8-27B-bf16`) :

1. **Échelle 4sur6 au lieu de max6** — gain prédit : le papier source (arXiv:2512.02010) rapporte
   un resserrement de l'écart de PPL de 1,82→0,20 et 2,49→0,34 (Llama3.1-8B/Qwen3-8B, unités PPL
   absolues, pas nos %) avec une technique de même famille (choix de bloc adaptatif) — un ordre de
   grandeur, pas un chiffre transposable tel quel à notre régime. Mesure sans carte : relancer
   `quantize_nvfp4(w, echelle="4sur6")` contre `echelle="max6"` sur un échantillon de tenseurs
   MLP/attention/GDN réels, comparer le SNR/MSE de reconstruction (`dequantize_nvfp4` contre le
   poids bf16 d'origine) — quelques minutes, aucun GPU.
2. **Attention/GDN en clair (comme Unsloth/RadixArk), MLP seul en nvfp4** — gain prédit : d'après
   arXiv:2609.04098 Table 1, l'écart de PPL tombe de +10,4 % (tout quantifié) à +3,0/+5,8 %
   (attention+GDN protégés) sur EXACTEMENT ce modèle — la transposition la plus directe des trois
   leviers. Mesure sans carte : même comparaison SNR/MSE, mais en excluant `self_attn.*` et
   `linear_attn.in_proj_*` du calcul d'erreur agrégée pour estimer la part de l'écart qui leur est
   imputable (déjà visible dans Table 3 du papier : `qkv`/`out`/`z` portent l'essentiel de
   l'erreur de sortie, les portes `a`/`b` presque rien — donc protéger `qkv`+`out` seuls pourrait
   suffire, moins cher que tout le bloc).
3. **Calibration AWQ (activations réelles, pas juste l'amax du poids)** — gain prédit : le plus
   incertain des trois sans mesure (ModelOpt/llm-compressor ne publient pas de comparaison RTN-vs-
   calibré isolée pour NVFP4 dans ce qui a été trouvé) — mais c'est la SEULE différence entre notre
   recette et TOUTES les recettes externes examinées (aucune n'utilise le RTN pur par défaut).
   Mesure sans carte, approximative : `calibrate.py` implémente déjà AWQ ; un proxy sans
   activations réelles (norme de ligne du poids comme substitut grossier de la saillance de canal)
   donnerait un premier ordre de grandeur, mais un vrai test demande des activations captées — qui
   nécessite un passage avant (carte), donc seulement les deux leviers 1-2 sont mesurables
   intégralement à sec ; celui-ci a besoin d'une prise, même limitée, pour être mesuré pour de
   vrai plutôt qu'estimé.

## Verdict

Notre +11,0 % n'est **pas manifestement une erreur de recette** au sens où même une recette
calibrée avec soin sur ce modèle précis (Minima, arXiv:2609.04098) montre une PPL similaire
(+10,4 %) quand elle quantifie TOUT comme nous. **Le principal levier identifié pour se rapprocher
des « ~1-3 % » publiés est de protéger l'attention et les portes GDN (comme fait le reste de la
communauté, Unsloth/RadixArk +3,0/+5,8 %)**, pas nécessairement d'ajouter une calibration —
même si l'absence totale de calibration (awq=false) reste la seule différence structurelle entre
notre recette et toutes les recettes externes trouvées, et mérite d'être testée séparément.
L'écart NInfer (+13,9 % mesuré contre +3,0 % publié pour Unsloth) reste ouvert, sans lien évident
avec la question de calibration posée ici.

Aucune carte ni code touché (recherche + lecture seules, ordre du chef).
