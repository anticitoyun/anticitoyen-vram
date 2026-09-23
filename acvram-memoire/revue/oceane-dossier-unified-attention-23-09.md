# Dossier : porter `kernel_unified_attention` de vLLM 0.29 (23/09, Océane)

Dossier préparé à sec : 0 min de carte, rien de codé. Il ne sert que si la cellule (b) de Manon laisse un écart d'attention au-delà de 2 σ.

* **source** : `/tmp/vllm-0.29`, tag `v0.29.0` (98dff2a), Apache-2.0. L'Apache-2.0 est compatible avec notre GPL-3.0 ; il faut garder l'en-tête SPDX et le copyright « contributors to the vLLM project », et signaler les modifications.
* **notre témoin** : `acvram/kernels/attn_paginee.py` à 2217a51c. Le défaut actuel est `_partiel_reduit_kernel`, avec la réduction déroulée à 8 warps.
* **chiffres repris** : p91 (traces servies bb6ec0c3 et 730b7075, -lgc 2700) et p92 (`banc-attn-decoupe.py`).

## 1. Fichiers et lignes (v0.29.0)
| quoi | où |
|---|---|
| noyau | `vllm/v1/attention/ops/triton_unified_attention.py:178-683` ; programme (q_block, tête KV, segment) :305-307 ; segment tiré de `seq_len` :323 (`tiles_per_segment = cdiv(seq_len, NUM_SEG·TILE)`) ; boucle de tuiles :420 ; K/V NHD par pointeurs :459-480 ; échelles par (jeton, tête) :495-515 ; score :540 ; softmax en ligne :561 (`softmax_step`, helpers :418, `tl.exp` comme chez nous) ; P·V :581-582 ; partiels :630 |
| réduction | `reduce_segments` :684-772, lancée à part (:1170, grille (jetons, têtes Q)) |
| enveloppe | `unified_attention` :802 ; `_get_tile_size` :784-800 ; BLOCK_M = 16 si n_rep ≤ 16 (:932) ; bascule 2D/3D :1041-1050 ; grille :1076-1082 |
| backend | `vllm/v1/attention/backends/triton_attn.py` : 16 segments (:54) ; `seq_threshold_3D` = 128 // HKV, ramené à la taille de capture la plus proche (:139-151) ; tampons de segments statiques (:155-173) ; blocs multiples de 16 (:307) ; disposition du cache (:382-396, :606-614) |
| helpers | `triton_attention_helpers.py` (440 l.) : on n'a besoin que de `cdiv_fn` :22, `resolve_seq_and_query_len` :45, `init_softmax_M` :110, `compute_tile_loop_bounds` :142, `store_segm_reduce_scalars` :242 et `softmax_step` :418 |

## 2. Leur régime au banc tracé, comparé au nôtre (Coder, HQ 32, HKV 4, D 128, n_rep 8)
| | vLLM servi (730b7075) | acvram (défaut) |
|---|---|---|
| cache | `fp8_e4m3` **par tenseur** (mode 1), NHD `[NB, 16, HKV, D]` après transposition | int8 + échelle fp16 **par (jeton, tête)**, `kc/vc [NB, 16, HKV, D]`, `ks/vs [NB, 16, HKV]` |
| octets par jeton et par tête | 128 | 128 + 2, soit le même flux |
| q | bf16 (`query_quantization=False`) | bf16 |
| tuile | **16 jetons** (`_get_tile_size`, q sur 2 octets), 4 warps par défaut Triton | 64 jetons (`PAGES_PAR_TUILE` = 4), 8 warps |
| découpage | **16 segments fixes**, longueur tirée de `seq_len` de la séquence | C fixé par `_tranches` à partir de N = godet (:246-256) : C = 8 à b=12 et godet 64 |
| b=12, ctx 768 | 12 × 4 × 16 = 768 programmes actifs, 48 jetons chacun (3 tuiles), plus 6 × 4 × 16 programmes qui sortent tout de suite (BLOCK_Q = 2) | 12 × 4 × 8 = 384 programmes, dont 6/8 travaillent sur 128 jetons chacun (2 tuiles) |
| réduction | deuxième lancement | fusionnée : le dernier arrivé réduit, sans memset, et rejoue sous graphe |
| capture | seuil 3D ramené à la taille capturée 24 ; tampons de segments statiques | godets `nblk` (`graphs.py:642`, `kvcache.py:39`) ; compteurs réservés hors capture (:279) |

Leur **mode 2** (INT8_PER_TOKEN_HEAD, :495-515, :540, :581) fait exactement notre arithmétique : un score × (scale · sk) et P × sv avant P·V. Notre cache se lit **tel quel** : NHD, blocs de 16, `stride_3 = 1`, et `ks/vs` passent en `k_scale_cache` avec les pas (16·HKV, HKV, 1). Aucune conversion de disposition n'est nécessaire.

## 3. Adaptation à faire, en ne gardant que le chemin décodage 3D
1. **P·V en fp16, pas dans le dtype de V.** À :581, leur `.to(V.dtype)` passe P en bf16. Chez nous, bf16 mettait 264 sorties sur 384 hors de 2⁻⁸ (Laure, `attn_paginee.py:81-83`). On garde notre cast fp16.
2. **Réduction** : on garde notre fusion « dernier arrivé » (:164-243). On n'importe ni `reduce_segments` ni son deuxième lancement, qui coûterait environ 2 µs par couche.
3. **Segments fixes C = 16**, chunk calculé dans le noyau à partir de `slen` (`cdiv(slen, 16·16)·16`), C constexpr. Conséquence : ni le graphe ni la sortie ne dépendent plus du godet du lot, et la sortie devient invariante au lot (défaut signalé à la p91). Au-delà de b = 24, on passe à C = 1, comme leur chemin 2D.
4. On retire tout le reste : alibi, softcap, sinks, mm_prefix, r_swa, chunk-lookback, TD Intel, fp8 Q, sortie fp8. La fenêtre glissante reste (Gemma), D ∈ {64, 128}, et la vérification spéculative (q_len > 1) reste sur le noyau CUDA.
5. BLOCK_M = 16 sur une seule requête. Leur BLOCK_Q = 2 lance des programmes vides à q_len = 1 ; on les retire, et la grille devient (B, HKV, 16).

Taille prévue : environ 180 lignes de Triton, derrière un drapeau `ACVRAM_ATTN_UNIFIE` et avec le témoin actuel nommé.

## 4. Coût
| étape | heures | carte |
|---|---|---|
| banc de réfutation (§ 7) | 1 | 5 min |
| port et drapeau | 4-5 | 0 |
| tests : fp64, bras qui cassent, fantômes, lot mêlé, fenêtre, invariance au lot | 1,5 | 2 min |
| capture sur 7 alias, ulp, KL sur 5 invites b=1/b=12, ABBA servi (Manon) | 1,5 | 25 min en 2 prises |
| **total** | **8-9 h, soit plus d'un jour : note au chef requise** | ≈ 30 min |

## 5. Gain prédit (écrit avant toute mesure)
* **b=12, ctx 768.** p91 : 14,88 contre 12,66 µs par couche ; après la réduction déroulée de la p92, environ 14,60. L'écart plafond est de 1,94 × 48 = **0,093 ms/pas**.
  * Prédit : **−0,055 à −0,093 ms/pas**, soit +0,9 à +1,6 % de débit.
  * Sur la moyenne du lot (15,00 → 14,72 contre 11,99) : −0,08 à −0,13 ms/pas, soit **+1,3 à +2,2 %** (1 916 → 1 940-1 958 t/s).
  * Cela couvre au plus la moitié des 4,9 % qui nous séparent de vLLM.
* **b=1, ctx 768.** Pas de trace vLLM à b=1, donc la prédiction est large : **0 à −0,07 ms/pas**, soit 0 à +2 % sur 3,21 ms.
  * Leur grille ne compte que 64 programmes pour 170 SM, contre 12 tranches actives de 64 jetons chez nous : une **régression est possible**.
  * Parade : garder l'ancien chemin sous un seuil de lot, si le banc le montre.
* **L'issue qui me gênerait** : l'avance de vLLM viendrait du format (fp8 par tenseur, sans chargement d'échelle par jeton) et non de la forme du noyau. Dans ce cas, le port ne gagne **rien**. Le bras (b) du § 7 la mesure.

## 6. Risque pour la justesse
* La sortie **n'est pas au bit** : la tuile de 16 et C = 16 changent l'ordre de rééchelonnement du softmax et de la fusion des tranches. C'est la même classe que la p82 ter.
* Critère, fixé avant la mesure :
  * test fp64 existant ≤ 2⁻⁸ ;
  * distance à fp64 ≤ 1,00 × celle du témoin ;
  * ≤ 1 ulp bf16 contre le témoin sur ≥ 99,9 % des sorties ;
  * KL b=1 identique au témoin à ±0,005 ;
  * KL b=12 ≤ témoin + 0,025 ;
  * fin de préfill identique ;
  * capture 28/28 sur 7 alias.
* Risques nommés :
  * le cast fp16 oublié (le bras qui doit casser l'attrape : 264/384) ;
  * les compteurs et C constexpr sous graphe ;
  * une séquence vide ou fantôme sur 16 segments ;
  * les bords de fenêtre, quand une tuile de 16 est entièrement masquée (`softmax_step` :431 met −inf à 0 comme nous).
* Gain de justesse : la sortie devient invariante au lot.

## 7. Ce qui le réfuterait au banc (≤ 5 min de carte, sous `carte.sh`, L2 froid, 48 caches, graphe, b=12, ctx 768 et b=1, ctx 768)
Trois bras. Chaque bras vendorisé imprime le chemin de son module.
* (a) `unified_attention` **tel quel**, venv vLLM (Triton 3.7.1), fp8 par tenseur. C'est le contrôle de l'instrument : il doit retrouver 12,66 ± 10 % à b=12.
* (b) le même noyau en **mode 2** sur **notre** cache int8 et nos échelles fp16. C'est le plafond du port, avant le fp16 de P·V.
* (c) notre défaut, dans notre venv (Triton 3.8.0 ; le compilateur est un confondu que (a) contre (b) ne lève pas, et qui est nommé comme tel).

Lecture :
| issue | condition | conclusion |
|---|---|---|
| **GO** | (b) ≤ (c) − 1,5 µs/couche à b=12 | ≥ 0,07 ms/pas à gagner |
| **RÉFUTÉ** | (b) ≥ (c) − 0,5 µs | le noyau n'est pas le levier |
| **RÉFUTÉ, format** | (b) − (a) ≥ 1,5 µs | le levier est le format (fp8 par tenseur), pas le noyau : autre pièce, qui ne tient pas au bit |
| **instrument faux** | (a) hors de 12,66 ± 10 % | on ne conclut rien |
| entre GO et RÉFUTÉ | | décision au chef |
| **garde-fou b=1** | (b) > (c) à b=1 | l'ancien chemin est gardé sous un seuil |

La durée prévue est une heure de script, une prise de 5 minutes et ce dossier. Aucune partie n'est codée tant que le chef n'a pas tranché.
