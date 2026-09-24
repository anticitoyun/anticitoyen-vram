# Pièce 156 (c) — scellé (poste5, 24/09, écrit AVANT le code et la prise) : fusions AU BIT F4, F5, F2

Feu de chef (24/09 14 h) : fusions au bit seulement, dans l'ordre F4, F5, puis F2 si elle tient au bit. Chacune avec son
test d'équivalence AU BIT dans le même commit, un bras cassant, et cette prédiction écrite avant. Une seule ABBA
Qwen3.8 b=8, défaut futur = config Marlin qualifiée (`ACVRAM_PROJ_MARLIN=1 ACVRAM_PROJ_MARLIN_DOUBLES=
ACVRAM_GEMV_MARLIN_V2=1 ACVRAM_GEMV_MARLIN_TPB=1 ACVRAM_GEMV_MARLIN_S=0`). Dossier : `poste5-piece156b-dossier-24-09.md`.

## F4 — état GDN écrit en place (opt-in `ACVRAM_GDN_ETAT_EN_PLACE=1`)

* Aujourd'hui (`gdn.py`, `decode_static_batch`) : fla alloue `final_state` (`fused_recurrent.py:211`), puis
  `S.copy_(S_new)` recopie 8 × 48 × 128 × 128 × 4 o = 25 Mo par couche (48 memcpy D2D par pas, 421 µs au mixte).
* Changement : lancer le MÊME noyau `fused_recurrent_gated_delta_rule_fwd_kernel`, mêmes arguments, `h0 = ht = S`.
* **Pourquoi au bit** : `p_h0` (l. 108) et `p_ht` (l. 177) ont la même indexation ; chaque programme de la grille
  (NV, N·HV) charge sa tuile [K, BV] une fois avant la boucle (l. 110) et l'écrit une fois après (l. 179) ; aucun
  programme ne lit la tuile d'un autre. Même noyau compilé, seule l'adresse de sortie change.
* Repli sur le chemin actuel si S n'est pas contigu (créneaux non contigus : `S.contiguous()` serait une copie).
* Le décodage b=1 (`decode_static`, créneau unique, passe par `forward`) n'est pas touché.
* Test : `decode_static_batch` avec et sans, plusieurs pas, `torch.equal` sur sorties ET états. Bras cassant : `ht`
  omis (état jamais écrit) → ROUGE au 2e pas.
* **Prédit : −0,35 à −0,42 ms/pas** à b=8 (48 memcpy de 8,8 µs retirés ; le noyau fla écrit déjà son état).

## F5 — résidu différé étendu aux couches GDN (opt-in `ACVRAM_GDN_RES_DIFFERE=1`)

* Aujourd'hui `ACVRamModel._res_differe` (`model.py:422`) refuse tout hybride non MLA : Qwen3.8 fait 2 additions bf16
  séparées par couche (`couches.py:227, :348` ; 112 additions par pas, 102 µs).
* Changement : `_res_differe` admet une couche `DecoderLayerGDN` dont l'attention linéaire est un `GatedDeltaNet`, avec
  MLP sur le même appareil et deux normes `RMSNorm` ; `decode_fixed_res` (`couches.py:232`) existe déjà et est générique.
* **Pourquoi au bit** : `add_norm` = `rmsnorm_bf16` avec résidu, `bf16(fp32(res) + fp32(y))`, soit l'addition bf16 de
  torch, puis la même norme (contrat `couches.py:238-241`, déjà servi au bit par défaut sur les modèles denses, C15).
* Test : une pile GDN + attention jouée par `decode_fixed` puis par `decode_fixed_res`, `torch.equal` sur x final.
  Bras cassant : `add_norm` remplacé par une somme fp32 non arrondie avant la norme → ROUGE.
* **Prédit : −0,08 à −0,12 ms/pas** (96 additions de ~1 µs ; les normes restent en nombre égal).

## F2 — conv de décodage fusionnée (opt-in `ACVRAM_GDN_CONV_FUSEE=1`), seulement si au bit

* Aujourd'hui (`gdn.py:261-274`) : 4 casts fp32, cat, copie de l'état de conv, conv depthwise, silu, 2
  `repeat_interleave`, contiguous : ~730 µs/pas au mixte.
* Changement : un noyau Triton lit qkv bf16, décale l'état de conv EN PLACE, calcule la conv à 4 prises et silu, et
  écrit q, k, v fp32 SANS répétition des têtes (fla indexe `i_h = i_hv // (HV // H)`, `fused_recurrent.py:68`, la
  disposition exacte de `repeat_interleave`).
* **Au bit visé, non garanti** : torch `conv_depthwise2d_forward_kernel_generic` accumule en fp32 depuis 0, prise par
  prise dans l'ordre → `tl.fma` explicite dans le même ordre ; silu de torch = `x / (1 + expf(-x))` → libdevice `exp`
  et `div_rn` (l'extension CUDA est compilée en `--use_fast_math`, `kernels/__init__.py:381`, donc elle ne peut pas le
  faire au bit : Triton seulement). Si le test au bit échoue, F2 n'est PAS livrée (ordre de chef) et je le dis.
* **Prédit : −0,50 à −0,65 ms/pas.**

## Prise unique (après les trois, ou F4+F5 si F2 tombe)

* `frontiere-pas.py`, Qwen3.8-27B-nvfp4, b=8, 300 pas, en processus, config Marlin qualifiée ; ordre A B B A
  (A : trois drapeaux à 0 ; B : à 1) ; -lgc 2700, cpu-safe. Rejeu au bit A contre B sur 64 jetons gloutons à b=8 (même
  invite ×8 et 8 invites distinctes) : les jetons doivent être identiques, sinon ce n'est pas au bit et rien n'est livré.
* **Prédit : pas b=8 −0,95 à −1,2 ms** (F4 + F5 + F2) ; sans F2 : −0,43 à −0,54. Seuil TENU : ≥ 70 % de la borne basse
  prédite ET jetons identiques. En deçà : le gain de noyau ne passe pas dans le pas (trou de graphe, recouvrement), je le
  dirai avec la trace.
