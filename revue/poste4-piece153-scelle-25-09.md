# Scellé — pièce 153 (attention + GDN int8 par canal, MLP nvfp4), écrit à sec avant la conversion réelle

Reprend et resserre `scratchpad/poste4-piece153-attn-gdn-int8-mlp-nvfp4-24-09/prediction-153.md`
(chiffres inchangés) après le blocage 1 (drapeau GDN absent, résolu) et l'échec de la première
conversion (0 octet int8, snr_floor=0.0 par défaut ne promeut rien). Protocole cette fois :
`--promotion-classes q_proj,k_proj,v_proj,o_proj,linear_attn.qkv,linear_attn.gate,linear_attn.alpha,
linear_attn.beta,linear_attn.out --max-promotions 1.0 --snr-floor 99 --attn-qkvo-int8-canal
--gdn-int8-canal --no-awq`, comme suggéré par les deux avertissements « SANS EFFET » (attn et GDN).

## Témoins T1/T2 et seuil KL — correction chef 25/09 : mesurés, pas empruntés

Erreur de la première version de ce scellé : des KL de Coder ne sont pas des témoins de Qwen3.8-27B.
**Règle fixée maintenant, chiffre calculé à la prise** : T1 = KL du bras B (153) contre HF sur la
séquence de calibration dans son ordre normal ; T2 = même paire, ordre inversé (même corpus, même
longueur, teacher forcing) — les deux mesurés dans LA MÊME PRISE que le bras B, sur Qwen3.8-27B nvfp4
(102) lui-même comme référence de « bruit d'ordre » attendu à ce KL. **Seuil = 2 × max(T1, T2)**,
appliqué au KL max par pas de 153 contre HF. Les valeurs Coder (T1=0,519 int8 / T2=0,735 nvfp4,
`ETAT.md` 22/09 12h2x, poste2 b2ab639a/84af7a90) restent un ordre de grandeur attendu SEULEMENT — si T1/
T2 mesurés sur Qwen3.8-27B sortent d'un facteur 3 de cette fourchette, le protocole de mesure (pas la
conversion) est le premier suspect avant de juger 153.

## PPL — référence explicite : contre HF bf16 (pas contre la 102)

D'après arXiv:2609.04098 Table 1 (Qwen3.8-27B, PPL@4K/@32K) : régime « Minima » (tout quantifié, proche
de notre 102 nvfp4 pur) +10,4 % contre HF ; régimes attention+GDN protégés (Unsloth/RadixArk) +3,0 % à
+5,8 %. **Prédiction : écart acvram-153 vs HF bf16 dans [+3 %, +6 %]** — c'est la référence HF du papier,
PAS la 102, qui sert de base de comparaison à ce chiffre. Ce que ça rendrait si faux : ≥+8 % vs HF
signale une perte du chemin acvram au-delà de la quantification ; <+2 % vs HF (mieux que tous les
points calibrés du papier, alors que ceci est du RTN sans calibration, `--no-awq`) serait surprenant et
mériterait une relecture avant d'être cru.

**PPL appariée contre la 102** (l'autre comparaison demandée par chef dans l'ordre initial) — dérivée,
pas mesurée séparément : la 102 est actuellement à +11,0 % vs HF (evaluate.py) ou +13,87 % (NInfer,
protocole différent, cf. 143). Si 153 atteint [+3 %,+6 %] vs HF, alors **153 vs 102 devrait être une
amélioration d'environ −4,5 % à −7,2 % de PPL relative** ((1,03 à 1,06) / 1,11 ≈ 0,928 à 0,955). Rendrait
faux : 153 PPL ≥ 102 PPL (aucune amélioration malgré le surcoût en octets) — contredirait toute la
prémisse de la pièce.

## Taille disque

Familles concernées (attn+GDN+MLP) : 13,41 Gio en 102 (tout nvfp4) → 16,52 Gio en 153 (attn+GDN int8-
canal, +3,11 Gio, +23,2 % sur ces familles). Le reste du modèle (tête, embedding, normes — bf16/nvfp4
selon promotion_classes non touchées) apportait 16,9 − 13,41 ≈ **3,49 Gio** dans la conversion ratée
d'hier (mesure directe, cette partie-là n'a pas changé de protocole). **Total prédit : 16,52 + 3,49 ≈
20,0 Gio** (confirme le contrôle de chef, « ~20 Gio »). Si le total mesuré s'écarte de plus de 5 % de
20,0 Gio, erreur de shape ou de classe de promotion, pas surprise physique.

## Vitesse ABBA b=1/b=8 contre la 102

Encadrement par octets lus, pas une mesure (comme la pièce 144) :
* **b=1 (décodage, GEMV borné bande passante)** : lire attn+GDN en int8 coûte ×1,819 plus d'octets que
  ces seules familles ; sur l'ensemble des poids lus par pas (MLP inclus, inchangé), plancher +23,2 %
  de trafic. **Ralentissement b=1 attendu entre +23 % et +82 %** (plafond théorique si seules attn+GDN
  comptaient).
* **b=8 (plus proche du calcul, GEMM dense)** — correction chef 25/09, chiffré : les poids d'une
  couche sont lus UNE FOIS par pas quel que soit b (un GEMM batché réutilise le même poids pour les 8
  lignes) — les +3,11 Gio supplémentaires sont donc le MÊME surcoût absolu de lecture par pas à b=1 et
  b=8, pas un surcoût qui grossit avec b. Ce qui change : à b=8 le pas total est dominé par le calcul
  (`gemm_dense_etroit_nvfp4` à 80,5 % du pas, `poste3-piece126-nsys-qwen38-24-09.md`), donc le même
  surcoût absolu de lecture pèse une part BIEN PLUS PETITE d'un pas bien plus long. **Encadrement :
  ralentissement b=8 entre +2 % et +15 %** — plancher >0 (strictement plus d'octets lus), plafond fixé
  en supposant que le surcoût de lecture ne mord que sur la portion non couverte par le GEMM dense
  MLP dominant (≤19,5 % du pas d'après la piece126) même s'il double cette portion. Rendrait faux :
  <+2 % signalerait un recouvrement quasi parfait calcul/lecture (à vérifier, pas exclu) ; >+15 %
  contredirait la dominance du GEMM MLP mesurée en 126 et signalerait un goulot ailleurs (glue,
  synchronisation) non prévu par ce raisonnement.

## Issues nommées

1. **KL au-dessus du seuil calculé (2×max(T1,T2), mesuré à la prise) ou PPL ≥+8 % vs HF** — chemin
   acvram fautif au-delà de la quantification (candidat de recherche, pas la recette elle-même, cf.
   prediction-153.md).
2. **int8 GDN n'apporte rien** — GDN (5,56 G paramètres/48 couches sur 27B) est peut-être déjà assez
   bien servi par nvfp4 seul (contrairement à l'attention pleine, plus concentrée sur 16 couches) : si
   la PPL avec attn seule en int8 (pas encore mesurée, hors scope 153) atteint déjà la même fourchette
   [+3 %,+6 %] que attn+GDN combinés, le coût disque/bande passante du GDN en int8 (une bonne partie
   des +23,2 %, GDN pèse plus que l'attention en paramètres) n'achète rien. Rendrait faux : une PPL
   153 dans la fourchette prédite ET un gain marginal <1 point de PPL par rapport à un bras « attn
   seule » hypothétique — non mesuré ici, à isoler dans une pièce séparée si le doute se confirme.
3. **Taille hors de 20,0 Gio ± 5 %** — shape/classe de promotion incorrecte, à corriger avant tout
   jugement qualité (la conversion serait à refaire, pas juste requalifiée).
4. **Promotion incomplète malgré `--snr-floor 99`** — un tenseur peut rester hors PROMOTE (biais,
   normes, SENSITIVE_SUFFIXES) même avec un plancher agressif ; le bilan `_bilan_attn_int8`/
   `_bilan_gdn_int8` (comptage exact promu/candidat) sert justement de garde avant la mesure GPU —
   à lire dans le manifeste AVANT de lancer PPL/ABBA, pas après.
