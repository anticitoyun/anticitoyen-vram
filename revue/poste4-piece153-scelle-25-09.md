# Scellé — pièce 153 (attention + GDN int8 par canal, MLP nvfp4), écrit à sec avant la conversion réelle

Reprend et resserre `scratchpad/poste4-piece153-attn-gdn-int8-mlp-nvfp4-24-09/prediction-153.md`
(chiffres inchangés) après le blocage 1 (drapeau GDN absent, résolu) et l'échec de la première
conversion (0 octet int8, snr_floor=0.0 par défaut ne promeut rien). Protocole cette fois :
`--promotion-classes q_proj,k_proj,v_proj,o_proj,linear_attn.qkv,linear_attn.gate,linear_attn.alpha,
linear_attn.beta,linear_attn.out --max-promotions 1.0 --snr-floor 99 --attn-qkvo-int8-canal
--gdn-int8-canal --no-awq`, comme suggéré par les deux avertissements « SANS EFFET » (attn et GDN).

## Témoins T1/T2 et seuil KL

Aucune mesure KL n'existe pour Qwen3.8-27B lui-même (recherché dans ETAT.md, absent). Faute de
témoin propre au modèle, T1/T2 sont transférés du seul couple KL/format mixte déjà mesuré dans le
projet sur le même principe (attention en int8, reste en nvfp4/int4), Coder KL max par pas contre
bf16 (`ETAT.md` 22/09 12h2x, poste2 b2ab639a/84af7a90) :
* **T1 = 0,519** (Coder, projections d'attention int8)
* **T2 = 0,735** (Coder, projections d'attention nvfp4)
* **seuil = 2 × max(T1, T2) = 1,47**

Réserve nommée : ce sont des témoins d'un AUTRE modèle (Coder, pas Qwen3.8-27B) et d'un régime pas
identique (Coder n'a pas de GDN) — un proxy, pas une mesure directe. Si la 153 KL max dépasse 1,47,
c'est un signal fort (loin au-dessus du pire témoin transféré) ; si elle est en dessous, ça ne prouve
pas l'absence de problème sur ce modèle précis, seulement l'absence d'un problème de l'ampleur déjà vu
ailleurs sur un mécanisme comparable.

## PPL (inchangé depuis prediction-153.md)

D'après arXiv:2609.04098 Table 1 (Qwen3.8-27B, PPL@4K/@32K) : régime « Minima » (tout quantifié, proche
de notre 102 nvfp4 pur) +10,4 % contre HF ; régimes attention+GDN protégés (Unsloth/RadixArk) +3,0 % à
+5,8 %. **Prédiction : écart acvram-153 vs HF dans [+3 %, +6 %]**, contre l'écart actuel de la 102
(+11,0 % evaluate.py / +13,87 % NInfer, protocoles différents). Ce que ça rendrait si faux : ≥+8 %
signale une perte du chemin acvram au-delà de la quantification ; <+2 % (mieux que tous les points
calibrés du papier, alors que ceci est du RTN sans calibration, `--no-awq`) serait surprenant et
mériterait une relecture avant d'être cru.

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
* **b=8 (plus proche du calcul, GEMM dense)** : chaque poids lu sert 8 lignes — impact relatif du
  surcoût devrait être PLUS FAIBLE qu'à b=1. **Aucun encadrement chiffré tenté, seulement la direction**
  (ralentissement b=8 < ralentissement b=1, en proportion).

## Issues nommées

1. **KL/PPL au-dessus du seuil transféré (1,47 / +8 %)** — chemin acvram fautif au-delà de la
   quantification (candidat de recherche, pas la recette elle-même, cf. prediction-153.md).
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
