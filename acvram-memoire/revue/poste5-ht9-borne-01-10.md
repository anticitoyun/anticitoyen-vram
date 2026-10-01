# ht9 — préfill MoE servi (Marlin W4A16, disposition unique) : borne et prédiction ÉCRITES AVANT toute trace (poste5, 01/10)

Ordre de chef (bd ht9, remplace 0si obsolète). Base : a77970f45 (origin/main), branche poste5-ht9.

## Modèle de référence

Kimi-Linear-35B-kda-nvfp4. C'est le modèle des menus kimi de l'utilisatrice, et l'instrument p81 profile déjà son
préfill de 8 k (`outils/gpu/mesure/nsys-kda-p81.sh`, plage `p81:prefill8k`, une séquence de 8 192 jetons).
Configuration (config.json et acvram_manifest.json) : 27 couches, dont la couche 0 dense (nvfp4) et **26 couches MoE** ;
180 experts nvfp4 (4,5 bits par poids), 8 actifs par jeton ; hidden 2 304, moe_intermediate 1 024 ; un expert partagé
en int8 par couche.

## Borne (MoE routé, préfill de T = 8 192 jetons)

* Paires jeton-expert : T·k = 65 536, soit en moyenne 364 jetons par expert. Les 180 experts sont tous actifs.
* **Octets** : 3 × 1 024 × 2 304 × 4,5/8 = 3,98 Mo par expert, 717 Mo par couche, **18,6 Go pour 26 couches** si chaque
  poids est lu une seule fois. Il faut y ajouter x et la sortie (≈ 2 Go). Avec les intermédiaires [65 536, 1 024] bf16
  matérialisés (porte, haut, activation, bas : ≈ 1,7 Go par couche), on compte jusqu'à ≈ 45 Go.
  Au plancher de 1,52 To/s : **13,6 ms minimum, ≈ 30 ms avec les intermédiaires**.
* **Calcul** : 2 × 65 536 × 3 × 1 024 × 2 304 = 0,93 TFLOP par couche, **24,1 TFLOP pour 26 couches**. Marlin W4A16
  calcule en bf16 sur les cœurs tensoriels. Crête bf16 dense mesurée dans ce dépôt : 207 TFLOPS (cutlass sm80), 229 au
  mieux à 8192³. Donc **105 à 116 ms minimum**.
* Intensité : 2 × 364 / 0,5625 ≈ 1 300 FLOP par octet de poids, contre une crête de 207/1,52 ≈ 136. **Le préfill MoE
  est borné par le CALCUL, d'un facteur 8 environ.** La borne pertinente est donc ≈ 110 ms par préfill de 8 k
  (4,2 ms par couche).

## Prédiction (scellée avant de lire la moindre trace)

* P1 — part des noyaux Marlin MoE (gate, up, down routés) dans le temps GPU de `p81:prefill8k` (≈ 1 394 ms au total,
  chiffre déjà lu au verdict ddw) : **25 à 50 %**, soit 350 à 700 ms.
* P2 — rendement de ces noyaux contre la borne de calcul de 110 ms : **15 à 35 %** de la crête bf16.
* P3 — colle MoE (routage, tri, quantification et dé-quantification, activation, réduction) : 3 à 10 % du préfill.
* Seuil de pièce : je propose une pièce seulement si (a) Marlin MoE pèse ≥ 15 % du préfill ET est à ≤ 50 % de la
  borne de calcul, ou (b) la colle pèse ≥ 5 %. Sinon : verdict « pas de levier ».
* FAUX (P1-P2) si la part est < 15 % ou > 65 %, ou si le rendement est > 50 %.
* Issues nommées :
  - (i) Marlin est déjà près de la borne (> 50 %). Alors pas de levier sur le noyau, et le préfill se perd ailleurs.
  - (ii) Le préfill est dominé par autre chose que le MoE : MLA causale, couche dense, int8 partagé, KDA (5,2 % mesuré
    au ddw). Alors la pièce change de cible, sur décision du chef.
  - (iii) Le temps est dans des noyaux non reconnus par ma classification. Alors la classification est publiée en
    entier, et rien n'est conclu sur la part.
  - (iv) Celle qui me gênerait : la part MoE est forte, mais l'écart vient du découpage en morceaux du préfill (godets,
    plusieurs lancements par expert), pas du noyau. La pièce serait alors dans l'orchestration, pas dans Marlin.

## Prise prévue (≈ 10 min, après poste3)

Instrument p81 inchangé (`nsys-kda-p81.sh`, ATTENDU = HEAD de poste5-ht9), `cd /`, ACVRAM_ARBRE fixé, sous carte.sh,
pause e50.2 si la campagne tient la carte, nvidia-smi au début et à la fin. Classification par noyau dans la plage
`p81:prefill8k` (script hors dépôt, publié avec le verdict). Contre-épreuve : la trace nsys-A de la prise ddw
(84123994d, même instrument) doit donner la même part à ± 5 %.
