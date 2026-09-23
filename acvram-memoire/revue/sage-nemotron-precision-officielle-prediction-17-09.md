# Sage — Nemotron « précision officielle » : prédiction posée avant la mesure (17/09)

Entrée : Manon 4c1fd65 + ddcbd34 — écart réel au checkpoint officiel : `mixer.in_proj/out_proj` en **FP8** (pas bf16, ma prédiction était fausse sur le format, juste sur les tenseurs) sur les 23 couches Mamba2, **et** `self_attn` des 6 couches `full_attention` jamais quantifié (absent de `quantized_layers` et de `ignore`). Corrigé (garde `model_type == nemotron_h`), reconverti : `Nemotron-3.5-Lightning-30B-A3B-nvfp4-precision-officielle`. Références : srcexl3 1,0795, srcbf16 tout-NVFP4 1,0632, vLLM officiel 0,987, classe ≤ 1,020.

## Prédiction (avant toute tranche)

Ce qui reste en NVFP4 chez nous après le correctif : les experts MoE (le régime Coder, ~1,015) et les projections denses restantes. Les 6-7 points d'excès (1,063 − ~1,015) venaient, selon l'hypothèse, des 23 in/out Mamba2 et des 6 attentions — les tenseurs les plus sensibles d'un hybride (état récurrent, GQA).

**PPL géo 3 tranches : 1,010-1,022, valeur centrale 1,016.** Trois bandes, trois issues :
* **≤ 1,020 → classée** ; l'écart était entièrement les exclusions ; pas de calibration ; ligne des menus « classé, précision officielle », et le régime (quels tenseurs en FP8/bf16) dans la signature du converti, pas dans la vigilance.
* **1,020-1,035 → exclusions partielles** : le reste est dans les experts (même signature que Qwen3.8 non calibré vs calibA : −0,005) ⇒ calibration bras A sur ce converti, prédiction alors 1,015-1,025 ; si ça ne classe pas, on ferme.
* **> 1,035 → la cause n'est pas là** : lecture des tenseurs `nemotron_h` (A_log, dt, D, conv, expert partagé, routage) — contrôle tenseur par tenseur contre le bf16 (cosinus par couche, à sec), **pas** de calibration.

Contrôle de vitesse dans la même fenêtre (obligatoire, le FP8 double les octets de 23 × 2 projections) : **b=12 689,8 → 600-680 t/s, b=1 269,7 → 240-265** ; faux si b=12 < 550 : la voie FP8 est lente (déquant à la volée ?) et non seulement plus lourde — poste à nommer avant de livrer. Issue qui me gênerait : classée à ≤ 1,020 **et** b=12 < 550 — un converti classé qu'on ne peut pas mettre en défaut.

Instrument : le même que `verdict-nemotron-srcbf16-17-09` (3 tranches, bf16 géo 13,416 déjà mesuré : pas à refaire), sha256 du converti et liste des tenseurs hors NVFP4 dans le verdict.

## Mesuré (Laure bc48d96) et suite — 17/09 soir

PPL géo **1,0304** (1,0327 / 1,0430 / 1,0155 ; srcbf16 1,0632) : bande 1,020-1,035 → **exclusions partielles, calibration bras A selon le protocole**. Ma prédiction 1,010-1,022 est réfutée de 0,008 (carnet). Vitesses : b=1 **209,0** (prédit 240-265 : réfuté, −22 %), b=12 **798,9** (prédit 600-680 : réfuté vers le haut, +16 %) — cause lue par Laure : les 46 projections FP8 passent par un GEMM qui lit les poids une fois par pas (b=12 favorisé) contre une GEMV NVFP4 O(b) (b=1 pénalisé par les octets doublés) : le même poste que le GEMM dense Qwen3.8, vu de l'autre bout. Le −22 % à b=1 est le prix de la précision officielle en octets, pas un défaut.

**Confirmé : calibration bras A sur le converti précision officielle** (Manon à sec ~1 h, sha256 corpus + converti ; Laure 20 min, mêmes 3 tranches). Prédiction déjà écrite, précisée sans bande orpheline : **1,015-1,025** ; ≤ 1,020 → classée ; 1,020-1,025 → non classée, chantier qualité Nemotron **fermé** (« non classé 1,02x, précision officielle + calibA », ligne des menus) ; > 1,025 → fermé aussi, et la calibration n'était pas le levier. Contrôle secondaire, scellé : la tranche 2 (1,0430) est à +0,010 des deux autres — si la calibration ne la bouge pas (reste ≥ 1,035 quand les tranches 1 et 3 baissent), le résidu n'est pas dans les experts mais dans un mécanisme dépendant du texte (état Mamba2 long, routage) : à nommer, pas à recalibrer.

Bonus attendu sans mesure dédiée : la GEMM Triton dense M ≥ 4 (palier 1, défaut) couvre aussi les projections NVFP4 restantes de Nemotron à b=12 — à lire dans le prochain JSON b=12, pas de fenêtre pour ça.

## Addendum — calibA Nemotron à 1,43 : résultat INVALIDE, pas « fermé » (17/09 soir)

Jérôme : precision-officielle sans calibration = 1,0304 (sain) ; calibA = **1,43**. Une calibration AWQ ne coûte pas +40 % : elle déplace des échelles entre activation et poids, à sortie mathématiquement inchangée avant arrondi. 1,43 est la signature d'une **échelle appliquée sans sa compensation** (poids multipliés par s, activation non divisée — ou l'inverse), ou appliquée aux mauvais tenseurs (les 46 projections FP8 ne doivent recevoir aucune échelle AWQ : sont-elles dans la liste des tenseurs calibrés ?). Donc : la bande « > 1,035 ⇒ la calibration n'est pas le levier » **ne s'applique pas** — elle supposait une calibration correcte. Verdict : *invalide, cause à nommer*, pas un résultat sur la calibration.

À sec (Manon, ≤ 1 h) : diff tenseur par tenseur calibA vs precision-officielle — liste des tenseurs qui diffèrent, rapport de norme par tenseur. Prédiction : les tenseurs FP8 (`in_proj`/`out_proj` Mamba2) ou `self_attn` des 6 couches diffèrent alors qu'ils ne devraient pas (aucune échelle AWQ ne doit les toucher), ou bien un tenseur expert porte un facteur s² (double application). Trouvé ⇒ correctif + test qui casse + reconversion, et la prédiction 1,015-1,025 reste celle qui compte. Non trouvé en 1 h ⇒ Nemotron reste « non classé 1,0304, précision officielle », chantier fermé sans autre tour : ce n'est pas la cellule phare.

GLM : confirmé touché (141 tenseurs `mlp.shared_expert.*`, chemin inconditionnel). Reconversion en cours ; la prédiction de `sage-expert-partage-cle-portee` § 3 tient (1,009-1,014 ; ≤ 1,013 remplace la colonne) ; la conclusion « calibration pas le levier sur GLM » est **suspendue** jusqu'à cette PPL.
