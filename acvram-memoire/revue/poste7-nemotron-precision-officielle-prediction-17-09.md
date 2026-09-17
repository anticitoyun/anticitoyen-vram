# poste7 — Nemotron « précision officielle » : prédiction posée avant la mesure (17/09)

Entrée : poste2 4c1fd65 + ddcbd34 — écart réel au checkpoint officiel : `mixer.in_proj/out_proj` en **FP8** (pas bf16, ma prédiction était fausse sur le format, juste sur les tenseurs) sur les 23 couches Mamba2, **et** `self_attn` des 6 couches `full_attention` jamais quantifié (absent de `quantized_layers` et de `ignore`). Corrigé (garde `model_type == nemotron_h`), reconverti : `Nemotron-3.5-Lightning-30B-A3B-nvfp4-precision-officielle`. Références : srcexl3 1,0795, srcbf16 tout-NVFP4 1,0632, vLLM officiel 0,987, classe ≤ 1,020.

## Prédiction (avant toute tranche)

Ce qui reste en NVFP4 chez nous après le correctif : les experts MoE (le régime Coder, ~1,015) et les projections denses restantes. Les 6-7 points d'excès (1,063 − ~1,015) venaient, selon l'hypothèse, des 23 in/out Mamba2 et des 6 attentions — les tenseurs les plus sensibles d'un hybride (état récurrent, GQA).

**PPL géo 3 tranches : 1,010-1,022, valeur centrale 1,016.** Trois bandes, trois issues :
* **≤ 1,020 → classée** ; l'écart était entièrement les exclusions ; pas de calibration ; ligne des menus « classé, précision officielle », et le régime (quels tenseurs en FP8/bf16) dans la signature du converti, pas dans la vigilance.
* **1,020-1,035 → exclusions partielles** : le reste est dans les experts (même signature que Qwen3.8 non calibré vs calibA : −0,005) ⇒ calibration bras A sur ce converti, prédiction alors 1,015-1,025 ; si ça ne classe pas, on ferme.
* **> 1,035 → la cause n'est pas là** : lecture des tenseurs `nemotron_h` (A_log, dt, D, conv, expert partagé, routage) — contrôle tenseur par tenseur contre le bf16 (cosinus par couche, à sec), **pas** de calibration.

Contrôle de vitesse dans la même fenêtre (obligatoire, le FP8 double les octets de 23 × 2 projections) : **b=12 689,8 → 600-680 t/s, b=1 269,7 → 240-265** ; faux si b=12 < 550 : la voie FP8 est lente (déquant à la volée ?) et non seulement plus lourde — poste à nommer avant de livrer. Issue qui me gênerait : classée à ≤ 1,020 **et** b=12 < 550 — un converti classé qu'on ne peut pas mettre en défaut.

Instrument : le même que `verdict-nemotron-srcbf16-17-09` (3 tranches, bf16 géo 13,416 déjà mesuré : pas à refaire), sha256 du converti et liste des tenseurs hors NVFP4 dans le verdict.
