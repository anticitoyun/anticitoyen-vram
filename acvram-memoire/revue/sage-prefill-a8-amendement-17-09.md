# Sage — Amendement à `sage-prefill-a8-verdict-17-09` : prédiction « ≤ 1,008 » retirée (prémisse fausse), cellule GLM remesurée pour son régime seulement, le témoin du W8A8 est un dense (17/09)

Entrées : Laurine 0971c90 (main fc04517) — § 1 + § 3 livrés, 48/55 convertis concernés, Coder officiel non ; correction de prémisse de Jérôme ; `verdict-prefill-bf16-17-09` (Laure, 5b5f972) : `-vllm-direct` sous `bf16` = **1,0096**, tout-torch 1,0104, Marlin 1,0164.

## 1. La correction est juste, la prédiction tombe

`-k48*` n'a que 4 projections NVFP4 non groupées (dense c.0, o_proj, q_a) : l'expert partagé y est int8. Ma prédiction « `-k48-calibA` privé ≤ 1,008 » transportait l'effet de `-vllm-direct` (146 projections, partagé NVFP4 × 46) sur un converti où le mécanisme est quasi absent — un gain qui ne se transporte pas d'un format à l'autre (REGLES § 4), et c'est moi qui l'ai fait. Retirée ; ce qui reste est un comptage : effet ≤ 0,002, sous le bruit (± 0,004). Un résultat sur ce converti ne dit donc **rien** du W8A8, dans un sens comme dans l'autre.

## 2. Ce qui se mesure, et pourquoi

| cellule | pourquoi | coût | prédiction / lecture |
|---|---|---|---|
| GLM `-k48-calibA` privé sous `bf16` | la cellule de la table doit porter le régime par défaut réel, pas un régime estimé « ≈ pareil » | Laure 20 min | 1,015 ± 0,004 ; aucune conclusion sur W8A8 ; le public (1,028) garde son chiffre avec la mention « prefill w8a8 sur 4 projections, effet ≤ 0,002 par comptage » |
| Ligne « mêmes poids » de la table | acquise : acvram W4A16 `bf16` 1,0096 contre Marlin 1,0164 (`verdict-prefill-bf16`) | 0 | se publie telle quelle, avec les trois régimes nommés (a8 1,028 / bf16 1,010 / Marlin 1,016) |
| Témoin dense : `Qwen2.5-Coder-14B-nvfp4` (337 projections), privé, `w8a8` vs `bf16` | c'est là que le mécanisme vit ; il décide si le chantier GEMM W4A16 fusionnée a un objet et documente le changement de défaut pour 48 convertis | Laure 2 × 20 min + 2 × 5 min prefill j/s | PPL : `bf16` sous `w8a8` de 2 à 5 % (146 proj. → 1,8 % ; 337 → plus, non linéaire, je ne chiffre pas au-delà de la fourchette) ; prefill j/s `bf16` −25 à −45 % ; issue qui me gênerait : PPL égales au bruit — alors le coût de a8 dépendait de l'expert partagé de GLM, pas du nombre de projections, et le défaut `bf16` coûte de la vitesse aux denses pour rien |

Scellé du chantier suivant (inchangé) : dense perd > 25 % de j/s **et** gagne ≥ 1 % de PPL en `bf16` → GEMM W4A16 à déquantification fusionnée pour Laurine, après la table ; l'un des deux manque → pas de chantier, on note.

## Ordre

1. Jérôme : ETAT — prédiction ≤ 1,008 retirée (prémisse) ; ligne « mêmes poids » = 1,010 / 1,016 / 1,028 avec régimes ; Coder non concerné ; réétiquetage des 48 d'après la liste de Laurine.
2. Laure (carte, dans l'ordre) : `-k48-calibA` privé `bf16` (20 min) → cellule de la table ; puis témoin dense Coder-14B (§ 2, 50 min), verdict `verdict-temoin-dense-w8a8-17-09` avec `regime_ligne()` en en-tête.
3. Laurine : liste des 48 à Jérôme (faite) ; rien d'autre avant le témoin dense.
4. Manon : veille.
