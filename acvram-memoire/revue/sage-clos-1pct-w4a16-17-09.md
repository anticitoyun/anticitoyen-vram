# Sage — Chantier « 1 % W4A16 GLM » : clos de bout en bout ; trois lignes à écrire, aucune mesure (17/09)

Entrée : `verdict-temoin-dense-w8a8-17-09` (Laure, 1530b4b, main 3dad7f4). Scellé de l'amendement : PPL ≥ 1 % gagné (1,0143, tenu) ; j/s > 25 % perdus (réfuté : `bf16` **+56 %**). Ma prédiction de vitesse était inversée : je bornais la GEMM (FP8 deux fois plus rapide que bf16) et j'ignorais la préparation — `nvfp4_mm_w4a8` requantifiait 337 poids par ligne **à chaque appel**. Un chemin « rapide » qui refait son travail à chaque fois est lent ; une borne de calcul ne dit rien du coût de préparation. Réfuté reste réfuté ; le chantier dérivé n'a pas d'objet, et c'est la meilleure issue.

## Confirmé : clos

Cause (Laurine 429902c), défaut corrigé avec noms de régime, `regime_ligne()` et masques par backend (0971c90), coût vérifié dans les deux sens (1530b4b), cellule GLM en régime réel (≤ 0,001), ligne « mêmes poids » 1,010 / 1,016 / 1,028. Rien à mesurer de plus.

## Ce qui reste, à sec, Jérôme seul

1. **Réétiquetage des 48 : une phrase en tête d'INDEX, pas 48 éditions.** « Toute PPL acvram mesurée en prefill avant 0971c90 (17/09) sur un converti à projections NVFP4 non groupées l'a été en régime `prefill w8a8` (liste : `verdict-prefill-defaut-bf16-17-09`) ; l'effet mesuré va de ≤ 0,001 (GLM `-k48`, 4 projections) à 1,4 % (dense, 337). » Quarante-huit corrections de mémoire en produiraient une fausse ; une phrase datée avec la liste ne se trompe pas.
2. **REGLES § 4, deux entrées datées** : (a) un régime porté par le défaut d'une variable d'environnement absente de l'en-tête est un régime invisible — `regime_ligne()` obligatoire dans tout en-tête, un verdict sans elle n'entre pas dans INDEX (Sage, 17/09) ; (b) une borne d'octets ou de FLOPs ne compte pas la préparation — avant de prédire « plus rapide », demander ce que le chemin recalcule à chaque appel (Sage réfutée, 17/09). REGLES § 9 : « ne jamais quantifier les projections MLA en NVFP4 » (1,33 × bf16), déjà ordonné.
3. **Ligne à l'utilisateur** : « Le défaut de prefill calculait en W8A8 caché depuis l'introduction de `a8` ; 48 convertis sur 55 étaient touchés (≤ 0,1 % à 1,4 % de PPL). Corrigé par le défaut `bf16`, qui est aussi plus rapide (+56 % sur un dense). Aucune reconversion nécessaire. La table GLM est publiée avec ses régimes. » Numéro de version : à Jérôme (un changement de défaut qui change la sortie mérite une ligne de journal de version).

## Ordre

1. Jérôme : les trois points ci-dessus ; ETAT — chantier « 1 % » CLOS, file de carte vide sauf demande ; prochaine question à Sage = résultat du comparatif complet ou scellé réfuté, rien d'autre.
2. Laure, Laurine, Manon, Océane : veille.
