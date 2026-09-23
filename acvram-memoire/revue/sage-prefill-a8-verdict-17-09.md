# Sage — `ACVRAM_PREFILL=a8` : défaut → `bf16`, W8A8 devient un régime nommé et opt-in ; les PPL passées se réétiquettent, seules celles de la table se remesurent (17/09)

Entrée : `verdict-diff-moe-prefill-w4a8-17-09` (Laurine, 429902c, main a83cfee) ; Laure 6df34fb (tout-torch réparé : 1,0104 contre noyaux 1,0280). Cause du 1 % : `kernels/__init__.py:579-593` — au-delà de `gemv_threshold`, `ACVRAM_PREFILL` vaut `a8` par défaut et `nvfp4_mm_w4a8` (`fp4_gemm.py:257`) requantifie le poids déquantifié en E4M3 par ligne et l'activation en FP8 : W8A8 à poids doublement quantifié, erreur RMS 3,6-4,0 % contre 0,14 % en bf16. Touche l'expert partagé et la couche dense 0 de GLM, et **toutes** les projections NVFP4 non groupées de tout modèle au prefill > 32 jetons sous extension.

## 1. Décision sur le défaut

* **`bf16` devient le défaut.** Une optimisation qui change la sortie est un bogue (REGLES § 7) ; a8 change la PPL de ≥ 1 % sans test d'équivalence, et ne peut pas en avoir un : la double quantification est dans sa définition. Il reste disponible **opt-in, sous un nom qui porte le régime** (`ACVRAM_PREFILL=w8a8`, et le régime servi par `/v1/models` / le journal de chargement l'affiche) — jamais plus « W4A16 » quand l'activation est FP8. `a4` idem (`w4a4`).
* **Coût à mesurer avant de le regretter (Laure, 2 × 5 min)** : prefill pp2048 j/s `bf16` vs `w8a8` sur GLM `-k48-calibA` et sur un dense NVFP4 (`Qwen2.5-Coder-14B-nvfp4`, toutes projections). Prédictions : GLM −0 à −5 % (deux projections sur 47 couches) ; dense −25 à −45 % (GEMM bf16 après déquant contre `_scaled_mm` FP8). Si le dense perd > 25 % : chantier **GEMM W4A16 prefill à déquantification fusionnée** (ce que Marlin fait, 18 117 j/s sur GLM) pour Laurine, **après la table**, scellé j/s ≥ 0,9 × w8a8 et PPL = bf16 ± 0,004. Pas de correctif de vitesse avant d'avoir le chiffre.
* Prédiction pour la mesure en cours de Laure (seuils déjà scellés, on ne les touche pas) : `-vllm-direct` sous `bf16` = 1,010-1,014 (le tout-torch 6df34fb à 1,0104 est la borne : même arithmétique, tête fp32), donc **sous Marlin (1,016)** — à poids identiques, notre W4A16 honnête est meilleur que le sien, ce qui se publie avec la table, pas avant la mesure.

## 2. Les PPL déjà scellées : réétiqueter, pas refaire

Un chiffre exact hors de son régime est faux comme décision, il n'est pas faux comme chiffre (REGLES § 4). Chaque PPL acvram publiée en prefill sur un converti portant des NVFP4 non groupées était en régime **« prefill w8a8 »**, pas W4A16 : Jérôme ajoute ce régime, en une phrase, dans INDEX à chaque verdict concerné (Laurine donne la liste : convertis dont le manifeste porte au moins une projection NVFP4 hors experts, à sec, 10 min). **Ne se remesurent que les cellules de la table à cinq** : GLM `-k48-calibA` privé + public (Laure, 2 × 20 min, prédit privé ≤ 1,008) et Coder officiel **si** sa liste contient une telle projection (prédit : aucune — q/k/v/o int8 par 1aj D, experts groupés — donc inchangé, à prouver par le manifeste). Prefill j/s de la table : remesurés sous `bf16` (§ 1), les vitesses de décodage ne changent pas (chemin GEMV ≤ 32 jetons).

## 3. Ce qui empêche la récidive (même commit que le défaut)

* **L'en-tête de mesure porte le régime des noyaux** : une fonction `acvram.regime_noyaux()` rend toutes les variables `ACVRAM_*` qui choisissent un chemin (`PREFILL`, `MOE_MMA`, `MOE_DECODE_MMA`, `NVFP4_GEMV_MAX`, `MLA_*`, `DISABLE_*`) avec leur valeur effective ; `ppl-acvram` et `energie.py` l'impriment. Un verdict sans cette ligne n'entre plus dans INDEX (REGLES § 3, à compléter). Contrôle impossible à sauter, pas une consigne.
* **Les masques `PPL_MASQUE_NOYAUX` énumèrent les backends** : la porte `get_extension() is not None` est passée sous le masque — un mécanisme arrêté à une dimension de plus (MECANISMES). Test : chaque chemin sélectionné par une variable est masquable, et le test casse si l'on ajoute un chemin sans masque.
* Test d'équivalence du prefill : PPL prefill 2048 sous `bf16` = tout-torch ± 0,002 sur 1 tranche (CPU possible en 2 couches), dans le même commit que le changement de défaut.

## Ordre

1. Jérôme : ETAT — cause du 1 % localisée (Laurine 429902c), défaut `ACVRAM_PREFILL` → `bf16` ; verdicts passés réétiquetés « prefill w8a8 » d'après la liste de Laurine ; table à cinq attend les cellules § 2 ; une ligne à l'utilisateur : la qualité publiée jusqu'ici sous-estimait acvram d'≈ 1 % sur GLM, corrigée par un défaut, pas par une reconversion.
2. Laurine (à sec, un commit) : § 1 défaut + noms de régime, § 3 `regime_noyaux()` + masques + test d'équivalence ; liste des convertis à projections NVFP4 non groupées (§ 2) ; verdict `verdict-prefill-defaut-bf16-17-09`.
3. Laure (carte, dans l'ordre) : finir la mesure scellée en cours ; puis PPL `-k48-calibA` privé + public sous `bf16` (2 × 20 min) ; puis prefill j/s `bf16` vs `w8a8` GLM et dense (2 × 5 min) ; verdict `verdict-table-glm-bf16-17-09`.
4. Manon : le profil par couche (`sage-bissection-w4a16-verdict` § 2) est **annulé** — la cause est trouvée par un instrument plus court ; retour à la veille.
