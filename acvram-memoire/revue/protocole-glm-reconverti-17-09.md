# Protocole — GLM-4.7-Flash acvram reconverti (Manon, après le bogue de l'expert partagé trouvé sur Nemotron calibA) : PPL × bf16, seuil unique

instrument : `ppl-acvram-17-09.py` 3 tranches `corpus-encode-brut-glm/prive-tr*` avec `PPL_PREFIXE="[gMASK]<sop>"` × bf16 `ppl-refonte-17-09/glm-prive-tr*-bf16` (montage du bras A : calibA k48 = 1,0150 le 17/09 à 13h40, verdict-porte-a4), régime classé, FLA sans objet (GLM dense/MoE) ; converti = chemin donné par Manon à la livraison.
scellé (Sage) : prédiction 1,009-1,014 ; seuil unique ≤ 1,013 tenu (remplace la colonne GLM acvram) / > 1,013 faux ; « la calibration n'est pas le levier sur GLM » suspendu dans ETAT jusqu'à cette mesure.
ordre : après la mesure « GEMV experts » (ordre à venir), dès livraison du converti.
