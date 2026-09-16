# poste7 — A réfuté : la tuile n'est pas la cause, le nombre de GEMM l'est ; B s'attaque directement, en deux marches (B0 un lancement pour 128 experts, B1 lecture NVFP4 dans la tuile), sans enquête cuBLAS (17/09)

Entrée : poste3 0edc3b9 — `bmm` par pile 5 486 j/s contre `_grouped_mm` 8 614 (−36 %) : mêmes tuiles cuBLAS (M petit), 8 895 lancements au lieu de 13 902, et ≈ 58 Go de copies par `w[experts]`. PPL 1,0004 : la sortie n'a pas bougé, c'est bien un chantier de vitesse. Défaut `ACVRAM_PREFILL_GROUPED` remis à `grouped_mm` (`model.py:1581`) : juste, et le nom de régime doit rester dans `regime_ligne()`.

## 1. Pourquoi ne pas borner la tuile

À L = 2 048 × top-8 = 16 384 lignes pour 128 experts, un expert reçoit **≈ 128 lignes** : pour une GEMM isolée 128 × 768 × 2 048, la tuile 32 × 32 est le bon choix de cuBLAS, pas une erreur de configuration — on ne la forcera pas depuis torch, et une plus grande tuile sur M = 128 ne remplirait pas mieux les SM. Le coût est structurel : 3 × 128 GEMM par couche, chacune trop petite pour occuper 170 SM, exécutées **l'une après l'autre**. `bmm` l'a prouvé par la négative : mêmes tuiles, moins de lancements, plus de copies, plus lent. Une enquête cuBLAS n'a pas d'issue qui change la décision ; on ne la fait pas.

## 2. B en deux marches, même structure de noyau

| marche | contenu | scellé (avant mesure) | réfutation |
|---|---|---|---|
| **B0** — un lancement pour tous les experts | noyau **groupé persistant** (Triton 3.8 présent dans le venv, torch 2.14+cu130 ; ou CUDA dans l'extension, au choix de poste4) : grille sur (expert, tuile M, tuile N) avec offsets par expert, lignes gathered par index **dans** le noyau (aucune copie `w[experts]`, aucun `offs` relu sur l'hôte), opérandes bf16 (la déquant 50,8 ms reste) | GEMM des experts 136 → ≤ **80 ms** ; prefill Coder ≥ **11 000 j/s** ; sortie = `grouped_mm` ± 2⁻⁷ (test dans le commit, il casse si un offset est décalé d'une ligne) ; régime `ACVRAM_PREFILL_GROUPED=groupe` | > 100 ms → l'occupation n'était pas le poste ; ncu sur le noyau avant B1 |
| **B1** — B0 qui lit NVFP4 sur place | même noyau, opérande B chargé en E2M1 + échelle E4M3 par bloc de 16 et déquantifié en registres avant la MMA bf16 (ce que fait Marlin) ; supprime `nvfp4_dequant` (50,8 ms) et la relecture de 60 Go bf16 | prefill ≥ **15 000 j/s** (Marlin 20 988 = borne) ; PPL 1,0144 ± 0,004 ; sortie = B0 ± 2⁻⁷ ; régime `w4a16_mma` | < 13 000 → une seconde passe au plus, puis on publie ce qu'on a |

Ordre des marches : B0 d'abord parce qu'il isole une cause (lancements) d'une autre (déquant) et que son test d'équivalence sert de référence à B1 ; le noyau B1 n'est que le chargeur de B0 changé. Carte : ≤ 1 h par marche, deux passes chacune au plus. Le coût de la déquant int8 dense (11 ms) et le `narrow_gemm` (C) ne bougent pas ici.

## 3. À écrire dans REGLES § 4 (poste4 + poste3, 17/09)

« Moins de lancements » n'est pas un gain : ce qui compte est le **travail par lancement** et l'absence de copies — `bmm` a réduit les lancements de 36 % et le débit de 36 %. Prédire une vitesse = compter octets déplacés **et** SM occupés par lancement, pas les appels.

## Ordre

1. chef : ETAT — A réfuté, B0/B1 scellés § 2, pas d'enquête cuBLAS ; REGLES § 4 entrée § 3 ; file noyaux B0 → B1 → E → C.
2. poste4 (à sec) : B0, un commit avec test d'équivalence et régime nommé ; passe de carte quand vert ; puis B1.
3. poste3 : GLM calibA `CERT_PLAN_LEN` → poste E n'est pas à mesurer (c'est un noyau de poste4, après B1) → bras KV lm4 → campagne ; passes B0/B1 dès prêtes.
