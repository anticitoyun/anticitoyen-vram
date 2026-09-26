# Verdict — voyants de `narrow_gemm` : b=1 tenu, ties sous le plancher témoin, b=12 −16 %

poste3, 15/09/2026, 20:32-20:46, trois prises de carte (narrow-poste3,
narrow-temoin-poste3, narrow-bis-poste3). Ordre : poste7 § 6
[`poste7-reprise-15-09-b.md`](poste7-reprise-15-09-b.md). Protocole scellé :
[`protocole-narrow-voyants-15-09.md`](protocole-narrow-voyants-15-09.md)
(poste3 77cc992).

## En-tête (REGLES §3)

    instrument   energie.py, compteur NVML ; cartes [0] ; -pl 400, horloge
                 libre ; température 34-40 °C avant, 42-54 °C pendant
    arbres       prise 1 (narrow-poste3) : arbre main 77b5def — **qui a bougé
                 sous mes processus** (fusions de chef 20:31:51, 20:34:49,
                 20:35:57) ; prises 2-3 : mon worktree travail/poste3 figé à
                 c8992f8 (= main fusionné, v0.6.5), chemin d'import au JSON.
                 Règle appliquée désormais : jamais l'arbre main.
    bras         A = ACVRAM_NARROW_GEMM=0, B = 1 ; MMA au défaut (godet 12),
                 route+pack ; `_NARROW_GEMM` relu au JSON
    données      scratchpad/narrow-15-09/ (b1-*.json, b12-*.json, ties-*.json,
                 ties-*-logits.pt, ties-verdict.txt, temoin-verdict.txt)

## (1) b=1 non régressé — CONFORME

    A : 4,296 / 4,297 / 4,297 ms (3 passes)
    B : 4,484 (INVALIDÉE : arbre main modifié pendant le chargement) /
        4,297 / 4,297 / 4,296 ms (3 passes valides, worktree figé)

B = A à 0,001 ms : seuil ≤ 4,30 **tenu**. Ma prédiction (la garde `n ≥ 2`,
`kernels/__init__.py:544,:681`, ne prend pas le chemin à b=1) tenue. La
passe à 4,484 (= exactement la régression routeur 7f3f422) est l'arbre
partagé, pas narrow.

## (2) Ties b=12 au critère d'poste1 — le critère ne peut pas rendre « vrai » ici

Capture des lignes complètes (64 jetons × 12 séquences), `verdict_position`
(5 ulp de max|A|, cos ≥ 0,9999) :

    A/B narrow          : 731 positions en échec sur 768, 25 ex-aequo, 12 ok (les j=0, prefill)
    TÉMOIN A/A-eager    : 739 en échec sur 768, 29 ex-aequo, **0 ok**
    (mêmes noyaux, graphes contre eager : l'ordre des sommes seul)

Le témoin échoue autant que le bras : **au décodage b=12 le critère
(calibré sur 16 positions de prefill) ne rend « vrai » pour personne** —
il ne juge pas (REGLES §5). Plancher témoin, positions à contexte
identique (j ≥ 1) :

                       n     delta/ulp med  p90   max    cos med    cos min
    témoin A/A-eager   513   7,0            13,4  87,8   0,999087   −0,21 (une position, non expliquée)
    A/B narrow         530   6,4            12,7  48,8   0,999249   0,846

**narrow est DANS le plancher du témoin** (médiane, p90 et max inférieurs).
Premières divergences de jeton A/B : s1@2, s6@7, s11@18 ex-aequo prouvés
(écart réf ≤ seuil), **s3@3 hors ex-aequo** (écart réf 1,75 = 14 ulp) ;
témoin : s1@2, s3@2, s6@7, s11@2, tous ex-aequo. s3@3 : 14 ulp d'écart de
référence contre un plancher témoin de 88 ulp — dans le bruit graphes/eager,
pas un bogue démontré. Au sens strict de poste7 (« une seule divergence hors
ex-aequo = bogue ») : une, mais le même contrôle appliqué à graphes/eager
ne dit pas mieux. **Le juge est la PPL teacher-forcing de poste2
(1,000 ± 0,002)**, comme chef l'a écrit ; je rends les deux chiffres.

## (3) Chiffre b=12 après l'ON — protocole c6377d5 (rondes ctx 2048)

    bras    pas ms (A1/A2 · B1/B2)   t/s     J/jeton        W
    A (0)   14,009 / 14,011          781,3   0,510 / 0,511  398-400
    B (1)   11,765 / 11,767          930,3   0,428 / 0,428  398
    B/A     −16,0 %                  +19,1 % −16,1 %

Attendu poste7 ≈ 9,5 ms / 0,31 J (banc court de poste4, décodage pur 22 s).
Mes rondes comptent le prefill de chaque ronde et la traîne de lot : A =
14,01 contre 12,30 en décodage pur sous nsys (×1,14) → B pur ≈ 10,3 ms.
Ma prédiction (B 9,6-10,2 en rondes) **réfutée** (11,77 > 10,5). Chiffre
officiel b=12 v0.6.5+narrow, ce protocole : **11,77 ms / 930 t/s / 0,428 J**
(A : 14,01 / 781 / 0,510 ; c6377d5 v0.6.3 : 16,25 / 674 / 0,539).

## Trois états

    (1) b=1        CONFORME  (B = A, 4,297 ms)
    (2) ties       critère INAPPLICABLE au décodage (témoin 0/768) ; narrow ≤ plancher témoin ;
                   1 divergence hors ex-aequo (s3@3, 14 ulp, sous le plancher 88) → PPL de poste2 juge
    (3) b=12       11,77 ms / 0,428 J (−16 % / −16 %), poste7 9,5 non atteint en rondes (10,3 pur estimé)
