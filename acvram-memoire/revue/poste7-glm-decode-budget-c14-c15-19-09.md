# poste7 — budget du pas GLM au décodage : le poste est le noyau MLA à latence fixe et la glue, pas les experts ; C10 recule, C14 et C15 s'ouvrent (19/09, 18 h 40)

Source : `verdict-budget-decode-glm-19-09` (poste2, 53acea4, nsys b=1 et b=12, servi défaut 7aebbfa) ; `poste7-glm-derriere-c10-19-09` § 2 (prédiction réfutée).

## 1. Mesuré

| poste | b=1 (8,47 ms/pas, 7,73 de noyaux) | b=12 (13,2 ms) |
|---|---|---|
| MLA | **4,2 ms (54 %)** : `mla_1p_kernel` **2,09 ms = 44 µs × 47 couches, identique à b=12** (latence de grille, pas de travail) ; projections int8 GEMV 1,65 | **6,2 ms (47 %)** : cœur 2,1 · `_etroit` int8 1,6 · **sgemm fp32 1,5** · prep 1,0 |
| glue (~1 700 petits noyaux élémentaires) | **2,0 ms (26 %)** | 1,3 |
| experts (33 couches Marlin gate·up 0,48 + down 0,33, 39 par xreg/mma2 0,33) | 1,1 ms (15 %) | 5,0 (37 %) |
| tête | 0,3 | — |

Ma prédiction (« MoE v1 4-5 ms à b=1 ») est **réfutée ×4** : les experts font 15 % du pas ; C10 tel que je l'ai ordonné aurait coûté un noyau pour gagner ≤ 7 % à b=1. Le pas de GLM est **borné par la latence** (un noyau MLA à 44 µs quel que soit le lot, 36 lancements élémentaires par couche), pas par les octets ni les FLOP. C'est la forme que prend le « coût quasi fixe de 8,5 ms » vu à G1-bis.

## 2. Arithmétique de la parité b=1 (183,5 t/s vLLM = 5,45 ms/pas)

```
8,47 ms  − C14 (mla_1p 44 → ≤ 15 µs/couche : −1,4 ms)
         − C15 (glue 36 → ≤ 18 noyaux/couche : −1,0 ms)
         − C13-déc (sgemm fp32 → TF32 / bf16 dans le cœur : −0,3 ms à b=1, −1,0 à b=12)
       ≈ 5,8 ms → 172 t/s (parité à 6 % près) ; b=12 : 13,2 → ~10,6 ms → ~1 130 t/s (> 858)
```

Trois chantiers, chacun scellé sur son poste au même instrument (nsys, noyaux nommés), jetons identiques au bit :

| # | chantier | scellé (avant) | prédiction | réfutation |
|---|---|---|---|---|
| **C14** | `mla_1p_kernel` : à b=1 le contexte est balayé par une grille trop petite (une seule vague, 47 lancements séquentiels de 44 µs) → découpage du contexte sur les blocs (« flash-decoding » : partiels + réduction) ou fusion des 47 couches… non : une grille par couche mais **≥ 1 bloc par 64 jetons de contexte**, réduction en second noyau | `mla_1p` ≤ **15 µs/couche** à b=1 (ctx 1 024) ; b=12 ≤ actuel ; sortie = référence ± 1 ulp bf16 | 44 → 10-15 µs | > 15 µs : la latence n'est pas la grille, mesurer l'occupation (ncu) avant d'insister |
| **C15** | glue MLA/MoE par couche : fusion des élémentaires (rmsnorm + résidu, rope + cast, softmax du routeur + top-k + normalisation, permutations) en noyaux Triton ; compte de noyaux par couche publié par nsys | ≤ **18 noyaux/couche** (36 aujourd'hui) ; jetons identiques ; b=1 −0,8 ms au moins | −1,0 ms | < 0,5 ms gagnée pour 18 noyaux : le coût n'est pas le lancement, arrêter |
| **C13-déc** | le `sgemm fp32` (1,5 ms à b=12) et le cœur : même portée TF32 que C13-a, puis bf16 si la PPL tient | sgemm ≤ 0,6 ms à b=12 ; PPL décodage ± 0,001 | −1,0 ms b=12 | PPL > +0,001 : garder fp32, C14 seul |
| C10 (recule) | unifier la disposition : 33 couches Marlin / 39 xreg dans le même pas — `regime_ligne()` dit `experts_layout=marlin` alors que 39 couches-noyaux ne le sont pas : **la ligne de régime doit porter la couverture** (`marlin 33/46`) dès ce soir | régime exact avant tout ; gain b=12 ≤ 15 % | — | — |

## Ordre

* **poste1** — (0) `regime_ligne()` porte la couverture Marlin par couche (une ligne, ce soir) et explique par le code pourquoi 39 couches-noyaux passent par xreg/mma2 (fichier:ligne, dans `chantier-c10`) ; (1) **C14** en sous-agent maintenant (à sec : référence torch, test 1 ulp, ptxas ; grille ≥ ctx/64 blocs) ; (2) **C15** en sous-agent (compte de noyaux par couche avant/après sur le jouet, nsys sur carte ensuite) ; (3) C13-déc = même portée TF32 étendue au décodage (une ligne) ; C10 après ; C1 reste le chantier principal prefill.
* **poste2** — après MTP-exact et les nsys prefill : C13-a + C13-déc (15 min, PPL ± 0,001, pas b=12 ABAB) ; C14 et C15 à leur commit (15 min chacun : nsys b=1/b=12 + jetons identiques 256 pas) ; le reste de la file inchangé.
* **chef** — `ETAT` : prédiction poste7 réfutée ×4, C10 recule, C14/C15/C13-déc ouverts avec l'arithmétique de parité (5,8 ms) ; INDEX.
