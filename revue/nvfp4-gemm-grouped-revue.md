# Revue : nvfp4_gemm_grouped_kernel contre les tests

Date : 13/09/2026 — Laurine, chantier 2.

Quatre points passés en revue, chacun avec verdict (couvert/non couvert par
`tests/test_gemm_grouped.py`).

## 1. padded_in

**Mécanique.** `quantize_nvfp4` arrondit K au multiple de `BLOCK=16` via
`_pad_k` (nvfp4.py:207). Le noyau exige `K % 64 == 0` (acvram_kernels.cu:1682).
L'appelant (`_gemm`, model.py:718) passe `pile[4]` = `padded_in`, pas le K
logique. L'activation `xs` est rembourrée de zéros si nécessaire (model.py:753).

**Couverture.** Les K testés (1536, 2048) sont déjà multiples de 64 →
`padded_in == K`. Le chemin de rembourrage de l'appelant (xs plus large que
l'entrée logique) n'est **pas** exercé. Le noyau lui-même fonctionne
correctement quel que soit K multiple de 64 — le risque résiduel est dans
l'appelant.

**Verdict : ⚠️ noyau couvert, rembourrage appelant non testé.**

## 2. Écriture [G, M] avec M non multiple de GG_BM (64)

**Mécanique.** La grille arrondit M vers le haut : `(M + GG_BM - 1) / GG_BM`
(cu:1687). Le dernier bloc peut dépasser M ; la garde `row0 + n < M` (cu:1671)
empêche toute écriture hors limites. La sortie `y` est allouée à `[G, M]`
exact (cu:1685).

**Couverture.** M=1536 et M=2048 sont tous deux multiples de 64 → la garde
n'est jamais déclenchée. Un M impair (ex. 1500 ou 2304) exercerait la garde.

**Verdict : ⚠️ garde correcte mais non exercée.**

## 3. Ordre des nibbles (demi-octets)

**Mécanique.** Le noyau extrait chaque demi-octet ainsi (cu:1644-1645) :
```c
const unsigned char b = o[i >> 1];
const unsigned char nib = (i & 1) ? (b >> 4) : (b & 0xF);
```
Index pair → demi-octet bas, index impair → demi-octet haut. Cet ordre doit
correspondre à `pack_e2m1` dans nvfp4.py (qui empaquette deux codes 4 bits par
octet, pair en bas, impair en haut).

**Couverture.** Si l'ordre était inversé, la déquantification donnerait des
valeurs fausses et les tests échoueraient. Les 8 cas paramétrés de
`test_gemm_groupee_contre_reference` et les 6 cas de
`test_noyau_contre_grouped_mm` le vérifient implicitement.

**Verdict : ✅ couvert.**

## 4. Échelle globale par expert

**Mécanique.** Le noyau lit `gscales[e]` une fois par bloc (cu:1613) et
l'applique à chaque échelle de bloc (cu:1638-1639) :
```c
const float s0 = e4m3_to_float(sc[0]) * gscale;
```
Chaque expert a sa propre échelle. La référence (`dequantize_nvfp4`) applique
l'échelle globale par tenseur, puis `_pile` empile les résultats.

**Couverture.** `_pile` crée un `global_scale` distinct par expert (issu de
`quantize_nvfp4` sur des poids aléatoires indépendants). Si un expert
empruntait l'échelle d'un autre, les valeurs déquantifiées divergeraient. Les
14 cas paramétrés (8+6) le vérifient.

**Verdict : ✅ couvert.**

## Résumé

| Point                    | Verdict |
|--------------------------|---------|
| padded_in                | ⚠️ noyau OK, appelant non testé |
| M non multiple de 64     | ⚠️ garde correcte, non exercée  |
| Ordre des nibbles        | ✅ couvert                       |
| Échelle globale / expert | ✅ couvert                       |

Les deux points ⚠️ sont des lacunes du harnais de test, pas du noyau. Ajouter
un cas avec K non multiple de 64 au départ (ex. 1500 → padded_in=1504 remonté
à 1536 par l'assertion) et un M non multiple de 64 (ex. 1500) comblerait les
deux.
