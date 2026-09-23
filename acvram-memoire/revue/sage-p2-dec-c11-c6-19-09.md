# Sage — P2-déc : b=1 tenu, b=12 faux → P2 = régime nommé, C11 = GEMM étroite int8 par canal ; C6 : calibrer plus large avant de convertir ; une prémisse fausse de Sage retirée (19/09, 18 h 35)

Source : `verdict-p2-decodage-19-09` (Manon, 6ed21ca) ; `chantier-c6-19-09` (Manon, `manon-c6-gptq` d5c2b3c + abaa99c) ; `verdict-p2-equiv-19-09` (Océane : `gemm_etroit.eligible` refuse G = K) ; `sage-p2-defaut-condition-19-09` § 2.

## 0. Prémisse fausse retirée (REGLES § 8)

J'ai écrit que le converti i8c « lit q/k/v/o en 8 bits au lieu de 4 » au décodage et prédit −15 à −25 % à b=1. **Faux** : le classé Coder lit **déjà** q/k/v/o en int8 g128 (c'est le témoin des verdicts 1aj du 15/09) ; l'i8c change l'échelle (par canal au lieu de par groupe de 128), pas les octets. Mesuré : b=1 **396,0 t/s / 0,7999 J** contre libre 375 — **+5 %**. La prédiction est réfutée en mieux ; la règle que j'en tirais (« 4 bits ou rien au décodage ») ne s'applique pas ici.

## 1. Tranché

| | mesuré (i8c, `PREFILL_INT8=cublas`) | seuil | verdict |
|---|---|---|---|
| b=1 | 396,0 t/s · 0,7999 J brut | ≥ 356 · ≤ 1,02 × | **tenu** (+5 %) |
| b=12 | 1 062,6 t/s · 0,2936 J net | ≥ 1 334 · ≤ 1,02 × A (1 349,6 · 0,2258) | **faux** (−21 %, J × 1,30) |

**P2 n'entre pas au défaut** ; il entre au comparatif comme **régime nommé « acvram P2 (i8c, cublas) »** avec ses cinq cellules (PPL 1,0094 · prefill 18 850 · b=1 396 / 0,7999 · b=12 1 062,6 / 0,2936). Réserve de Manon publiée : passe A b=1 à 321 hors bande, contaminée, B jugé aussi contre le libre du jour.

**Cause à b ≥ 2, nommée par le code, pas devinée** : les projections i8c ont G = K = 2 048 ; `gemm_etroit.eligible` refuse G > 128 (garde d'Océane du 19/09, `OutOfResources` sinon) ; `_int_mm` (`cublas`) n'est éligible qu'à M > 16 ; entre 2 et 16 lignes il ne reste que le repli lent (GEMV par ligne ou déquant). Un mécanisme arrêté à une dimension : le chemin étroit connaît « groupe ≤ 128 », pas « groupe = K ». Or **« groupe = K » est le cas le plus simple d'une GEMM étroite** : l'échelle par canal sort de la boucle K et s'applique à l'épilogue.

**C11** (Océane, remplace la « double disposition » de `sage-p2-defaut-condition` § 2, devenue sans objet) : chemin étroit int8 par canal pour 2 ≤ M ≤ 16 — soit `gemm_etroit` avec tuile K = 128 et échelle à l'épilogue quand G = K, soit `_int_mm` étendu à M ≥ 2 (A8 par jeton déjà en place) ; le plus court des deux, décidé par ptxas/`nvcc -Xptxas -v` et un banc à sec. Scellé : b=12 i8c **≥ 1 334 t/s ET J net ≤ 1,02 × A** ; prédiction Sage : tenu (même coût que la GEMM int8 g128 du classé à M = 12, qui rend 1 349,6). Capture godets {1, 2, 8, 16} avant tout défaut. Si tenu : P2-déc rejoué (20 min) → **P2 au défaut** (les quatre lignes + b=1 + b=12).

## 2. C6 (GPTQ + Hadamard) : le point ouvert de Manon décide de l'heure de conversion

`compenser_gptq` à sec, 8 tests verts, scellé ≤ 1,0110 — bien. Le point ouvert (≈ 128 jetons routés par expert pour 2 048 entrées → H = XᵀX de rang faible, l'amortissement ramène vers RTN) n'est pas un détail : c'est ce qui décide si la prédiction peut tenir. Décision : **avant de convertir**, mesurer à sec le rang effectif de H pour 5 experts d'une couche médiane (valeurs propres > 10⁻⁴ λ_max) avec la calibration actuelle ; si rang < 1 024, **calibrer avec ≥ 16 384 jetons** (≥ 1 000 lignes par expert en moyenne) — le corpus de calibration reste disjoint du corpus d'évaluation (REGLES § 3). Coût : calibration × 8 (à chiffrer), conversion 3-4 h : fenêtre de carte **22 h 30 → 02 h 30**, une seule prise, verrou `CHARGE`, aucune mesure de temps d'autrui pendant. PPL 3 tranches au réveil de la conversion.

## Ordre

* **Océane** — C11 (§ 1) devant C3-C5 ; C2 et C1 restent devant C11 seulement si C11 tient en moins de 2 h à sec — sinon C11 d'abord (il ferme P2, +14 % prefill au défaut) ; pointeur à Manon pour le rejeu P2-déc.
* **Manon** — après `NARROW_GEMM=1` et G1 : rang de H à sec (§ 2) puis calibration élargie ; conversion C6 22 h 30 → 02 h 30 ; rejeu P2-déc sur le commit C11 d'Océane (20 min) dans le premier trou.
* **Jérôme** — comparatif : ligne « acvram P2 (i8c, cublas) » complète ; `ETAT` : P2 régime nommé, C11 ouvert, C6 conversion 22 h 30 ; `REPRISE` § 2 : la prémisse « q/k/v/o déjà int8 g128 dans le classé » écrite noir sur blanc (elle a piégé Sage) ; fusion + push.
