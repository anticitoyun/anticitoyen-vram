# Chantier C14 — grille de `mla_1p_kernel` au décodage b=1 (19/09, à sec, branche `oceane-c14-mla-1p-grille`)

Source : `sage-glm-decode-budget-c14-c15-19-09` § 2 ; budget `verdict-budget-decode-glm-19-09` (Manon, nsys). Aucun GPU ce soir : le .cu est modifié, non compilé.

## Objectif
`mla_1p_kernel<20,32,4>` coûte 2,09 ms/pas à b=1 = 47 × 44 µs, la même durée à b=12 (2,11 ms) : borné par la chaîne de latence d'un CTA, pas par les octets. Donner au lot assez de CTA (une par tuile à b=1), raccourcir la chaîne d'un CTA, sans changer l'arithmétique.

## Prédiction scellée (Sage, recopiée)
grille ≥ ctx/64 blocs + réduction → **≤ 15 µs par couche à b=1** tenu (44 aujourd'hui) ; sortie **± 1 ulp fp32** de la version actuelle (même arithmétique, recombinaison flash-decoding exacte) ; jetons identiques 256 pas. Réfutation : > 15 µs → la latence n'est pas la grille, ncu avant d'insister.
Précisions à sec (Océane) : (a) le régime nsys est **L = 512**, pas 1 024 : 256 jetons d'invite + 70 pas (decode-nsys.py:11,21,24) → `godet_mla(326)` = 512 (mla.py:139-152, `MLA_BUCKET` 128 mla.py:30, graphs.py:397). (b) « ± 1 ulp » élément par élément n'est pas tenable : o_h somme 512-1 024 termes de signes mêlés, ses petits coefficients portent l'erreur absolue (milliers d'ulp) ; le seuil mesurable est **≤ 8 ulp de l'amplitude de la tête** (2^-23·max_c|o_h|), 4,7 mesuré à sec au pire — et l'ancien chemin est lui-même à 3,2 de l'exact float64. Le vrai critère reste **jetons identiques 256 pas**. (c) Mon modèle : t_CTA = f + n_tuiles·t_tuile, 44 = f + 2·t₃₂, part f/t_tuile **non vérifiée** (pas de ncu) ; issue qui me gênerait : f > 20 µs (lancement + copie de q dominent) → la demi-tuile ne suffit pas et la suite est q hors shared ou fusion des lancements.

## Ce qui existe (acvram_kernels.cu, après édition)
- Occupation avant (d'après le code) : shm = 46 080 (q fp32) + 37 376 (tuile 32×584 bf16) + 5 120 + 240 + 1 152 = **89 968 o = 88 Ko** (l.5647-5649 ; ≤ 99 Ko l.5658) → **1 CTA/SM** (100 ou 128 Ko/SM selon sm_120 — non vérifié, lire `cudaDevAttrMaxSharedMemoryPerMultiprocessor` à la fenêtre), 8 warps/SM sur 48 ; l'ancien commentaire « 384 CTA (2 par SM) » était faux. Ancienne règle S = min(32, ⌈tuiles/2⌉) → 64 lignes/CTA :

| régime | tuiles | S avant → CTA (vagues) | S après (TL) → CTA | octets cache/CTA avant → après |
|---|---|---|---|---|
| b=1 L=512 (nsys) | 16 | 8 → 8 sur 170 SM | 32 (16) → 32 | 73,7 Ko → 18,4 Ko |
| b=1 L=1 024 | 32 | 16 → 16 | 64 (16) → 64 | 73,7 → 18,4 |
| b=1 L=2 048 | 64 | 32 → 32 | 128 (16) → 128 | 73,7 → 18,4 |
| b=1 L=4 096 | 128 | 32 → 32 (4 tuiles/CTA) | 128 (32) → 128 | 147 → 36,9 |
| b=12 L=512 (nsys) | 16 | 8 → 96 (1) | 8 (32) → 96 : **identique** | 73,7 → 73,7 |
| b=12 L=2 048 | 64 | 32 → 384 (2,3) | 13 (32) → 156 (1) | 73,7 → 184 |

Par CTA en plus : q 46 080 o lus, ws H·(R+2)·4 = 41 120 o écrits. Warps actifs à b=1 L=512 : 64 sur 8 160 créneaux (0,8 %) → 256.
- Geste 1, `mla_1p_tranches(L, TL, B)` l.5592-5602 : S = min(tuiles, SM/B) (SM lu par `cudaDevAttrMultiProcessorCount`, l.5561), puis S ramené aux tranches non vides avec l'arrondi du lanceur (point fixe, prouvé par test). Plafond : rien d'autre ne borne S — ws = B·S·H·(R+2) floats (≤ 7 Mo à S·B ≤ 170), `combine` boucle `for s < S` sans borne (l.5537-5554), blockIdx.y ≤ 65 535 ; borne posée `MLA1P_S_MAX` = 1 024 (l.5330) pour les poids en shared du combine.
- Geste 2, demi-tuile l.5638-5640 : TL=16 (RW=2, instanciation `<20,16,2>` l.5660, 67 Ko shm) si H ≤ 20, B ≤ 2 et ⌈L/16⌉ ≤ SM/B — sinon elle n'achète pas de CTA. Chaîne d'une demi-tuile : 16 lignes × 1 152 o = 18,4 Ko à lire (5 tours de 256 uint4 → 2 à PF=4), 16 scores × 20 têtes, softmax sur 16, o_lat 16 lignes × 20 têtes × 2 FMA/fil.
- Geste 3, lectures groupées `MLA1P_PF` = 4 (l.5336 ; q l.5383-5394 : 12 tours → 3 ; tuile l.5406-5428 : 9 → 3 à TL=32, 5 → 2 à TL=16) : pure copie, sortie identique au bit.
- Geste 4, combine l.5528-5556 : poids e^{m_s−M} calculés une fois par tranche en shared, mêmes `__expf`, mêmes produits, même ordre → identique au bit à l'ancien combine.
- Latence attendue (modèle, non vérifié) : f ≈ 14 (lancement 3 + q 12 tours ≈ 7 + reste) → t₃₂ ≈ 15, t₁₆ ≈ 8 ; après : (14 − 5 groupage) + 8 ≈ **17 µs** à b=1 L=512 ; ≤ 15 µs demande que le groupage tienne ses ~5 µs sur q et 2 sur la tuile. b=12 L=512 : lancement identique (S = 8) donc ≤ actuel par construction ; b=12 L=2 048 : 156 CTA × 5 tuiles en une vague contre 384 en 2,3 vagues — ≤ actuel si t_tuile ≤ 12.

## Preuve à sec faite ce soir
`tests/test_mla_1p_grille_c14.py` — **96 passed** (`CUDA_VISIBLE_DEVICES= ACVRAM_TESTS_PENDANT_MESURE=1 python -m pytest tests/test_mla_1p_grille_c14.py -q -p no:cacheprovider`, 0,2 s) : miroir Python de la règle (table ci-dessus, invariants sur 8 lots × 8 contextes, point fixe de rows, jamais moins de CTA qu'avant à b=1, source .cu porte la signature et la demi-tuile) ; recombinaison flash-decoding rejouée (partiels (o, m, l) par tranche avec le softmax en ligne du noyau, combine du noyau) pour S ∈ {1, 16, 32, 64} × TL ∈ {16, 32}, contexte plein et partiel (tranches vides m = −inf) : ≤ 8 ulp d'amplitude entre tout S et S = 1 (4,7 mesuré), tout S ≤ 8 de l'exact float64 ; **témoins cassants** : m non rescalé → > 8·10⁴ ulp d'amplitude ; tranche vide comptée l = 1 → > 80. `tests/test_regime_noyaux.py` + `test_cadrage_perplexite.py` : 20 passed (aucune variable ACVRAM_* ajoutée : le régime est porté par B et L dans le lanceur).

## État
FAIT : règle, demi-tuile, lectures groupées, combine, test à sec, fiche. RESTE : compilation, mesure. NON VÉRIFIÉ : compilation du .cu (aucune ce soir) ; part fixe/tuile des 44 µs ; shared par SM sur sm_120 ; que nvcc ne groupait pas déjà les lectures (SASS) ; tout chiffre « après » ci-dessus.

## Première fenêtre carte (dans l'ordre)
1. compile à sec (`ACVRAM_VERBOSE_BUILD=1`), `-Xptxas -v` inutile (même noyau) sauf pour lire les registres de `<20,16,2>` ; 2. équivalence : q/cache réels (GLM, 3 couches, b ∈ {1, 12}, L ∈ {512, 2 048}) — nouveau `mla_decode_1p` contre `ACVRAM_MLA_UNE_PASSE=0` (chemin `mla_decode` de référence) ET contre le binaire d'avant : ≤ 8 ulp d'amplitude, réfuté au-delà ; 3. jetons identiques 256 pas b=1 ; 4. nsys b=1 (chaîne `chaine-decode-glm.sh`) : `mla_1p_kernel<20,16,2>` ≤ 15 µs/couche tenu, 15-25 = grille juste mais chaîne encore longue (lire ncu : stall sur q ou sur la tuile), > 25 = réfuté ; `mla_1p_combine` ≤ 3,2 µs (actuel) ; 5. b=12 ≤ 2,11 ms ; capture {1, 2, 8, 12, 16}.
