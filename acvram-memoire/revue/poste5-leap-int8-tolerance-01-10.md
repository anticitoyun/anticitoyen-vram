# LeapQuant INT8 : TOL_NOYAU = 1e-3 était posée à la main — remplacée par des bornes dérivées ; l'écart sature, pas de dérive (poste5, 01/10)
* instrument : à sec `scratchpad/poste5-int8-01-10/derive_deux_quantifs.py` (sortie `.txt`) ; `tests/test_gdn_etat_int8.py` (vérificateur `verifier_resynchro`)
* commit : poste5-leap (voir git log)
* régime : à sec, CUDA_VISIBLE_DEVICES="" ; aucune prise
* scellé : bornes ci-dessous écrites à sec, AVANT le nouveau passage sur carte ; elles ne sont pas réglées sur 1,24e-3
* mesuré : à sec, K = V = 128, 256 pas, 16 gels (deux gels CORRECTS, forme close contre récurrence séquentielle)
* verdict : (1) 1e-3 posée à la main, l'aveu est entier ; (2) pas de dérive : l'écart sature à ~1,4 ε ; test refait par pas resynchronisés
* durée : 0 (à sec)
## Réponses aux deux questions de chef
1. **TOL_NOYAU = 1e-3 était posée à la main**, avec le commentaire « autre ordre de sommation, arrondis à égalité » et
   aucune borne derrière. Elle comparait deux trajectoires LIBRES, or deux gels corrects ne rendent pas les mêmes codes :
   une différence fp32 infime avant un gel fait basculer des arrondis INT8, et l'écart devient de l'ordre d'une erreur de
   quantification.
2. **L'écart croît puis sature** : gel 1 2,0e-4, gel 2 7,3e-4, gel 4 2,4e-3, gel 8 4,8e-3, gels 11-16 5,6-6,5e-3, avec
   ε (erreur d'un gel) ≈ 4,3e-3, soit écart/ε 0,04 → 1,35-1,45. C'est la saturation à √2 ε de deux quantifications
   indépendantes : il n'y a pas de dérive au-delà. Le 1,24e-3 de la carte au gel 2 a le même ordre que les 7,3e-4 à sec,
   où seul l'ordre de reconstruction différait ; le noyau diffère aussi par l'itération de puissance et les exp.
## Le test refait (bornes d'analyse d'erreur directe, aucune élargie pour passer)
Chaque pas part du MÊME état compressé (recopié), et chaque écart a sa borne :
* sortie et enregistrement w : γ_n·(décomposition de l'état en valeurs absolues : résidu, compensateurs, tampon), avec
  n = 2K + 2P + R + 16 et u = 2⁻²⁴ ; w stocké en fp16 : + 2⁻¹⁰ |w|. La borne est élément par élément ;
* état après un gel : (‖C_T‖‖B_T‖ + ‖C_R‖‖B_R‖)/254 + γ_n‖·‖ en norme de Frobenius par tête, soit la demi-marche de
  l'INT8 symétrique de chacun autour du même état d'avant gel ;
* justesse et dérive, à part : le noyau en roue libre sur 256 pas contre la récurrence exacte (TOL_INT8 3e-2, sans dérive).
Contrôle qui peut rendre faux : une lecture fautive Z/126 casse dès le pas 0, à 123 × la borne. Une variante correcte
(reconstruction séquentielle, celle du noyau) tient sur 40 pas et 2 gels. À sec : 6 passés, 4 sautés (carte).
SUITE : prise demain (tests carte + banc) ; nouveau sha quand c'est tenu sur carte.
## Issues de la prise carte (scellées avant ; script `scratchpad/poste5-int8-01-10/prise.sh <commit>`, HEAD asserté)
**C1** resynchro, roue libre, couche et export verts → banc int8, jugé au § 3 du scellé (moyenne b=12 ≤ 30 µs/couche).
**C2** resynchro ROUGE sur la sortie hors gel → le noyau `_pas_kernel` calcule autre chose que la référence : bogue,
arrêt, aucun banc. **C3** resynchro rouge au gel seulement → `_bord_kernel` quantifie hors de sa demi-marche (arrondi,
échelle, lissage) : bogue du gel. **C4** resynchro verte, roue libre rouge ou en dérive → les deux sont exacts pas à pas
mais le gel du noyau est moins juste que celui de la référence (itération de puissance) : chiffrer ε des deux, ne pas
élargir TOL_INT8. **C5** export → chargement non au bit → aller-retour faux, bloquant pour le service.
