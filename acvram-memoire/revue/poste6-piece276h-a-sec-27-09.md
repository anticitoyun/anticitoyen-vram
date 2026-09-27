# Pièce 276 h — deux flux annexes pour la tour de vision (une image par flux, au bit conservé) : à sec, scellé AVANT le code

poste6, 27/09/2026 17 h 5x. Ordre chef après la 276 g (fusion 0.7.9, défaut 1) : le mur n'était pas tenu parce que les 12 tours
d'une rafale restent SÉRIELLES (un flux annexe, un verrou de capture exclusif) — la dernière requête attend 11 tours avant la sienne.

## Mécanisme
`Engine.encoder_images` prend un flux dans une réserve de `ACVRAM_TOUR_FLUX` (défaut 2) flux CUDA annexes et le rend après ;
le verrou de capture devient lecteurs/rédacteur : N tours en vol (lecteurs) coexistent, une capture (rédacteur) attend qu'il
n'y ait plus de tour en vol et bloque les suivantes. Chaque tour reste UNE image, MÊME `traits_niveaux` (la 276 f a montré que le
LOT n'est pas au bit ; deux appels indépendants sur deux flux le restent — c'est le critère). Deux gestionnaires HTTP en fils
(`to_thread`) lancent donc deux tours qui se recouvrent sur la carte, pendant le pas en cours.

## Critère d'identité
Au bit exigé : traits et niveaux calculés par DEUX fils en même temps sur deux flux = série (12 images distinctes, deux passages,
`identite-276h.py` : 12/12 ou refusé) ; b = 1 jetons 5/5 = main. b = 4/12 : publiés, inapplicables (composition des lots, 276 f/g).

## Prédiction et seuils (échelle M G H H G M, arbre contrôlé, charge hôte par cellule, serveur neuf par bras)
M = main a6268af2e ; G = poste6-276g 5206ee47f ; H = cet arbre. Qwen3-VL-2B, une image 448×448, b = 12 × 7 tours, solo × 5.
Référence 276 g : M mur 297 (moy. des deux bras), TTFT moyen 218 ; G mur 286, moyen 175, `_admit` 0,1.
* **Mur H ≤ 0,85 × M** (≈ 252 ; prédit 215-240 : les 12 tours passent en ≈ 6 paires, 108 → ≈ 60-70 ms sur le chemin de la
  dernière requête — « −80 ms » si le recouvrement est plein, −40 si les deux tours se partagent les SM à moitié).
* **TTFT moyen H ≤ 0,80 × M** (≈ 174) et **H ≤ G** (pas de perte sur ce que la 276 g a gagné).
* `_admit` médian ≤ 10 ms (inchangé) ; solo b = 1 : H = G ± 1 ms (une seule tour, rien à recouvrir).
* Si l'hypothèse est fausse (les deux tours se sérialisent sur la carte, ou le second flux vole ses SM au pas) : mur H = G ± 3 %
  — le contrôle rend « faux ».
* **Issue nommée** : deux tours en vol ralentissent le pas de décodage (SM partagés) → TTFT moyen H > G alors que le mur baisse ;
  ou pire, pas au bit entre deux fils (état partagé dans la tour HF : cache de fréquences rotatives) → refusé, quel que soit le gain.
