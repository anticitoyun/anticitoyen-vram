# Scellé — pièce 156 (d) : F1, F3, F6 du décodage Qwen3.8, en opt-in (poste5, 24/09 18 h 1x, AVANT le code)

Feu de chef (24/09, régime plein). Dossier : `revue/poste5-piece156b-dossier-24-09.md` § 3. Défaut à 0 ; bascule
seulement sur le mot de chef, après le verdict.

## Ce qui change (fichier:ligne lus avant d'écrire)

| # | drapeau (opt-in, 0 = témoin) | site | effet sur la sortie |
|---|---|---|---|
| F1 | `ACVRAM_GDN_PORTES_NOYAU` | `gdn.py:292-298` (`_lot_projete`, voie F2) et `_recurrence_en_place` (voie F4) : `a` et `bt` bruts passés au noyau fla avec `A_log`, `dt_bias`, `APPLY_BETA_SIGMOID=True` (fla 0.5.2 `fused_recurrent.py:124-135`) | ± ulp fp32 : `softplus` de fla en `ex2.approx`/`lg2.approx` (`fla/ops/utils/softplus.py`) contre `log1p(exp)` de torch ; `tl.sigmoid` contre `sigmoid` de torch. Pris SEULEMENT au décodage du lot (`decode_static_batch`, b ≥ 2) : b=1 non touché |
| F3 | `ACVRAM_GDN_NORME_FUSEE` | `gdn.py:161-166` (`_norm_gated`) : un noyau Triton (une ligne de dv = 128 par programme) rend le bf16 | ± ulp : seul l'ordre de la somme des carrés change ; `rsqrt`, `exp` (`libdevice`) et division (`div_rn`) comme torch. Pris partout où `_norm_gated` tourne sur carte : lot, b=1 et préfill |
| F6 | `ACVRAM_NORME_REGISTRES` | `layers.py:587-610` (`RMSNorm`, `add_norm`) → nouveau noyau `rmsnorm_bf16_reg` (`acvram_kernels.cu`, à côté de `rmsnorm_bf16_kernel:7557`) | **visé AU BIT** : même découpe TH, même arbre de somme que le noyau à bloc ; la ligne reste en registres (pas de relecture de `xn` ni de `x`), `w` chargé avant la réduction, la somme des TH/32 warps déroulée dans le même ordre. Si le test au bit tombe, F6 passe en ± ulp et la KL juge |

## Tests (sur carte, sous mon verrou) et bras cassants (chacun doit rendre ROUGE)

* F6 : `torch.equal` contre `rmsnorm_bf16` pour H ∈ {128, 256, 2 048, 5 120, 8 192}, R ∈ {1, 8, 48}, avec et sans résidu.
  Cassant : la somme des carrés d'un fil prise dans l'ordre inverse.
* F1 : sortie et état contre la voie F4 actuelle, écart ≤ 2⁻¹⁶ en relatif (ulp fp32 amplifiées par 1 pas) ; chemin pris
  vérifié. Cassant : `dt_bias` omis (`dt_bias=None`).
* F3 : contre `_norm_gated` torch, ≤ 1 ulp bf16 élément par élément. Cassant : poids de la norme oublié.
* `regime.VARIABLES` : les trois drapeaux y figurent (`test_toute_variable_de_chemin_est_dans_la_table`).

## Vitesse — prédiction (Δ du pas GPU médian, `frontiere-pas.py`, 300 pas, Qwen3.8-27B-nvfp4, config par défaut, ctx 2 048, invite 256, -lgc 2700)

| | b=8 (ms/pas) | b=1 (ms/pas) | raisonnement |
|---|---|---|---|
| F1 | −0,20 à −0,26 | 0 | 6 petits noyaux × 48 couches (sigmoid, exp, neg, add, softplus, mul) ≈ 0,86 µs chacun (dossier § 2) |
| F3 | −0,21 à −0,27 | −0,17 à −0,25 | ≈ 8 noyaux × 48 couches (≈ 380 µs à b=8) remplacés par un noyau de 2,5-3,5 µs par couche |
| F6 | −0,15 à −0,21 | −0,15 à −0,21 | 129 appels ; 3,8 → 2,2-2,6 µs (deux allers-retours L2 et 32 lectures partagées en série de moins) |
| **tout** | **−0,56 à −0,74** | **−0,32 à −0,46** | somme |

* **ABBA** : A (trois drapeaux à 0) B (trois à 1), b=8 : A1 B1 F1 F3 F6 B2 A2 (F1, F3 et F6 seuls, une passe chacune :
  indicatif, pas de verdict par fusion) ; b=1 : A1 B1 B2 A2.
* **FAUX** si Δ(b=8) > −0,39 ms (70 % de la borne basse) ; si Δ(b=1) > −0,22 ms. Au-delà de la borne haute : noté, pas
  revendiqué sans explication.
* Issues nommées d'avance : (a) F6 sans gain (le coût est le lancement, pas les allers-retours) : ma lecture du noyau
  serait fausse ; (b) F3 plus lent que prévu, parce qu'un noyau Triton sous graphe coûte plus que 3,5 µs ; (c) F1 alourdit
  la récurrence (softplus recalculé par chacun des 16 programmes d'une tête) : visible sur la durée du noyau fla, si nsys.

## Qualité — critère (instrument `scratchpad/poste5-p156d-24-09/kl-decode.py`, un processus par bras)

Décodage FORCÉ (teacher forcing, `engine._sample_only` remplacé comme dans `ppl-decode-kv.py`) sous le moteur servi
(pipeline, graphes). Deux compositions de 8 fenêtres de `wiki-gptq.txt` : C1 préfixes 8 × 256, C2 préfixes mêlés
[40, 78, 120, 200, 33, 90, 150, 64], puis 512 jetons forcés par fenêtre. Log-probs fp32 complètes de A gardées pour les
64 premiers pas de chaque fenêtre (1 024 positions), NLL de tous les pas.
* Bras : A, A′ (rejeu), B (tout), F1, F3, F6 ; **témoins mesurés dans la même prise** : T1 (chaque fenêtre seule, b=1),
  T2 (même lot, requêtes admises en ordre inverse).
* **Instrument valide** seulement si A′ = A au bit (log-probs et NLL). Sinon : pas de verdict.
* Par bras X ∈ {B, F1, F3, F6} et par composition :
  1. KL_max(A‖X) ≤ 2 × max(KL_max(A‖T1), KL_max(A‖T2)) ;
  2. désaccord d'argmax(A, X) ≤ 2 × max des désaccords des témoins — contre les témoins, pas contre un chiffre rond
     (leçon de la 150 bis) ; si les deux témoins ont 0 désaccord, X doit avoir 0 ;
  3. PPL par fenêtre (512 jetons) : PPL_X / PPL_A − 1 ≤ +0,5 % sur CHACUNE des 16 fenêtres ; ΔNLL moyen et son z
     relevés (|z| ≥ 2 nommé).
* Une fusion au bit (F6 si le test tient) doit donner X = A au bit : c'est ce qui est vérifié, pas la KL.
* Prédit : F6 au bit ; F1 et F3 KL_max ≤ T1 (T1 change de chemin GDN et de formes GEMV, F1/F3 ne changent que des ulp
  fp32 dans des éléments) ; désaccord d'argmax F1/F3 < T1 ; |ΔPPL| < 0,1 %. Issue qui me gênerait : T2 = 0 et T1 petit,
  F3 au-dessus de 2 × T1 — ce qui dirait qu'un ulp dans la norme de sortie GDN se propage plus que je ne crois.

## Limites posées d'avance

Qwen3.8-27B-nvfp4 seul (config par défaut actuelle, `PROJ_MARLIN=1` au défaut) ; l'alias mixte et les autres hybrides
GDN (Qwen3.5) passent par le même code, non mesurés. Préfill touché par F3 et F6 : couvert par la KL (les préfixes sont
préfillés sous chaque bras), pas mesuré en vitesse.
