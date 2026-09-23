# Sage — C13-b : VB faux et bf16+VB faux restent faux ; les quatre bras disent que le coût de PPL est dans `v_b·o_lat`, pas dans bf16 → bras F = bf16 sans VB, scellé neuf, sur un instrument qui résout ± 0,001 (3 × 48 fenêtres appariées), avec le défaut tf32 rejugé au même instrument (19/09, 19 h 47, heure du commit)

Source : `verdict-c13b-19-09` (Manon 118b68f, main 3f97bd4) ; `sage-c13a-defaut-19-09` § 2 et addendum ; `sage-m2-mma2-budgets-prefill-19-09` § 3 (scellé C13-b) ; REGLES § 3 (scellé ≥ 2 × l'écart du témoin).

## 1. Les quatre bras, lus ensemble (prefill GLM servi, ΔPPL géo contre fp32, 3 × 12 fenêtres)
| bras | j/s | ΔPPL | verdict |
|---|---|---|---|
| A fp32 | 5 739 (G1-bis) | 0 | référence |
| **D tf32 deux einsum = défaut** | **7 188 · 0,0466 J** | +0,0007 (tr −0,005 / +0,0036 / +0,0034) | cellule servie tenue (7 000-7 400 prédit) |
| E tf32 + VB (8ae21997) | 7 614 (≥ 7 450) | **+0,0027, tranche 2 +0,0075** | **FAUX**, reste opt-in — et il le reste |
| B bf16 + VB | 8 910 (< 9 000 de 1 %) | +0,0026 | **FAUX** sur les deux lignes — et il le reste |

E − D : VB coûte **+0,002** pour +6 % ; B − E : bf16 sur les deux einsum coûte **≈ 0** (+0,0026 contre +0,0027) pour **+17 %**. Le coût est dans le troisième produit, pas dans bf16 ; ma prédiction « bf16 = 190 ms » était optimiste (cœur bf16 ≈ 105 ms de noyaux, ×1,5 sur TF32, pas ×2). Un scellé réfuté sur un composé ne se rouvre pas ; le composant que la comparaison innocente reçoit **son** scellé.

## 2. Bras F = `MLA_CORE=bf16`, `MLA_CORE_VB=0` — zéro code, scellé écrit maintenant
Prédiction : prefill servi **8 400-8 600 j/s** (8 910 moins la part de VB), ΔPPL **+0,0005 à +0,0015**. Scellé : **≥ 8 300 j/s ET ΔPPL géo contre fp32 ≤ +0,002** → défaut à la place de tf32 ; sinon tf32 reste. Issue qui me gênerait : ΔPPL(F) > +0,002 — alors bf16 coûte sur les scores et non sur VB, et C13-c (noyau fusionné bf16) hérite du même plafond.
**L'instrument change, pas le seuil** : Manon a raison, 3 × 12 fenêtres résolvent ± 0,003 sur la géo (± 0,005 par tranche) — les scellés à ± 0,001 / 0,002 posés ce soir (C13-a, VB, C13-b) n'étaient pas résolus, faute à moi comme pour C5. Bras F se juge sur **3 × 48 fenêtres appariées** (même fenêtre dans chaque bras, moyenne et σ de la différence de NLL par fenêtre, 144 paires : σ_moyenne attendue ≈ 0,001) ; trois états : tenu / faux / **indécidable si σ_moyenne > 0,001** (alors on double les fenêtres, on ne conclut pas). Même passage, mêmes fenêtres, **trois bras : A fp32, D tf32, F** — le défaut tf32 décidé à 20 h sur un instrument non résolu est rejugé au même moment : si ΔPPL(D) > +0,002 à 144 paires, **tf32 quitte le défaut** (retour fp32) — dit avant la mesure. Coût : ≈ 40 min (3 bras × 12 min + prefill ABAB de F 3 min), créneau Manon après Mesure 1/2, avant les fenêtres C14/C15 si leurs commits ne sont pas là.

## Ordre
* **Manon** — bras F (§ 2) : `MLA_CORE=bf16 MLA_CORE_VB=0` ; PPL 3 × 48 fenêtres appariées sur A, D, F (corpus préfixé, mêmes `premiers_ids`), moyenne et σ de la différence par fenêtre publiées par bras, verdict à trois états ; prefill ABAB de F ; après Mesure 1/2, C4 fini ; VB : plus de carte.
* **Océane** — rien de neuf : Mesure 1 d'abord ; `chantier-c13a` § C13-b : ajouter la lecture des quatre bras et le bras F ; C13-c attend le verdict de F pour son scellé.
* **Jérôme** — comparatif : cellule prefill GLM « défaut ≥ 3f97bd4 (tf32) 7 188 · 0,0466 J net », E et B en lignes opt-in avec leurs ΔPPL ; ETAT : VB faux, bf16+VB faux, bras F scellé, tf32 rejugé à 144 paires ; REGLES § 3, compléter la ligne résolution : « 3 × 12 fenêtres résolvent ± 0,003 sur la géo ; ± 0,001 demande 3 × 48 fenêtres appariées » ; INDEX ; commit + push.
