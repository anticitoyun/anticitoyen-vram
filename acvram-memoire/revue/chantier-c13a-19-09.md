# Chantier C13-a — TF32 à portée limitée sur le cœur d'attention MLA de GLM (à sec, 25 min) ; à mesurer par Manon

Objectif (Sage, `sage-c7-clos-c13-attention-glm-19-09`) : le cœur d'attention MLA tourne en fp32 plein (`mla.py` : einsum scores q·C, o_lat = probs·V), sans TF32 — 8,6 TFLOP par pas de prefill GLM, ≥ 82 ms des 372 ms ; les tensor cores TF32 (entrées 10 bits de mantisse, > bf16 ; accumulation fp32) sont 8× plus rapides que le fp32 CUDA-core.

Prédiction scellée (Sage, recopiée) : pas de prefill GLM 372 → **315-325 ms** ; scellé (Manon) : PPL = défaut **± 0,001** ; prefill **≥ 6 200 j/s** (5 502 au défaut).

## Le geste

`acvram/engine/mla.py` : `ACVRAM_MLA_TF32=0|1` (défaut 0, `regime.VARIABLES`, liste de garde), contexte `_tf32_coeur()` qui pose `torch.backends.cuda.matmul.allow_tf32 = True` **pendant les deux einsum du cœur seulement** (scores et o_lat, chemin chunké par 256 requêtes et chemin non chunké : 4 sites) et restaure le drapeau après — la portée est le cœur, pas le processus ; `v_b`, `k_b`, les projections et tout le reste gardent leur précision. Rien ne change au défaut.

## Preuve à sec (`tests/test_mla_tf32_c13.py`, 2 tests)

TF32 émulé (mantisse tronquée à 10 bits = borne haute de l'arrondi matériel) sur les formes de GLM (20 têtes, rank 512 + rope 64, 2 047 clés) : écart au fp32 plein **< 2⁻¹⁰ relatif** sur scores, o_lat et y (v_b réel de la couche 3 du converti calibA lu depuis le disque) ; témoin cassant : bf16 (7 bits) dépasse la borne. Contexte : drapeau posé puis restauré ; identité au défaut ; les 4 sites comptés dans le source.

## Reste / non vérifié

* Le temps réel (allow_tf32 sur `einsum` → cuBLAS TF32 sur sm_120 : à vérifier par nsys que les noyaux `*tf32*` apparaissent, sinon la variable est inerte — REGLES § 4, contrôle qui peut rendre faux).
* La PPL sous TF32 (Manon : `ppl-acvram` GLM 3 tranches, `[gMASK]<sop>`, contre le défaut du même arbre).
* C13-b (cœur bf16 sur noyau) après le nsys GLM.

## C13-b (19/09 soir, à sec, commit `oceane-11 6bd11a13`) — `ACVRAM_MLA_CORE=fp32|tf32|bf16` remplace le booléen `ACVRAM_MLA_TF32`

Geste (`sage-m2-mma2-budgets-prefill-19-09` § 3) : une seule variable de régime, trois bras. `mla.py:48-70` : `_MLA_CORE` lu une fois (valeur hors {fp32, tf32, bf16} → `ValueError` à l'import), `_dt_coeur()` (bf16 sous `bf16`, fp32 sinon), `_tf32_coeur(decode=)` inchangé mais armé seulement sous `tf32`. Les opérandes des trois produits du cœur (scores q·C, o_lat = probs·V, y = v_b·o_lat) sont transtypés en `_dt_coeur()` aux 5 sites de préfill (`mla.py:351-375`) et aux 4 sites de décodage (`mla.py:341, 485`) ; la sortie est refloatée avant la suite — la capture graphes ne change pas (mêmes tenseurs d'entrée/sortie, un cast de plus dans le graphe). `regime.py` : `Variable("MLA_CORE", "fp32", …)` ; `cli.py` : garde mise à jour. Hors défaut, `regime_ligne()` imprime `ACVRAM_MLA_CORE=bf16`.

Preuve à sec (`tests/test_mla_tf32_c13.py`, 3 tests, 32 passed / 23 skipped sur les 8 fichiers MLA/régime) : `test_coeur_bf16_borne_a_2_moins_7_par_ligne_formes_reelles` — formes GLM réelles (nh 20, r 512, rope 64, **L = 2047**, t = 64 requêtes), v_b réel du converti si présent sinon aléatoire fixé une fois pour les deux bras, bras bf16 contre fp32 : |Δ| ≤ 2⁻⁷ · max|ligne| sur **chaque ligne** des scores, de o_lat et de y ; **bras qui doit différer** : une ligne de y perturbée de 2⁻⁶ fait rendre « faux » à la borne. `test_contexte_tf32_restaure_et_identite_au_defaut` : sous `fp32` le contexte ne touche pas `allow_tf32`, sous `tf32` il le pose puis le restaure.

Reste (carte, Manon) : bras `bf16` scellé avant mesure par Sage — prefill GLM **≥ 9 000 j/s ET ΔPPL ≤ +0,002** → devient le défaut ; sinon `tf32` (C13-a) reste le candidat. Non vérifié à sec : la vitesse (bf16 double le débit des tensor cores contre TF32, prédiction ≥ +5 % sur le préfill GLM si les 82 ms du cœur sont bien la part dominante) ; l'effet sur la PPL en décodage (2 sites, teacher forcing b=1).

## Défaut tf32 (19/09 soir, `sage-c13a-defaut-19-09` § 1 + addendum 20 h 05) — commit sur `oceane-11`, à fusionner par Jérôme

Trois variables pour trois portées, chacune dans `regime.VARIABLES` et sur la ligne de régime hors défaut ; `ACVRAM_MLA_TF32` disparaît sans alias.

| variable | défaut | portée | ouvre |
|---|---|---|---|
| `ACVRAM_MLA_CORE` | **`tf32`** (depuis ce commit ; C13-a tenu : 7 191 ≥ 6 200 j/s, ΔPPL géo +0,00066 ≤ 0,001, Manon 3bcc173 / main a7397e2) | les deux einsum du préfill, scores q·C et o_lat = probs·V (`mla.py`, chunké et non chunké : 4 sites) | `fp32` référence de qualité · `bf16` bras C13-b |
| `ACVRAM_MLA_CORE_VB` | `0` (fp32) | le 3e produit du préfill y = v_b·o_lat (8ae21997, 2 sites) suit `MLA_CORE` sous `1` | scellé § 2 : prefill ≥ 7 450 j/s ET ΔPPL géo ≤ +0,001 contre 2 produits → défaut `1` |
| `ACVRAM_MLA_CORE_DECODE` | `fp32` | le cœur du décodage y = v_b·o_lat (4 sites, sgemm 1,5 ms/pas à b=12), **indépendant de `MLA_CORE`** | niveau 2, scellé § 2 : sgemm ≤ 0,6 ms ET ppl-decode-kv ± 0,001 ET capture 5/5 → défaut |

Preuve à sec (`tests/test_mla_tf32_c13.py::test_defaut_tf32_prefill_seul_et_portees_opt_in`, 43 passed / 23 skipped sur les fichiers MLA/régime) : sans variable, `_tf32_coeur()` pose `allow_tf32` pendant le bloc et le rend après ; `_tf32_coeur(vb=True)` et `_tf32_coeur(decode=True)` restent inertes au défaut, et chacun s'ouvre par sa seule variable (bras qui doivent différer) ; `bf16` au préfill n'atteint ni le 3e produit sans VB ni le décodage ; comptes de sites vérifiés dans la source (4 / 2 / 4), aucun `ACVRAM_MLA_TF32` restant. Le test bf16 ≤ 2⁻⁷ par ligne (C13-b) est inchangé.

Bras de Manon, à écrire tels quels : défaut = `tf32` ; addendum 8ae21997 = `ACVRAM_MLA_CORE_VB=1` ; niveau 2 = `ACVRAM_MLA_CORE_DECODE=tf32` ; C13-b complet (trois produits) = `ACVRAM_MLA_CORE=bf16 ACVRAM_MLA_CORE_VB=1`, contre `tf32` (défaut) ET contre `fp32` (qualité).

## C13-b, lecture des quatre bras et bras F (Sage, `sage-c13b-faux-bras-f-19-09`, 21 h 05 ; mesures Manon `verdict-c13b-19-09`, main 3f97bd4)

Prefill GLM servi, ΔPPL géo contre fp32, 3 × 12 fenêtres : **A** fp32 5 739 j/s (référence) · **D** tf32 deux einsum = défaut 7 188 j/s, 0,0466 J, +0,0007 (tranches −0,005 / +0,0036 / +0,0034) · **E** tf32 + VB (8ae21997) 7 614 j/s, **+0,0027** (tranche 2 +0,0075) → faux, reste opt-in · **B** bf16 + VB 8 910 j/s (< 9 000 de 1 %), +0,0026 → faux sur les deux lignes. Lecture : E − D dit que VB coûte +0,002 pour +6 % ; B − E dit que bf16 sur les deux einsum coûte ≈ 0 (+0,0026 contre +0,0027) pour +17 % — **le coût de PPL est dans le 3e produit `v_b·o_lat`, pas dans bf16** ; ma prédiction « cœur bf16 ≈ 2× TF32 » était optimiste (≈ 1,5×, ~105 ms de noyaux). Un scellé réfuté sur un composé ne se rouvre pas : B et E restent faux ; le composant innocenté reçoit son scellé.

**Bras F** = `ACVRAM_MLA_CORE=bf16 ACVRAM_MLA_CORE_VB=0` — zéro code (c'est le défaut de VB depuis bd372dfa). Scellé (Sage, avant mesure) : prefill servi **≥ 8 300 j/s ET ΔPPL géo contre fp32 ≤ +0,002** → défaut à la place de tf32, sinon tf32 reste ; prédiction 8 400-8 600 j/s, +0,0005 à +0,0015 ; issue gênante : ΔPPL(F) > +0,002 → bf16 coûte sur les scores et C13-c (noyau fusionné bf16) hérite du plafond. **Instrument** : 3 × 48 fenêtres appariées (même fenêtre dans chaque bras, moyenne et σ de la différence de NLL par fenêtre, 144 paires, σ_moyenne ≈ 0,001), trois états tenu / faux / indécidable (σ_moyenne > 0,001 → doubler, ne pas conclure) ; A, D, F dans le même passage — **le défaut tf32 est rejugé au même instrument : ΔPPL(D) > +0,002 à 144 paires → tf32 quitte le défaut** (dit avant). Manon, ≈ 40 min, après Mesure 1/2. C13-c (noyau fusionné bf16) attend le verdict de F pour son scellé.

## Règle des 2 048 clés (`sage-c14-defaut-tf32-8k-20-09` addendum 02 h 25 ; `verdict-tf32-8k` Manon c2c1e87 : max |Δ| 6,23 % > 2 % à 8 k, géo −0,22 %) — ce commit
Le régime réduit (`MLA_CORE=tf32`, et `bf16` = F, et C13-c) ne s'applique qu'aux morceaux de préfill qui **voient ≤ 2 048 clés** (`_MLA_CORE_MAX_CLES`, `mla.py`) ; au-delà, fp32 pour ce morceau. Chemin chunké : `cles = passe + d1` par morceau de 256 requêtes (les clés vues en causal, passé compris), `_dt_coeur(cles=)` / `_tf32_coeur(cles=)` aux 4 sites + le 3e produit (`cles=total`) ; chemin non chunké : `cles = total`. Le cache est converti une fois par dtype rencontré (`C_dt`). Ligne de régime (module et moteur) : **`mla_core=tf32(≤2048 clés)`** hors fp32. Test à sec `test_regime_reduit_seulement_sous_2048_cles_vues` : 4 096 en 16 morceaux → 8 tf32 (clés 256..2048) puis 8 fp32 ; bf16 même règle ; préfill continué (passe 1 800 + 256 = 2 056 → fp32) ; 43 passed. La cellule GLM L=2 048 (7 268 j/s) reste vraie ; L=8 192 en fp32 : Manon.
