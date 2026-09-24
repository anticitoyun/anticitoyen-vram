# Verdict — pièce 123 : AWQ des experts porté par le chemin tensor (prise de justesse) — 24/09 00 h (poste1)

* **instrument** : `scratchpad/poste1-123-24-09/prise-123.sh` (garde HEAD) — `tests/test_moe_tensor_awq_123.py` + tests tensor voisins, `casser-123.sh carte-b`, `kl-123.py` (kl-b.py + `echelle_awq` atteinte), `outils/gpu/mesure/capture-godets.py` 8,16
* **commit** : 8b09381d (poste1-mtp ; code 123 = 04a94d0d)
* **régime** : Qwen3-Coder-30B-A3B-assemble-S1b-proj-tete-i8-23-09, témoin `ACVRAM_MOE_TENSOR=0` (GEMV, `echelle_awq=gemv`), candidat défaut (`echelle_awq=aligneur`) ; horloge libre (justesse, pas de temps)
* **scellé** : `revue/poste1-piece123-awq-experts-tensor-23-09.md` § 3-4 (écrit avant) — KL b=12 ≤ témoin + 0,025, b=1 identique au bit, tests (a)(b) au bit, capture 8/16 sans repli
* **mesuré** :
  * Tests : **4 failed, 24 passed** — les 4 `test_aligneur_xs_au_bit` (godets 2/5/12/16) ; (b) vert à 3 godets, tensor voisins verts. Bras cassant (b) : ROUGE (3/3), attendu.
  * KL b=12 : témoin 0,1316, candidat 0,1317, ΔKL_max par invite ≤ **+0,0039** ; 5/5 ≤ 0,74 ; `echelle_awq` = ["aligneur"] (candidat), ["gemv"] (témoin), 0 refus « chemin tensor non pris ».
  * KL b=1 : kl_max global égal (0,1144) mais **pas au bit** : invites 0, 1, 3 diffèrent (|Δ| ≤ 0,022 par pas, argmax 8/8 dans les deux bras), invites 2 et 4 au bit ; chemin atteint candidat `gemv_marlin+marlin_tensor`.
  * Capture godets 8 et 16 : ok, graphes on, 0 repli eager, 4,872 / 6,396 ms/pas (horloge libre, non comparable).
* **verdict** : **FAUX au scellé**, deux causes nommées :
  1. **Défaut de code trouvé par (a)** : `moe_aligner_petit_xs` indexait la table par `e * K` alors que la garde admet une table [E, ≥ K] (`acvram_kernels.cu:4705` au commit 04a94d0d) ; le test (a) donne une table de largeur K + 64 exprès. Sans effet sur S1b (table de largeur K : (b) et KL b=12 verts), faux pour toute table rembourrée. **Corrigé** (pas de ligne `table.stride(0)`), recompilé à sec rc 0 ; (a) et bras (a) à rejouer sur carte.
  2. **Prédiction « b=1 identique » réfutée** : le préfill COURT (T ≥ 8 jetons de l'invite) passe par `_forward_grouped` → chemin tensor (pièce 82 ter) ; S1b le refusait à cause des tables AWQ, la 123 l'ouvre. Les premiers logits changent donc à 2⁻⁷ près (autre GEMM) ; le décodage b=1 reste GEMV. Ce n'est pas un défaut de justesse (KL b=1 égale au 10⁻⁴ près), mais le scellé l'excluait : réfuté reste réfuté.
* **durée** : prévue ≈ 10 min, prise 23:50:20 → 23:52:16 (116 s de script après attente du verrou) ; compute-apps début = fin (llama-server 4627 seul)

Décision au chef (une ligne, deux options) : (i) nouveau scellé « b=1 : décodage au bit, préfill court ≤ 2⁻⁷ comme 82 ter » et S1b → cellule servie de poste2 après la 123-bis ; ou (ii) AWQ-tensor limité au décodage (préfill court en GEMV) pour tenir le scellé tel quel.
Rejeu obligatoire dans tous les cas (123-bis, ≈ 6 min) : tests (a)(b) après correctif + bras cassant (a) (cache séparé).
