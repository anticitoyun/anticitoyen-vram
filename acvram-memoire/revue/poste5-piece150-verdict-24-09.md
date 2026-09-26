# Verdict — pièce 150 : les ~5 ms/pas « de service » sont du PRÉFILL (poste5, 24/09)

* **instrument** : `scratchpad/poste5-p150-24-09/banc-service.py` (charge du banc chat de la 102 : lots de 8, même invite,
  256 jetons, non-stream ; `/metrics` avant et après les lots mesurés, chauffe exclue), `ACVRAM_TRACE_STEPS=1`,
  `frontiere-pas.py` (défaut), nsys en service limité aux lots mesurés ; `prise-service.sh`
* **commit** : 8b54f1ea (branche poste5), `acvram.__file__` = worktree
* **régime** : b=8, `--max-model-len 4096`, -lgc 2700, cpu-safe 100 ; défaut `Qwen3.8-27B-nvfp4` ; contrôle mixte sous
  PROJ_MARLIN ; addendum du scellé (deux chaînes, pytest du chef : lots hors fenêtre)
* **scellé** : `revue/poste5-piece150-scelle-24-09.md` (écrit avant)
* **mesuré** (par lot de 8 × 256 jetons, invite 78 jetons par requête) :

  | alias | pas en service | pas en processus | préfill / lot | reste hors pas / lot | mur / lot |
  |---|---|---|---|---|---|
  | défaut | 25,4 ms | 25,36 ms | **0,41 s** (1,7 pas) | 0,02 s | 6,98 s |
  | mixte (Marlin) | 21,5 ms | 21,5 ms | **1,37 s** (2,0 pas) | 0,02 s | 6,95 s |
  | NInfer (139 c) | — | — | surcoût total 0,33 s/lot | | 4,42 s |
* **verdict** : prédiction (a) **FAUSSE** : aucun coût par pas en service (service = processus au dixième de ms), donc
  ni la détokenisation, ni la livraison, ni le GIL. (c) tenue par le bas : carte oisive entre lots 0,02 s. **Issue (b)
  TENUE** : sur le mixte, le préfill fait 1,37 s par lot (> 0,5), soit tout le surcoût (5,4 ms/pas en équivalent). Sur
  le défaut, il fait 0,41 s par lot (1,6 ms/pas).
* **durée** : prévu 2 × ≤ 10 min / tenu 4 prises (faute, addendum) de 134 à 176 s chacune

## Cause du préfill lent de l'alias mixte (trace de 12:13, 5 préfills)

Par préfill : **0,74 s dans `int8_gemv_kernel<4,16>` / `<4,14>`** (4 607 appels, 142 µs médians), 56 ms de déquant, 121 ms
de GEMM cutlass bf16, 48 ms de dépaquetage Marlin. Le préfill passe séquence par séquence (78 jetons chacune), et
`int8_matmul` (`acvram/kernels/__init__.py`, début de la fonction) envoie tout n ≤ `ACVRAM_INT8_GEMV_MAX` = 80 au GEMV. Ce
GEMV relit le poids par tranches de lignes. Le seuil de 80 a été réglé sur un poids g128 5120² (docstring : croisement
vers 88). Sur les 233 poids par canal servis par leur vue g128 (5120 × 6144 à 17408), à 78 lignes, il perd nettement
contre la déquantification et le GEMM.

## Leviers (ordre de gain)

1. **Mixte : seuil GEMV/GEMM des int8 par canal au préfill.** Descendre le seuil à 16 pour ces poids (les chemins étroits
   du décodage b ≤ 16 ne changent pas). Gain attendu par lot ≈ 0,74 × 2 − (déquant + GEMM) ≈ **−1,0 à −1,3 s/lot**, soit
   ≈ −4 à −5 ms/pas en équivalent au banc. Sortie : GEMM bf16 contre GEMV, pas au bit (ordre des sommes) → KL.
2. **Tous les hybrides : préfill groupé entre séquences** (8 × 78 jetons en un passage, pas 8 × 78). Le défaut paie
   0,41 s/lot contre 0,33 s de surcoût TOTAL chez NInfer. Gain ≤ 0,3 s/lot (~1 ms/pas).
3. Le banc chat, qui attend la fin de tout un lot, rend le préfill visible à chaque lot. En trafic continu, il se
   répartirait autrement. Le levier reste réel pour le temps jusqu'au premier jeton.
