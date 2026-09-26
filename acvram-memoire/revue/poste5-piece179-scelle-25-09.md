# Scellé — pièce 179 : B' étendu à la déquant int8, et l'admission en deux pas (poste5, 25/09 03 h 3x, AVANT la prise)

Ordre de chef, suite de la 177.

## (1) B' int8
Code : `kernels.int8_matmul`, chemin « déquant bf16 puis GEMM » (M > ACVRAM_INT8_GEMV_MAX, poids non éligible à
cuBLAS int8 : sur le mixte, les tenseurs d'origine fp8), matrice entière et tranches, passé par `_w_partage` (172). La
clé : adresses et formes des codes, échelles et zéros du poids SERVI, relevées avant `vue_g128`. Au bit par
construction.
Réserve : `ext.int8_dequant` écrit directement le bf16, sans intermédiaire fp32 (`acvram_kernels.cu`, `torch::empty`
au dtype). Les octets retenus sont donc les mêmes qu'en nvfp4, déjà comptés par le terme de la 172
(`config.py:395-397`). Le test le MESURE : `test_octets_retenus_dans_la_reserve`.
Tests : `tests/test_depaq_int8_179.py` (au bit, 1 + 7, témoin GEMM groupée ≠, octets retenus ≤ réserve). Bras cassant :
partage retiré du chemin int8 → ROUGE.

## (2) Pourquoi 7 puis 1
Lecture : `runner._admit` (runner.py:1096) admet TOUT ce qui attend, tant qu'il y a des blocs KV et de la place dans le
lot : ni budget de jetons (`ACVRAM_BUDGET_JETONS`, 0 par défaut), ni blocs KV (8 × 92 jetons ≈ 48 blocs, pour des
milliers libres). Le fil moteur (`server/app.py:_run`) sonde la file toutes les 2 ms et fait un pas dès qu'elle n'est
pas vide. Les 8 requêtes du banc arrivent en quelques ms (analyse HTTP, gabarit, tokenisation) : le premier pas prend
celles qui sont là. **Cause : l'ordre et le moment d'arrivée.**
Opt-in : `ACVRAM_ADMISSION_FENETRE_MS` (0 par défaut). Moteur vide et file non pleine : attendre que la file cesse de
grossir pendant la fenêtre (plafond 4 fenêtres). **Sortie** : l'arithmétique d'un lot donné ne change pas, mais la
COMPOSITION des lots de préfill change. Or elle dépend déjà aujourd'hui du moment d'arrivée : c'est le témoin T1/T2 de
la 165, qui n'est pas au bit. Je la chiffre donc en opt-in ; son passage au défaut reste une décision.

## Mesures et prédictions
* `diag172.py` sur le mixte : logits C1 et C2, A (partage coupé) contre B' **au bit** ; forward 8 × 78 et mêlées :
  **−35 à −45 %** (177 : LOT ramène 907 à 398 ms ; B' garde 80-95 % de ce gain).
* Banc chat de la 102, ABC CBA ABC CBA ABC (5 passes par bras), A = `DEPAQ_PARTAGE=0`, B = défaut,
  C = défaut + fenêtre de 5 ms ; pas de préfill relevés par passe :
  * mixte : B/A **+5 à +8 %** ; C/B **+1 à +2 %**, avec ≈ 1 pas de préfill par lot sous C (≈ 2 sous B) ;
  * Qwen3.8 : B/A +0,5 à +2 % (B' nvfp4, préfill ≈ 10 % du lot) ; C/B +0,5 à +2 %.
* **FAUX** si : B' ≠ A au bit ; mixte B/A < +3 % ; C sans baisse des pas de préfill.

**Ajout 25/09 04 h 1x (après la prise tests/diag et le banc mixte, avant l'enquête)** : diag mixte, logits au bit ;
forward C1 (8 × 78) 0 % et 0 réutilisation (78 ≤ 80 : GEMV, rien à déquantifier) ; C2 −21,5 % (576 réutilisations).
Prédiction −35 à −45 % FAUSSE. Banc mixte (15 passes, 0 nulle) : A 326,4 t/s, B 323,7 (**B/A −0,8 %, FAUX** : prédit
+5 à +8), C 332,5 (**C/B +2,7 %**, au-dessus de la prédiction +1 à +2) ; pas de préfill par passe : A 8-10, B 9-10,
C 5. Enquête (`iso179.py`, 8 × L, L = 92, 78, 120, compteurs de chemins int8 et nvfp4) : quel chemin prennent les
int8 du mixte à L = 92, et pourquoi B' n'y gagne rien.
**Ajout 04 h 2x — iso179 (en processus, 8 × L)** : L = 92 : 905,4 → 439,1 ms (−51,5 %, 1 344 réutilisations, 1 240
déquant int8) ; L = 120 : 957,9 → 492,2 (−48,6 %) ; L = 78 : 1 195 = 1 199 (GEMV, 0 partage). B' int8 marche en
processus ; le banc servi n'en tire rien. Prochaine mesure, écrite avant : `eng179.py`, le moteur sans HTTP avec
l'invite réelle du banc. Prédit : si B' y gagne (mur par lot −0,3 à −0,5 s), le défaut est dans le chemin HTTP ou
dans la forme des requêtes servies ; s'il n'y gagne pas, dans le moteur (admission, préfill par séquence, cache).
