# Verdict — pièce 187 : tranche du GEMV int8 réglable par plage, tranche 6 (poste5, 25/09)

* **instrument** : `scratchpad/poste5-p187-25-09/` : `banc-tranche.py` (prises 0 et 1), `tests/test_int8_tranche_187.py`
  + cassant (`prise2.sh`), `prise3.sh` (banc chat de la 102, ABBA ×5, fenêtre 20 s) → `prise{0,1,2}.txt`, `prise3-*.txt`,
  `balayage.jsonl`, `cellule.jsonl`
* **commit** : 72ef86f5 (étape 0), 9534dd6d (prise 1), 8bfcbae5 (prises 2 et 3)
* **régime** : défaut + `ACVRAM_INT8_TRANCHE` / `_PREFILL` (A = 16/16, B = 6/6), vus dans la ligne de régime des 30 passes ;
  `repli_eager=0` partout ; -lgc 2700 ; bridage puissance dans 30/30 passes, également réparti ; cpu-safe 100 au début et à la fin
* **scellé** : `revue/poste5-piece187-scelle-25-09.md` (prémisse corrigée, étape 0, balayage, règle de choix)
* **mesuré** : banc chat b=8 mixte **+6,65 %** (326,8 → 348,6 t/s, z 19,6), J/jeton −3,3 % ; b=16 mixte **+10,93 %**
  (563,9 → 625,6, z 50,3), J/jeton −5,4 % ; b=8 Qwen3.8 nvfp4 −0,20 % (z −0,7, témoin nul tenu)
* **verdict** : **au bit TENU** (6/6 contre la tranche 16, cassant ROUGE) ; gain servi **au-dessus des deux prédictions**
  (b=8 : +2 à +5 % prédit ; b=16 : 0 à +3 % prédit, donc prédiction FAUSSE vers le haut)
* **durée** : prévu ≤ 30 min par prise ; tenu 07:4x-07:45:14, 07:57:11-08:00:27, 08:06:01-08:11:38, 08:19:26-08:28:16,
  08:41:36-08:50:10, 09:00:35-09:09:39

## Mécanisme
Registres par NV (ptxas via cuobjdump, bf16) : 4 = 99, 6 = 128, 8 = 168, 12 = 217, 16 = 254. À 256 fils, cela fait 2 blocs par
SM pour NV ≤ 6, contre 1 bloc de 7 à 16. L'occupation double l'emporte sur les relectures de W : 13 lectures au lieu de 5 à
N = 78, qui ne sont que ≈ 0,02 ms chacune. Banc isolé, tranche 6 contre 16 : −22 à −44 % de N = 16 à 78, −3 à −23 % à N = 12.

## Pourquoi b=16 dépasse la prédiction
J'avais écrit que les i8c à M ≤ 16 prenaient aussi les chemins étroits. Le +10,9 % montre que `int8_gemv` sert une grande
partie du pas au godet 16. Le compteur `CHEMINS_INT8` n'a pas été relevé dans la prise (limite) : la cause précise reste à
compter, pas à supposer.

## Proposition (décision : chef)
Tranche 6/6 **au défaut** : au bit, test cassant, gain servi +6,7 à +10,9 %, J/jeton −3 à −5 %. Avant la bascule, la règle
§ 3 demande une passe de capture aux godets {1, 2, 8, 16} avec `warm_graphs` (b=8 et b=16 déjà servis sous graphes,
`repli_eager=0`), puis la suite complète. Autres alias int8 (Coder MLA q/k/v/o, tête int8) non mesurés : au bit par
construction, gain non chiffré. Le docstring d'`int8_matmul` (« tranche de 8 ») est à corriger avec la bascule.

## Leçon
Ma prémisse (NV↑) était fausse deux fois : la tranche réelle, puis le sens du levier. Un témoin existant
(`ACVRAM_INT8_TRANCHE=12`) a tranché en 3 min de carte, sans une ligne de code. Relire l'occupation (ptxas) AVANT de
chiffrer un levier « lire W moins souvent ».

## Bascule au défaut (feu de chef, 1793f434)
Défaut 6/6 dans `acvram_kernels.cu` (`int8_tranches()`, exposé dans l'extension), la table de régime et le CHANGELOG ;
docstring d'`int8_matmul` corrigé. `test_defaut_tranche_6_et_temoins` : cassant (défaut remis à 16) ROUGE.
Capture aux godets 1/2/8/16 (`capture-godets.py`, 09:20) : mixte-i8c et Coder-30B nvfp4, **8/8 ok**, lot = godet,
0 repli eager. Suite complète sous verrou (09:28:01-09:34:54) : **2 801 passés, 1 échec** : `test_octets_retenus_dans_la_reserve`
(179). Rejoué seul à 09:35, il est vert avec la tranche 6 comme avec 16 : l'échec dépend de l'ordre de la suite
(allocations retenues par les tests précédents, même classe que la leçon de la 179) et ne vient pas de la 187. Il n'est
pas corrigé ici.
