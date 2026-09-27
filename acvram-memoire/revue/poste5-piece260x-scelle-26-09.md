# Pièce 260x — copie signée xor du chemin cublas, AU BIT, au défaut : SCELLÉ (poste5 26/09, avant mesure)

Ordre chef : isoler la trouvaille annexe de la 260, indépendante de l'opt-in FP8. Branche poste5-260x depuis origin/main
(0.7.0, 3d2993ccc).

## Changement
`kernels.copie_signee` : q − 128 (uint8 à zéro 128 → int8) par `q.view(int8).bitwise_xor(-128)` — un noyau, 1 o lu + 1 o
écrit par poids — au lieu de `(q.to(int16) − 128).to(int8)` (trois noyaux, ≈ 10 o de trafic). Appelée par `_i8c_poids`
(copie TRANSITOIRE depuis la 201 : payée à chaque appel cublas hors portée partagée). `ACVRAM_I8C_COPIE` = xor (défaut) |
int16 (témoin). Sortie identique au bit par construction (q ^ 0x80 relu en int8 = q − 128 sur les 256 valeurs).
Micro-banc 260 (formes du mixte) : qkv 10240×5120 à n = 624, 472 → 196 µs par appel ; I = J au bit sur 20 cas.

## Tests (`tests/test_i8c_copie_260x.py`)
Équivalence au bit (256 valeurs, GEMM cublas entier) ; **cassant** : au défaut, `_i8c_poids` ne lance qu'UNE opération de
calcul aten, `bitwise_xor` (TorchDispatchMode) — l'ancienne copie la rend rouge même à sortie égale ; contrôle : le témoin
int16 échoue ce critère.

## Banc servi (`scratchpad/poste5-p260x-26-09/prise.sh`)
`Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c` servi (`--no-prefix-cache`, max-batch 8, 4 096, -lgc 2700, ACVRAM_ECO=off), serveur neuf
par passe, A (int16) B (xor) B A A B B A ; par passe : TTFT b=1 (`ttft-service-p145.py`, L = 78, 512, 2047) puis banc chat
b=8 × 256 (contrôle). Preuve de prise : `/metrics` `int8_chemins` (`cublas` et `i8c_fabrique` > 0) et `regime_ligne`.

### Prédictions
Poids int8 par canal du Coder : q 4096×2048, k/v 512×2048, o 2048×4096 × 48 couches = 906 M. Économie ≈ 5,3 µs par M de poids
et par appel (banc 260) → **≈ 4,8 ms par passe de préfill à n > 80** (3,5-5,5 ms), indépendante de L.
* **L = 512 et L = 2047 : TTFT B − A = −3,5 à −5,5 ms** ; seuil : **gain médian ≥ 2,5 ms aux DEUX longueurs et > 2 ×
  l'étendue des médianes de A**.
* L = 78 (n ≤ 80 : GEMV, pas de copie) : |B − A| ≤ 2 % (contrôle de dispatch ; au-delà : le changement touche autre chose).
* Banc chat b=8 : |B − A| ≤ 1 % (un préfill de ≈ 624 jetons par lot sur ≈ 1,5 s : −0,3 % attendu, invisible).
* Au bit : sortie de B = A (tests) ; aucune mesure de qualité requise.

### Issues nommées
(i) gain < 2,5 ms : la copie n'était pas sur le chemin critique (recouverte, ou M ≤ 16 / éligibilité ailleurs) → reste au
défaut quand même (au bit, jamais plus lent au banc 260), sans revendication ; (ii) L = 78 bouge : un effet de bord à
chercher avant tout défaut ; (iii) le Coder n'appelle pas cublas au préfill (chemin pris ≠ attendu) → mesure nulle.
