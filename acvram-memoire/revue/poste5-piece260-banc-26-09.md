# Pièce 260 — micro-banc W8A8 int8 (origine fp8) : RÉSULTAT (poste5 26/09 06 h 1x)

Prise sur 9f5ce6a67 (0.7.0 + 260), 06:09:07-06:11:58, carte 0 sans PID hors prise avant/après ; tests 260 + régime + défauts +
porte A8 : 25 passed. Brut : `scratchpad/poste5-p260-26-09/banc260.{json,txt}`. µs par appel.

| forme | n=78 T_part | I_part | J_part | I/T_part | n=624 T_seul | I_seul | J_seul | I/T_seul |
|---|---|---|---|---|---|---|---|---|
| qkv | 175,6 | 40,5 | 74,4 | **0,23** | 1 029,9 | 196,2 | 472,2 | 0,19 |
| gate | 101,6 | 28,8 | 40,8 | **0,28** | 598,5 | 114,9 | 221,4 | 0,19 |
| out | 92,2 | 30,1 | 42,9 | **0,33** | 608,4 | 125,6 | 235,0 | 0,21 |
| down | 281,6 | 79,5 | 139,8 | 0,28 | 1 741,7 | 415,7 | 876,4 | 0,24 |

* **S2 (critère principal) : Σ I_part / Σ T_part GDN à n = 78 = 99,3 / 369,4 = 0,269 ≤ 0,50 — TENU** (prédit 0,44-0,55 : FAUX
  dans le bon sens). n = 624 : 0,19-0,24 ≤ 0,50 sur les quatre formes — tenu (prédit 0,25-0,35).
* S1 (information) : qkv 0,23, out 0,33 — tenu (prédit FAUX sur out : ma prédiction de I_nu, 45-55 µs, reprenait le bras I
  non fusionné de la 255 ; fusionné, I_nu out = 30,6).
* **Copie xor = int16 au bit** (I = J sur 5 n × 4 formes) ; elle rend 34 µs par appel partagé (qkv) et **276 µs par appel
  hors portée** (qkv n = 624 : 472 → 196). Elle accélère aussi, au bit, le chemin cublas DÉJÀ servi des -qkvo-i8c.
* Justesse (x aléatoire, chemin cublas à n ≥ 128) : I 1,30-1,45 % contre T 0,97-1,13 % (prédit 1,29-1,45 : tenu).
* Contrôle de dispatch : n = 17 et n ≤ 80 hors portée → GEMV pour T comme pour I (± 0,3 %) ; chemins relevés : T_part
  `dequant`, I_part `cublas` + `i8c_fabrique`/`i8c_reutilise`.
* Réserve : dans `_part`, les 8 appels relisent le MÊME poids (L2 partiellement chaud) pour T comme pour I — biais commun, non
  celui du service (entre deux séquences d'une couche GDN, z et out passent) ; l'étape moteur tranche.

Décision (scellé, règle de chef) : S2 tenu → étape moteur, PPL, mini lm-eval, sans arrêt.
