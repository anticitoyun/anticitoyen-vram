# Verdict — 142, famille 32B qwen2/qwen3 (`DeepSeek-R1-Distill-Qwen-32B-srcQ4_K_M-nvfp4`, 2 alias) : disposition Marlin unique + v2 + 134 contre défaut — 24/09 09 h 5x (poste1)

* **instrument** : `scratchpad/poste1-p142-24-09/prise-famille.sh` — `kl-chemins.py` (T2 même prise), `acvram eval` (fenêtres de la 102), `certifie-b12.py` CERT_PUR, ABBA 5 lots par bras, -lgc 2700
* **commit** : kl, b1 0a6319c6 ; b8 final b514c59b (ligne « commit » de la prise ; worktree importé)
* **régime** : RTX 5090, cpu-safe=off (100/100) ; tous les lots NOMINAUX, 0/64 couche exilée ; B `+marlin(doubles=0,seuls=257)` ; dossier sur disque dur (chargements hors fenêtre)
* **scellé** : `scratchpad/poste1-p142-24-09/scelle-qwen32.md` + 2 addenda (avant les relances)
* **mesuré** :
  * **KL** : T2 0,000689 → seuil 0,001378 ; B **0,001218**, 40/40 → **TENUE** (marge 12 %).
  * **PPL** : A 6,9405, B 6,9408 → **+0,004 %** ≤ +0,5 % → **TENUE** (pas au bit, contrairement à Qwen3.8 : probablement les échelles minuscules que le repack Marlin annule, `depaqueter_marlin`, exception documentée). Eval B/A **+7,1 %** (15,14 → 16,21 s).
  * **b=1** (ctx 2 560) : A 14,054 ms, 71,15 t/s, 5,619 J ; B 13,674 ms, 73,13 t/s, 5,476 J → B/A **0,973** (prédit 0,98-1,02 : B plus rapide que prévu, côté favorable) ; J −2,5 %. A bridé par la puissance dès b=1 (400 W, SM ≈ 2 380).
  * **b=8** : à ctx 2 560, **B REFUSÉ au chargement** (capacité KV 14 256 < 20 480) alors que A passe. Mesuré à CERT_CTX=1 664, CIBLE_S=15 (deux relances : B épuisait ses pas avant la fin de la fenêtre, voir les addenda) : A **27,642 ms**, 289,4 t/s, 1,379 J (SM ≈ 2 045, bridé à 400 W) ; B **14,640 ms**, 546,5 t/s, 0,731 J (SM ≈ 2 641, 399 W) → B/A **0,5296** (prédit 0,62-0,75 : dépassé, côté favorable), débit **+88,8 %**, J/jeton **−47,0 %**.
* **verdict** : **TENU** (vitesse b=1/b=8, KL, PPL). Deux réserves à publier avec : (1) **la disposition réduit la capacité KV** sur ce modèle — 8 × 2 560 n'est plus servable avec PROJ_MARLIN (14 256 jetons max), alors que le défaut le sert ; (2) une partie du gain à b=8 vient du plafond de puissance (A à 2 045 MHz, B à 2 641 : même 400 W, moins d'énergie par octet), comme sur Qwen3.8 — c'est le gain dans le régime servi, pas à horloge égale.
* **durée** : kl → 09:0x ; b1 → 09:19:33 ; b8 final → 09:50:12 (chaque prise < 30 min)
