# Verdict — 142, famille gemma4 31B (`gemma-4-31B-it-nvfp4-vision`, 5 alias) : disposition Marlin unique + v2 + 134 contre défaut — 24/09 08 h 5x (poste1)

* **instrument** : `scratchpad/poste1-p142-24-09/prise-famille.sh` — `kl-chemins.py` (5 invites neuves 129, 8 pas, témoin T2 `ACVRAM_NVFP4_GEMV_MAX=0` même prise), `acvram eval` (fenêtres de la 102), `outils/gpu/mesure/certifie-b12.py` CERT_PUR, A B B A A B B A A B, -lgc 2700
* **commit** : kl 616c3cf9 ; b8 et b1 cf3138c8 (worktree importé : chemin absolu dans le journal de prise)
* **régime** : RTX 5090, cpu-safe=off (100/100) ; compute-apps début = fin (llama-server sur la 3080 Ti) ; A et B NOMINAUX, 0/60 couche exilée ; B `+marlin(doubles=0,seuls=307)`
* **scellé** : `scratchpad/poste1-p142-24-09/scelle-gemma31.md` + 3 addenda, tous avant les mesures qu'ils règlent
* **mesuré** :
  * **KL** : témoin T2 0,016473 → seuil **0,0329** ; B **0,007186**, 40/40 argmax → **TENUE**.
  * **PPL** : A = B = 334 991 (au bit) → **instrument invalide sur gemma4**, non mesuré. Le temps d'eval de B est **+69 %** (34,89 contre 20,66 s) : au préfill, le dépaquetage Triton de la 134 coûte cher sur ce modèle (tête 262 144 × 5 376, MLP 21 504). C'est un reste nommé.
  * **b=8** (CERT_CTX=448, CIBLE_S=2 ; ctx 2 560 non mesurable dans les deux bras, KV ≈ 3,8 k jetons) : A **26,94 ms**, 297,0 t/s, 1,327 J/jeton ; B **16,80 ms**, 476,3 t/s, 0,796 J/jeton ; lots ≤ 0,2 % d'écart. Pas B/A **0,6235** (prédit 0,62-0,75) ; débit **+60,4 %** ; J/jeton −40 % (indicatif, fenêtre de 2 s, drapeau « trop courte »).
  * **b=1** (rejouée : la 1re est contaminée par le sous-agent local du chef, 07:31-07:50) : A **15,842 ms**, 63,12 t/s, 6,318 J/jeton ; B **15,704 ms**, 63,68 t/s, 5,879 J/jeton. B/A **0,9913** (prédit 0,98-1,02) ; J/jeton **−7,0 %**.
* **verdict** : **TENU** en vitesse (b=8 +60 %, b=1 −0,9 % de pas) et en KL ; PPL non mesurée (instrument). Reste nommé : le préfill (+69 % à l'eval) — à mesurer en service (préfill d'une invite longue) avant toute décision « défaut » sur cette famille.
* **durée** : kl 07:04:22 → 07:06:28 ; b8 → 08:26:31 ; b1 → 08:51:15 (chacune < 30 min)
