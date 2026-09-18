# Verdict — in situ RPW ABAB, Coder b=12 : rpw=4 **9,51 ms/pas nu contre 10,76 (× 0,884)** → **TENU** (seuil ≤ 0,95) ; J/jeton × 0,892 tenu ; bridé 1 034 → 1 162 t/s (+12 %)

instrument : `scratchpad/gemv-experts-rpw-18-09/situ-abab.sh` — ABAB `ACVRAM_GROUPED_RPW=1/4/1/4`, chaque bras `certifie-b12-15-09.py` (bridé 400 W, J/jeton brut, `regime_ligne`+`engine_regime` dans le JSON) puis `profil-pas-coder-17-09.py … b12` (40 pas nus, `ms_par_pas_nu`) ; `abab-{cert,profil}-rpw*-p*.json` ; modèle `Qwen3-Coder-30B-A3B-nvfp4`
commit : f38ef72 (poste3 = main d2670cb, poste4 a850937), 06:45-06:51, une seule fenêtre
régime : classé, graphes on, b=12 ; ligne [régime] : `GROUPED_RPW=4` présente dans les JSON des bras B (rpw=1 est le défaut : absent de la ligne `hors_defaut` des bras A, voir `regime.variables`)
scellé (poste7, poste7-rpw-in-situ-gui-18-09) : seuil unique — moyenne B (ms/pas nu) ≤ 0,95 × moyenne A → tenu ; J/jeton second juge, même seuil, informatif ; ma prédiction −4 à −6 %
mesuré : nu A **10,734 / 10,782** ms (1 118 / 1 113 t/s) · B **9,479 / 9,537** ms (1 266 / 1 258 t/s) → **B/A = 0,884** ; J/jeton A 0,3846 / 0,3854 · B 0,3430 / 0,3441 → **0,892** ; bridé (398-399 W) A 11,58 / 11,62 ms 1 036 / 1 033 t/s · B 10,32 / 10,35 ms 1 163 / 1 160 t/s ; les deux paires ABAB concordent à 0,5 %
verdict : **TENU** — −11,6 % de ms/pas nu et −10,8 % de J/jeton, sous le seuil 0,95 avec 6,6 points de marge ; ma prédiction (−4 à −6 %) réfutée en mieux : le gain du banc (−11 %) se retrouve entier dans le pas ; bit-exact déjà tenu (verdict-gemv-experts-rpw-18-09 : ppl-decode-kv 1 024 NLL identiques)

## Lecture
- Le pas Coder b=12 nu passe de 10,76 à 9,51 ms : les 1,25 ms gagnés valent plus que les 0,87 ms/pas prédits par le banc à 48 couches synthétiques — en situ la GEMV experts pèse davantage (échelles réelles, 48 couches MoE Coder avec I=768 réel) et la bande libérée profite aussi aux noyaux voisins.
- Règle 9 remplie : équivalence au bit, mécanisme compris (activation étagée une fois par bloc de GW_WARPS×RPW lignes), gain mesuré en situ nu et en J/jeton ; le passage de `GROUPED_RPW` à 4 en défaut relève de poste7/poste4.
- Cellule Coder b=12 à publier (poste8) après décision de défaut : nu 1 262 t/s / bridé 1 162 t/s 0,344 J/jeton (contre 1 116 / 1 034 / 0,385).
