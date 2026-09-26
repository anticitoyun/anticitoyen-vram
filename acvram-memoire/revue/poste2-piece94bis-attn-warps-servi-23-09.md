# Pièce 94 bis — attention 4 vs 8 warps, servi ABAB, ≥5 lots/bras, seuil 2σ (poste2, 23/09, ordre chef)

instrument : `scratchpad/poste2-piece95-attn-warps-23-09/bloc.sh <b> <ctxmax> <jetons>`, ABAB
  (X = acvram serve défaut `ATTN_WARPS_COMPACT=8`, Y = `ACVRAM_ATTN_WARPS_COMPACT=4`), même
  moteur/modèle (Qwen3-Coder-30B-A3B-nvfp4), 5 répétitions du palier par bras (méthode 89),
  ctx long : b=1 max-model-len 4352 / 4096 jetons décodés (ctx final ≈ 4352) ; b=12 max-model-len
  2048 / 1536 jetons (ctx final ≈ 1792, allégé après un premier bloc en TIMEOUT à 4096 jetons)
commit : d79f08d5 (main, worktree poste2-w-21-09)
régime : -lgc 2700 posé par le moteur (horloge moy 2657-2692) ; compute-apps propre hors llama-server (idle, hors mesure)
scellé : écart déclaré si |moyenne_X − moyenne_Y| > 2·σ_diff (Welch, n=10/bras = 2×5), sinon égalité
  (`scelle.md`, avant la prise)
mesuré :
  b=12 (ctx ≈1792) — défaut8 n=10 moy 1 950,2 t/s σ 82,9 ; warps4 n=10 moy 1 867,8 σ 98,1 ; diff
  +82,5 (+4,41 % pour défaut8), seuil 2σ 81,2 → **AU-DELÀ, défaut8 devant**. J/jeton : défaut8
  0,1412 (σ 0,0065) ; warps4 0,1468 (σ 0,0083) ; diff −0,0056, seuil 2σ 0,0067 → **sous le seuil,
  égalité**.
  b=1 (ctx ≈4352) — défaut8 n=10 moy 293,5 t/s σ 0,2 ; warps4 n=10 moy 299,0 σ 0,1 ; diff −5,5
  (−1,85 % pour défaut8, donc warps4 devant), seuil 2σ 0,17 → **AU-DELÀ, warps4 devant**. J/jeton :
  défaut8 0,6299 (σ 0,0037) ; warps4 0,6259 (σ 0,0029) ; diff +0,0040, seuil 2σ 0,0030 → **AU-DELÀ,
  warps4 moins cher aussi**.
verdict : confirme le désaccord banc/servi déjà noté par poste1 (pièce 92) — **b=12 servi : 8 warps
  reste devant** (+4,4 % vitesse, énergie égale), aucun changement de défaut justifié ; **b=1 servi
  ctx long : 4 warps devant** (+1,85 % vitesse, énergie −0,6 % aussi en sa faveur), les deux écarts
  passant le seuil 2σ. Décision réservée à poste2 par la pièce 92 : **je ne bascule pas le défaut**
  — 8 warps sert la charge b=12 qui domine le profil, gagner b=1 en coûterait sur b=12 sans variable
  de régime par godet pour découpler les deux (pas construite ici). Écart restant : sans objet pour
  trancher un nouveau noyau d'attention à b=12 (8 warps déjà devant) ; à b=1 l'écart (+1,85 %) est
  trop petit pour justifier seul un nouveau noyau.
incident : premier bloc b=12 à ctx 4352/4096 jetons — TIMEOUT carte.sh à 1800 s après 2/4 bras
  (X1, Y1 seulement) ; rejoué entièrement à ctx 2048/1536 jetons, les 4 bras du bloc allégé retenus
  ici, aucune donnée de l'essai écarté publiée.
durée : bloc b=12 (écarté, timeout) 1800 s ; bloc b=1 (retenu) 663 s ; bloc b=12 allégé (retenu) 833 s

suite : ma file — (b) rejeu vLLM b=12 de la pièce 89 aux nouveaux défauts (réduction déroulée +
  4 warps inchangé) ; chef : écart d'attention restant contre vLLM sans objet à b=12 tant que 8
  warps est devant en servi.
