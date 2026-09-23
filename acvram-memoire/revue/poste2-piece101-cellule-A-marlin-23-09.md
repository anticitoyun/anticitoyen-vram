# Pièce 101 — cellule HTTP alias A, Marlin dense (X/Z) contre vLLM (Y) — INFORMATION, alias non servable (poste2, 23/09)

**Alias non servable** (chef, ordre reçu avant la prise) : A (`Qwen3-Coder-30B-A3B-nvfp4-qkv-alphaqkv-23-09`)
n'est pas qualifié en lot mêlé (100 B, FAUX définitif). Cette cellule ne sert PAS à qualifier un
défaut ; elle informe si les projections nvfp4 + Marlin dense ferment l'écart d'énergie avec vLLM.

instrument : `scratchpad/poste2-piece101-cellule-A-23-09/bloc.sh <b>`, ABBA X (`ACVRAM_PROJ_MARLIN=1`)
  / Z (`ACVRAM_PROJ_MARLIN=0`, témoin) / Y (vllm serve `Qwen3-Coder-30B-A3B-Instruct-FP4-a16`,
  TRITON_ATTN), `--speculative none`, `-lgc 2700` posé et relevé des deux côtés, 5 répétitions du
  palier par bras (méthode 89)
commit : a6c75acc (main, worktree poste2-w-21-09)
régime : -lgc 2700, horloge moy proche de la bande habituelle (à relire dans les logs bras par bras
  si besoin — RAS au journal carte.sh, aucune invalidation)
scellé : `scratchpad/poste2-piece101-cellule-A-23-09/scelle.md` (avant la prise) ; écart déclaré si
  |diff| > 2σ_diff (Welch, n=10/bras = 2×5) ; preuves de configuration : `.so` Marlin chargé (fait
  hors fenêtre, `marlin_gemm` présent) ; compte-appels AVANT séance = 96 (`compte-avant.log`) ;
  **compte-appels APRÈS séance en attente de la carte** (poste6 la tient au moment d'écrire ce
  verdict) — à faire dès qu'elle la rend, sinon la cellule est nulle par la règle même de la
  consigne
mesuré :
  b=1 — X(marlin1) n=10 moy 332,0 t/s σ 0,48 ; Z(marlin0) n=10 moy 332,0 σ 0,32 ; diff −0,03 %,
  seuil 2σ 0,37 → **égalité** (prédit : identique — TENU). J/jeton : X 0,4990 ; Z 0,5028 ; diff
  −0,74 %, seuil 2σ 0,0042 → **égalité**. Contre vLLM (284,7 t/s) : X devant de **+16,62 %** (AU-DELÀ)
  et **−16,27 %** de J/jeton (AU-DELÀ, X moins cher).
  b=12 — X(marlin1) n=10 moy 1 613,4 t/s σ 23,8 ; Z(marlin0) n=10 moy 1 544,4 σ 25,1 ; diff **+4,47 %**
  seuil 2σ 21,9 → **AU-DELÀ, Marlin dense devant** (prédit −2 à −4 % de ms/pas, soit ≈ +2 à +4 % de
  t/s — TENU, légèrement au-dessus de la fourchette haute). J/jeton : X 0,1960 ; Z 0,2044 ; diff
  **−4,09 %**, seuil 2σ 0,0025 → **AU-DELÀ** (prédit −2 à −5 % — TENU). Contre vLLM (1 945,1 t/s) :
  X **derrière de −17,05 %** (AU-DELÀ) et **+56,19 %** de J/jeton (AU-DELÀ, vLLM bien moins cher).
verdict : le Marlin dense (X/Z) tient sa prédiction aux deux critères scellés — à b=12, débit
  +4,47 % (seuil X ≥ Z+1,5 % en ms/pas : **TENU**) et J/jeton −4,09 %, à b=1 égalité comme prédit
  (M=1 ne change pas de chemin). **Mais l'alias A lui-même reste loin derrière vLLM à b=12**
  (−17 %, énergie +56 %) : le gain Marlin (+4,5 %) ne ferme pas l'écart d'énergie avec vLLM sur cet
  alias — un meilleur alias (qualifié) resterait à mesurer pour juger si Marlin suffit ailleurs.
  À b=1, l'alias A DEVANCE vLLM (comme le Coder par défaut, cohérent avec pièces 78/89/94 ter) —
  ce sens-là n'a rien à voir avec le Marlin (égalité X/Z), c'est le régime b=1 général.
suite : compte-appels après séance dès que la carte se libère (poste6 → moi selon la file) ; sinon
  cellule à annoter « nulle » et refaire. Ma file ensuite : pièce 115 (balayage d'horloge SM),
  après poste1 (MTP), puis pièce 102 (NInfer).
