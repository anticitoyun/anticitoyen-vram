# Pièce 96 — rejeu vLLM b=12 (pièce 89) aux nouveaux défauts (Manon, 23/09, ordre Jérôme)

instrument : `scratchpad/manon-piece96-vllm-rejeu-23-09/bloc.sh 12 ABBA`, copie exacte du script de
  la 89 (acvram serve défaut / vllm serve --attention-backend TRITON_ATTN, -lgc 2700 explicite autour
  de chaque bras vLLM), 5 répétitions du palier par bras, aucune variable forcée (code de main tel
  quel : réduction déroulée en défaut depuis la 92, 8 warps confirmé devant à b=12 par la 95 — pas de
  changement de défaut, pas de `4 warps` à retenir ici)
commit : c8da314c (main, worktree manon-w-21-09)
régime : -lgc 2700 posé par le moteur (acvram) et explicitement autour de vLLM ; compute-apps propre
scellé : écart déclaré si |moyenne_acvram − moyenne_vllm| > 2·σ_diff (Welch, n=10/moteur = 2×5),
  sinon égalité (`scelle.md`, avant la prise)
mesuré : acvram n=10 moy 1 995,1 t/s σ 59,8 ; vllm n=10 moy 2 027,0 σ 34,3 ; diff −31,9 (−1,58 %),
  seuil 2σ 43,6 → **sous le seuil, égalité** (pièce 89 : −4,70 %, au-delà). J/jeton net : acvram
  0,1349 (σ 0,0052) ; vllm 0,1261 (σ 0,0046) ; diff +0,0088 (+6,96 %), seuil 2σ 0,0044 → **AU-DELÀ,
  vllm moins cher** (pièce 89 : +6,82 %, deja au-delà — inchangé).
verdict : la réduction déroulée (pièce 92, −1,9 % au bit) a fermé l'écart de VITESSE b=12 contre
  vLLM (−4,70 % → −1,58 %, passe sous le seuil statistique) — **PORTE VITESSE TENUE** (à égalité,
  plus de retard significatif). L'écart d'ÉNERGIE reste ouvert et significatif (+6,96 %, quasi
  identique à la 89) : la réduction déroulée gagne du temps de calcul mais pas de puissance en
  proportion — piste distincte de l'attention (bridage/horloge/hôte), pas tranchée ici.
suite : Jérôme — écart d'attention résiduel contre vLLM (piste noyau à tuiles, pièce 92 §5) reste
  d'intérêt pour l'ÉNERGIE, plus pour la vitesse à b=12 ; ma file : sur ordre.
