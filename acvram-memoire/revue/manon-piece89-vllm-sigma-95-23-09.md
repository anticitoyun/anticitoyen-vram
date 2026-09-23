# Pièce 89 — comparatif vLLM à ≥5 lots/bras, ABBA, σ + IC95 %, seuil 2σ (Manon, 23/09)

instrument : `scratchpad/manon-p78-23-09/run.sh` (X = acvram serve défaut, Y = vllm serve
  --attention-backend TRITON_ATTN, -lgc 2700 explicite autour de chaque bras Y), adapté en
  `scratchpad/manon-p89-23-09/bloc.sh <b> <ABBA|ABAB>` avec BANC_SLOTS=b,b,b,b,b (5 répétitions du
  palier, méthode 87 bis) au lieu d'une fenêtre unique ; deux blocs séparés sous carte.sh (b=12 puis
  b=1), chacun < 13 min, hors plafond 30 min
commit : 653416aa (les deux blocs)
régime : -lgc 2700 ; b=12 bridage puissance nommé sur les DEUX moteurs (horloge 2664-2672, même bande) ;
  b=1 aucun bridage (horloge 2674-2678, même bande) ; compute-apps propre avant/après chaque bloc
scellé : écart déclaré seulement si |moyenne_acvram − moyenne_vllm| > 2·σ_diff (Welch, n=10 par moteur =
  2 occurrences × 5) ; sinon égalité ; moyenne/σ/IC95%/J-jeton publiés dans tous les cas
  (`scratchpad/manon-p89-23-09/scelle.md`, avant la prise)
mesuré :
  b=12 — acvram n=10 moy 1 931,2 t/s σ 52,3 IC95[1 898,8 ; 1 963,6] ; vllm n=10 moy 2 026,5 σ 54,5
  IC95[1 992,7 ; 2 060,3] ; diff = −95,3 t/s (−4,70 %), σ_diff 23,9, seuil 2σ = 47,8 → **au-delà**.
  J/jeton net : acvram 0,1376 (σ 0,0045) ; vllm 0,1288 (σ 0,0045) ; diff +0,0088 (+6,82 %), seuil
  2σ 0,0040 → **au-delà**.
  b=1 — acvram n=10 moy 312,3 t/s σ 0,1 IC95[312,3 ; 312,4] ; vllm n=10 moy 284,8 σ 0,3
  IC95[284,6 ; 285,0] ; diff = +27,5 t/s (+9,67 %), σ_diff 0,09, seuil 2σ 0,18 → **au-delà**.
  J/jeton net : acvram 0,6035 (σ 0,0025) ; vllm 0,6032 (σ 0,0018) ; diff +0,0002 (+0,04 %), seuil
  2σ 0,0019 → **sous le seuil, égalité**.
verdict : les DEUX écarts de vitesse (b=12 vLLM devant, b=1 acvram devant) passent le seuil 2σ —
  confirmés, à publier au README. L'énergie à b=12 (acvram plus cher) passe aussi le seuil — confirmée.
  L'énergie à b=1 est une ÉGALITÉ (diff sous 2σ) — ne PAS republier comme « acvram devant en énergie »,
  seulement « à égalité ». Cohérent en signe et en grandeur avec la pièce 78 (b=12 −4,87 % puis −4,70 % ;
  b=1 +9,39 % puis +9,67 % ; énergie b=1 déjà à égalité en 78, confirmée ici avec un seuil statistique
  explicite cette fois). Contrôle Laure (commit cb7c4147, 12h05:52) : tombe dans la fenêtre du PREMIER
  essai b=12 (12h01-12h13), essai perdu par un bug de nommage de fichiers (voir Incident) et donc DÉJÀ
  écarté sans lien avec la contamination ; le bloc b=12 retenu (12h24-12h36) et le bloc b=1 (12h13-12h23)
  ont tous leurs `load1_max` ≤ 1,52 (aucun `processus_charges` étranger nommé) — aucun lot contaminé dans
  les données publiées ici.
incident (à consigner, sans conséquence sur ce verdict) : la première exécution du bloc b=12 (12h01-12h13)
  a écrasé ses propres fichiers `banc-*.log`/`serveur-*.log` par ceux du bloc b=1 lancé juste après, faute
  de `$B` dans les noms de fichiers de `bloc.sh` — corrigé avant le second essai (`banc-b$B-$nom.log`),
  b=12 entièrement rejoué proprement ; aucune donnée publiée ici ne vient de l'essai écrasé.
durée : bloc b=12 (écrasé, écarté) 719 s ; bloc b=1 646 s ; bloc b=12 (redo, retenu) 709 s ; total
  carte 2 074 s sur trois prises séparées, chacune bien sous 30 min (journal `tenue=`)
