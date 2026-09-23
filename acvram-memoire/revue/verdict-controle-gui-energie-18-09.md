# Verdict — contrôle GUI `energie.j_par_jeton` (/metrics, Jérôme 4b83a2f) contre `energie.py` sur la même fenêtre : **0,3794 contre 0,3801 J/jeton, écart −0,18 %** → **TENU** (± 10 %)

instrument : `scratchpad/controle-gui-18-09/{chaine.sh,controle.py,controle.json}` — serveur `acvram.cli serve Qwen3-Coder-30B-A3B-nvfp4 --port 8094 --max-batch 12` (v0.6.11) sous carte.sh ; chauffe 12 × 256 ; fenêtre : `GET /metrics` → `Energie()` → 6 salves × 12 requêtes × 256 jetons → `GET /metrics` dans le `with` ; dénominateur commun = Δ `engine.decode_tokens`
commit : ce03a74 (laure = main 4b83a2f), 06:57, régime classé, rpw=4 défaut, graphes on
régime : b=12, 72 requêtes, 0 échec, 16,8 s, Δ decode_tokens 18 360 (usage sommé 18 432 : −0,4 %, la chauffe/les EOS ne biaisent pas le dénominateur), 1 092 t/s sous 400 W ; deux cartes comptées des deux côtés (carte 1 au repos, 210 MHz)
scellé (Sage) : |GUI / energie.py − 1| ≤ 0,10 ; prédiction < 3 %
mesuré : GUI **0,3794** J/jeton · energie.py 6 978 J / 18 360 = **0,3801** J/jeton (413 W moyens deux cartes, 16,88 s, invalidation « bridage : puissance » attendue) · écart **−0,0018**
verdict : **TENU** — même compteur NVML (`TotalEnergyConsumption`) lu aux mêmes bornes, même dénominateur ; l'ajout GUI rend le J/jeton de la fenêtre à 0,2 % près, sous la prédiction 3 %

## Lecture
- Le champ est une fenêtre « depuis le dernier appel /metrics » : la console qui le lit à intervalle régulier verra le J/jeton de son propre intervalle ; deux lecteurs concurrents se volent la fenêtre (l'un lit un delta court, l'autre n'a pas de delta) — à documenter dans la console, pas un défaut de mesure.
- La fenêtre de 16,8 s a suffi (écart 0,2 % pour une résolution NVML de 1 mJ et des salves régulières) ; sur une fenêtre où le décodage s'arrête (delta_tok = 0) le champ rend `null`, conforme.
- Deux cartes sommées des deux côtés : le J/jeton publié inclut le repos de la carte 1 (~15-20 W) ; pour la cellule menu c'est le même biais qu'`energie.py` (déjà le cas de toutes les cellules J).
