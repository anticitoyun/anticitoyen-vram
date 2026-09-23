# Verdict — M2, S2 parc sur le .deb 0.6.31 installé (Sage sage-tests-rapides-cloture-20-09 § 2 M2) : **TENU — 16/16 alias OK, 0 repli, 0 échec** ; graphes capturés partout (gemma compris : la prédiction « ≥ 1 repli gemma b=1 » ne se réalise pas sur 0.6.31), 0 couche / 0 expert exilé ; `inactive_split_bytes` après piles publié par alias (chiffre, pas seuil) : **Coder i8c 8,42 Gio (7,34 en petit pool : les tables i8c par canal fragmentent le petit pool)**, GLM-Flash 1,8-3,6, Grande 2,6, Kimi 1,8, les autres 0,3-1,5

instrument : `outils/gpu/mesure/non-regression-parc.py` réécrit (manon 40d4518f : par alias, régime servi lot 12 / ctx 2 304 / graphes, un jeton décodé au godet 1, `torch.cuda.memory_stats` après piles, verdict par rc 0/2/1 OK/REPLI/ECHEC, racine `racine_modeles()`, aucun chemin en dur, aucun texte de modèle sur la sortie — REGLES § 6) + `scratchpad/m2-parc-20-09/chaine.sh` (nue, un `carte.sh` par alias, `ACVRAM_CPUS=0-15`, code mesuré = `/usr/share/acvram` du .deb sous le venv du lanceur `~/.local/share/acvram/venv`, instrument depuis l'arbre) ; 16 alias de `outils/poste/alias-servis-20-09.txt` sous `/mnt/AI_GENERATOR/models_acvram` (repli `~/.config/acvram/modeles`, la chaîne ayant effacé `ACVRAM_MODELES` par son `unset`) ; 11:27:41-11:40:30, `poste=20-09-1030` ; **premier lancement 11:26 annulé** : l'instrument insérait l'arbre en tête de `sys.path` et mesurait `manon/acvram` au lieu du .deb (en-tête `acvram=` le montrait) → `sys.path.append`, en-tête relu hors de l'arbre ; premier alias 195 s = compilation de l'extension du .deb (une fois), les autres 9-90 s
scellé (Sage, avant) : tout alias OK ou repli ANNONCÉ ; 0 échec silencieux ; inactive_split publié par alias ; un alias OK le 20/09 07 h 10 qui échoue = régression du 0.6.31
mesuré (alias | verdict | inactive_split Gio | petit | grand | réservé | alloué | s) :
| DeepSeek-Coder-V2-Lite-Instruct-nvfp4 | OK | 1.382 | 0.100 | 1.282 | 10.04 | 8.66 | 24 |
| GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA | OK | 2.747 | 1.292 | 1.455 | 20.00 | 17.25 | 11 |
| GLM-4.7-Flash-srcbf16-nvfp4-k48 | OK | 1.809 | 1.297 | 0.513 | 19.08 | 17.27 | 38 |
| GLM-4.7-Flash-srcbf16-nvfp4 | OK | 3.550 | 1.309 | 2.241 | 20.80 | 17.25 | 40 |
| GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4 | OK | 2.577 | 0.188 | 2.388 | 25.50 | 22.93 | 49 |
| Jan-v2-VL-max-srcQ4_K_M-nvfp4 | OK | 1.064 | 0.435 | 0.629 | 18.89 | 17.83 | 45 |
| Kimi-Linear-35B-kda-nvfp4 | OK | 1.789 | 0.268 | 1.520 | 21.79 | 19.75 | 87 |
| LFM2.5-8B-A1B-nvfp4 | OK | 0.605 | 0.150 | 0.455 | 6.04 | 5.44 | 9 |
| Nemotron-3.5-Lightning-30B-A3B-srcexl3_6bpw-nvfp4 | OK | 0.614 | 0.075 | 0.540 | 18.09 | 17.23 | 37 |
| Ornith-1.0-35B-kimi-nvfp4 | OK | 1.005 | 0.227 | 0.778 | 21.28 | 20.02 | 49 |
| Qwen2.5-Coder-14B-pur-nvfp4 | OK | 0.275 | 0.003 | 0.271 | 11.15 | 10.69 | 10 |
| Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c | OK | 8.415 | 7.339 | 1.076 | 26.88 | 18.46 | 193 |
| Qwen3-Coder-30B-A3B-nvfp4 | OK | 1.543 | 0.467 | 1.076 | 19.89 | 18.35 | 46 |
| Qwen3.5-35B-A3B-srcQ4_K_M-nvfp4 | OK | 0.984 | 0.193 | 0.791 | 20.20 | 19.21 | 28 |
| Qwen3.8-27B-nvfp4-calibA | OK | 0.343 | 0.011 | 0.332 | 17.93 | 16.90 | 34 |
| gemma-4-26B-A4B-heretic-APEX-I-Quality-nvfp4 | OK | 0.917 | 0.222 | 0.694 | 17.64 | 16.72 | 36 |
verdict : aucune régression du parc sur 0.6.31 ; le petit pool inactif de 7,3 Gio de l'i8c est le seul chiffre hors du lot (réservé 26,9 pour 18,5 alloués : 8,4 Gio de VRAM tenus sans usage après un seul jeton) — à nommer avant qu'un lot 12 long ne le rencontre (S2 : `PYTORCH_CUDA_ALLOC_CONF` ou `empty_cache` après piles comme 0.6.29 l'a fait pour gemma)
durée : 12,8 min de carte (11:27:41-11:40:30) + 1,3 min du lancement annulé
suite : Sage : catalogue marqué 0.6.31 ; Océane : i8c petit pool 7,3 Gio (mesure sous lot 12 × 1 024 avant de conclure) ; Jérôme : indexer, fusionner l'instrument ; ma file : M3 2a-bis (6661bb96) dès « trou fini » → M4 pièce 3 (6643f243)

## Rejouable
`cd <worktree manon> && POSTE=20-09-1030 ACVRAM_CPUS=0-15 ACVRAM_ARBRE=$PWD bash scratchpad/m2-parc-20-09/chaine.sh` (13 min avec l'extension du .deb en cache ; sorties `scratchpad/m2-parc-20-09/<alias>.json`).
