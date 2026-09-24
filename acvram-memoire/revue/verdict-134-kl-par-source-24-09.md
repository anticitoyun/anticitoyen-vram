# Verdict — 134 : d'où vient la KL par pas de la disposition Marlin unique + GEMV v2 — 24/09 04 h 5x (poste1)

* **instrument** : `scratchpad/poste1-p134-24-09/kl-sources.py` — un processus, chemin par défaut chargé, `acvram.kernels.nvfp4_matmul` intercepté ; pour chaque poids éligible à la disposition (mêmes règles que `preparer_disposition_marlin`), la sortie Marlin est calculée SUR LA MÊME ENTRÉE par le vrai chemin de l'unique (`kernels._marlin_seul`, v2 TPB 1 S auto), disposition préparée à la volée puis libérée ; substitution par source : P (préfill, M ≥ 2 hors tête), D (décodage, M = 1 hors tête), H (tête), T (tout), A (rien). KL(A ‖ bras) en teacher forcing, 5 invites neuves de la 129, 8 pas.
* **commit** : c291a5fe (poste1-mtp)
* **régime** : Qwen3.8-27B-nvfp4, b=1, horloge libre (grandeur numérique) ; cpu-safe=off (100/100) ; compute-apps début = fin
* **scellé** : `scratchpad/poste1-p134-24-09/scelle.md` (avant la mesure) — contrôles A = 0 et T = B3 ± 30 % ; seuil 0,00491 inchangé ; « une source porte l'excès » = elle seule > 0,00491
* **mesuré** :

| bras | KL max | KL pas 0 (préfill) | KL pas 1-7 | argmax |
|---|---|---|---|---|
| A (contrôle) | **0** | 0 | 0 | 40/40 |
| P préfill Marlin | **0,005452** | 0,005452 | 0,000747 | 40/40 |
| D décodage v2 | 0,002377 | 0 | 0,002377 | 40/40 |
| H tête | 0 | 0 | 0 | 40/40 |
| T (contrôle) | **0,005452** | 0,005452 | 0,002269 | 40/40 |

  Erreur relative max par appel (bras T) : préfill 0,0067-0,0078 sur tous les rôles (down, gate‖up, GDN out/qkv/gate, attention) ; décodage 0,0018-0,0059 ; tête 0.
* **verdict** : contrôles **tenus** (A = 0 exactement, T = B3 au 10⁻⁶ : l'émulation reproduit l'unique au bit). **Une seule source porte l'excès : la GEMM Marlin au PRÉFILL** (P seul = 0,005452 > 0,00491, tout au pas 0) ; le décodage v2 seul (0,002377) et la tête (0) sont sous le seuil. Ma prédiction (P 0,003-0,005, « réparti probable ») a sous-estimé P et surestimé le partage.
* **correctif proposé** (code, opt-in, 3fada002) : dans la disposition unique, préfill (M > 32, régime bf16) = dépaquetage EXACT de la disposition Marlin (`depaqueter_marlin`, au bit de `nvfp4_dequant` ; échelle globale par colonne ajoutée au noyau Triton pour les piles, segment seul pour les vues de pile) puis F.linear, soit l'arithmétique du défaut. Le décodage (M ≤ 32) garde Marlin et v2 : aucune perte de vitesse au décodage ; au préfill, le coût du défaut (une déquantification + cuBLAS). Mesure du correctif : prise 134b (scellé en addendum, avant mesure).
* **durée** : 04:50:19 → 04:51:04

## Correctif mesuré — prise 134b (commit 3fada002, 04:54:12 → 04:55:54 ; addendum au scellé écrit avant) et 134c (4cf04e6e)
* Tests carte : **29 passed** (préfill **au bit** du défaut à M = 300 : poids seul, pile q/k/v à échelle par colonne, vues).
  Bras cassants : préfill rendu à Marlin **ROUGE** (« pas au bit du défaut ») ; échelle par colonne coupée **ROUGE** ;
  branche « segment seul » des vues : **VERTE à la 134b** — mon test ne la gardait pas (le repli pile entière puis découpe
  rend ici les mêmes bits ; la branche ne change que le coût) → test du compteur de chemin ajouté (4cf04e6e), bras
  **ROUGE** (« vue servie par la pile entière »), 7 passed.
* KL par source, correctif en place : A 0, **P 0**, D 0,002377, H 0, **T 0,002377**.
* **KL de l'unique corrigé réel** (disposition unique + v2 + correctif, `kl-chemins.py`) : **0,002377 ≤ 0,00491** (40/40).
* **PPL** (fenêtres de la 102) : A 4,0938, **B 4,0938** (identique — le préfill est au bit). Temps d'eval : A 16,69 s,
  B 17,45 s, **+4,6 %** (seuil scellé 5 %).
* **verdict** : correctif **TENU** sur les trois critères de l'addendum (PPL = 4,0938, KL ≤ 0,00491, eval ≤ +5 %). La
  disposition Marlin unique + v2 + dépaquetage au préfill est **qualifiée en KL** (0,002377, la seule part du décodage
  v2) sans seuil déplacé. Reste nommé : le préfill 4,6 % plus lent sur l'eval (dépaquetage Triton contre `nvfp4_dequant`
  CUDA) ; le décodage (b=1, b=8) n'est pas touché (M ≤ 32 inchangé) — l'ABBA de la 130 reste valable, à rejouer au
  besoin sur le code corrigé.
