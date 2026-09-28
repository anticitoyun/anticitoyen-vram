# Pièce 276 i — main non reproductible à b = 1 entre deux serveurs neufs : à sec, prédiction scellée AVANT la mesure

poste6, 27/09/2026 18 h 0x. Fait : 276 g cellule b = 1 (serveurs neufs, b = 1 SEULEMENT) : G = M 5/5 ; 276 h cellule b = 1 (mêmes
serveurs APRÈS les cellules b = 4 et b = 12, mêmes invites 0-4 que le b = 12) : H = M 9/10, et M1(276 h) ≠ M1(276 g) sur l'invite 1.
Relu : 213/213 b (poste5 26/09) — RoPE du premier lot construit en fp32 puis bf16 (`RotaryEmbedding` sans dtype), corrigé ; ce n'est
pas un défaut d'ordre de lot mais de dtype à la première forme : la 213 b le clôt, on ne le redécouvre pas ici.

## Prédiction (où je pense que ça diverge)
1. **Ce n'est pas un défaut de déterminisme des noyaux mais le CACHE DE PRÉFIXE** (`enable_prefix_cache` au défaut, runner.py:1268
   `match_prefix`) : l'invite k du b = 1 est EXACTEMENT celle du b = 12 qui l'a précédée sur le même serveur ; ses blocs KV de tête
   ont été calculés dans un préfill de 12 séquences (M = Σ jetons) et sont resservis à b = 1. Un GEMM cuBLAS bf16 choisit ses tuiles
   et son découpage selon M : à M différent, sommes dans un autre ordre, ± 1 ulp bf16 sur K/V — ce sont ces K/V que le décodage
   à b = 1 relit. Deux serveurs neufs qui ont vu des lots de composition différente (les vagues varient) ont des K/V différents.
2. Donc : **serveurs neufs, b = 1 seulement, sans trafic avant : 5/5 sur les trois configurations** (VL-2B image, VL-2B texte, Qwen3-4B
   texte) ; **après un b = 12 sur les mêmes invites : des bascules (1-2/5) dans les TROIS configurations** — le défaut est au MOTEUR
   (forme du lot), pas à la vision. Avec `--no-prefix-cache`, après b = 12 : 5/5 (le contrôle qui rend faux : si les bascules
   persistent sans cache, la cause est ailleurs — noyau non déterministe).
3. **Premier site de divergence** : la sortie de la première projection linéaire de la couche 0 (q/k/v, `nn.Linear` bf16 → cuBLAS),
   |Δ| = 1 ulp bf16 sur une fraction des éléments, entre « invite k seule (M = n_k) » et « invite k en tête d'un lot de 12 (M = Σ n) »,
   à entrée identique (embeddings au bit). Les couches suivantes amplifient. Pas de divergence attendue dans l'attention à entrée
   identique (varlen par séquence, réduction par ligne).

## Ce qui rendrait la prédiction fausse
* A1 ≠ A2 (serveurs neufs, b = 1 seul) sur une configuration → noyau non déterministe (atomiques, autotuning, capture) : à localiser
  par le diff de couches entre deux passages du MÊME lot.
* Bascules après b = 12 même avec `--no-prefix-cache` → pas le cache ; chercher l'état du serveur (graphes capturés, godets).
* Premier site ailleurs que le GEMM d'entrée (attention, norme) → nommé tel quel.

## Protocole (sous carte.sh, arbre main a6268af2e contrôlé)
Pour chaque configuration : deux serveurs neufs successifs ; sur chacun : A = invites 0-4 à b = 1 (24 jetons, temperature 0) ;
B = b = 12 (invites 0-11) ; C = invites 0-4 à b = 1 ; puis un troisième serveur neuf `--no-prefix-cache` : B puis C'. Compare
A1/A2, C1/C2, A/C, C'. Localisation : en process (`diverge-276i.py`) : mêmes prompt_ids, préfill de k seule contre k en tête d'un
lot de 12, crochets sur tous les modules : premier module (ordre d'exécution) dont la sortie diffère sur les lignes de k.
