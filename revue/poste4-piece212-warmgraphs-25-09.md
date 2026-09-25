# Pièce 212 — coût VRAM de `warm_graphs`, hors de la marge fixe (poste4, 25/09)

instrument : `scratchpad/poste4-p212-25-09/mesure-warmgraphs.py` (copie de `capacite201.py`, poste5) +
`prise-warmgraphs.sh`
commit : 3cd287381 (poste4-212 = origin/main + poste5-201 fusionnés)
régime : B=8, CTX=2048, eco=2700
scellé : —
mesuré : 5 modèles, un processus chacun
verdict : 3/5 modèles DÉPASSENT la marge fixe (1 536 Mio) — pas seulement l'i8c
durée : 5 min (5 chargements + warm_graphs)

## (1) Fichier:ligne — ce que `warm_graphs` alloue et pourquoi ce n'est pas compté

`warm_graphs` (`acvram/engine/graphes.py:40`) capture, pour chaque longueur `L` (128, 256, 512, 1024,
2048 — puissances de deux jusqu'à `max_len`) et chaque forme de lot, un graphe CUDA via `_capture`
(`acvram/engine/graphs.py:983`). Deux coûts distincts s'y ajoutent :
* les tampons PAR GODET (`entry["x"]`, `positions`, `slots`, `tables`, `seq_lens`, `sortie` —
  `graphs.py:987-994`), petits individuellement mais multipliés par le nombre de formes capturées ;
* le **pool mémoire du graphe CUDA** (`graphs.py:1075-1086`, `torch.cuda.CUDAGraph()` puis
  `self._pool = graph.pool()` au premier appel, réutilisé ensuite) : CUDA y range TOUTE allocation
  transitoire rencontrée PENDANT la capture (workspace d'attention, de routage MoE, etc.), pour CHAQUE
  forme distincte — et ce pool ne se libère jamais tant que les graphes vivent (nécessaire au replay).

Non compté parce que `_marge_carte`/`_reserve_prefill` (`loader.py:1173`/`1889`, et le correctif 201 de
poste5) raisonnent sur le PIC d'UN forward transitoire AVANT le chargement — jamais sur la croissance
CUMULATIVE, PERSISTANTE du pool de graphes qui se produit APRÈS le chargement, pendant `warm_graphs`
(appelé par le serveur juste après `load_model`/`Engine.__init__`, hors du chemin que `_reserve_prefill`
couvre). Seule la marge fixe `_KV_MARGE_MIN` (`loader.py`, 1 536 Mio) l'absorbe, par accident, pas par
conception.

## (2) Mesure par modèle (B=8, CTX=2048)

| modèle | libre après `Engine` | libre après `warm_graphs` | coût | > marge (1 536 Mio) |
|---|---|---|---|---|
| Qwen3.8-27B-nvfp4 | 13 013 Mio | 11 133 Mio | **1 880 Mio** | **OUI** (+344) |
| Qwen3.8-27B-unsloth-mixte-i8c | 8 903 Mio | 6 169 Mio | **2 734 Mio** | **OUI** (+1 198) |
| Qwen3.8-27B-nvfp4-attn-gdn-i8c | 9 167 Mio | 7 447 Mio | **1 720 Mio** | **OUI** (+184) |
| gemma-4-31B-it-nvfp4-vision | 3 679 Mio | 4 547 Mio | −868 Mio (libère) | non |
| Qwen3-Coder-30B-A3B-nvfp4 | 11 767 Mio | 11 501 Mio | 266 Mio | non |

**Corrélation nette** : les trois modèles qui dépassent sont TOUS les trois variantes Qwen3.8 (couches
GDN/hybrides) ; les deux qui ne dépassent pas (gemma31, Coder-30B) n'ont pas de couche à récurrence
linéaire. `_capture` snapshotte/restaure explicitement l'état des `hybrid_layers` (`graphs.py`, juste
avant la capture) — cohérent avec un pool qui grossit plus sur ces modèles. `mixte` (2 734 Mio, le pire)
et `i8c` (1 720 Mio) sont tous deux des Qwen3.8 attn+GDN promus — cohérent avec un travail
transitoire int8/mixte plus lourd par forme capturée en plus du coût GDN de base. gemma31 négatif
(libère 868 Mio) n'est pas expliqué ici — pourrait être une libération différée d'un tampon de
chargement, à creuser séparément si utile.

## (3) Proposition — compter le terme, sur le modèle de la 201

Impossible de le calculer analytiquement à l'architecture seule (c'est un pic de pool CUDA, pas une
taille de tenseur) — contrairement aux termes de la 153/201 (poids). Deux options, dans l'ordre de
préférence :

**a) Mesurer une fois par famille, coder la borne comme poste5 a codé `poids_bf16_couche_lineaire_bytes`
pour la 172** : `_KV_MARGE_MIN` (`loader.py`, actuellement 1 536 Mio fixe) devient
`max(1536, 2816) * 2**20` quand le modèle a des couches `linear_attention` (GDN/KDA/mamba2/lfm2 — même
test que `spec.layer_types` utilisé ailleurs) : 2 816 Mio couvre le pire mesuré (2 734) avec ~3 % de
marge, sans toucher les modèles sans GDN (gemma31, Coder, futurs modèles denses). Le point faible : une
constante calibrée sur CES CINQ modèles, pas dérivée d'une formule — à recalibrer si un GDN plus gros
(plus de couches hybrides, ou un `max_concurrent_seqs` plus élevé que B=8) apparaît.

**b) Mesurer en direct** : appeler `torch.cuda.mem_get_info()` avant/après `warm_graphs` et RÉDUIRE le KV
budget déjà accordé (`engine.stats.kv_blocks_total`) après coup si le résultat mord la marge — plus
exact, mais change l'ordre (KV puis graphes puis KV à nouveau) et risque le même problème que la pièce
153c : si le KV est déjà à son plancher (`_kv_plancher`), il n'y a rien à réduire, il faudrait alors
refuser le chargement ou exiler — je n'ai pas d'ordre pour toucher ce couplage (confié à poste5, 201).

Je recommande (a) : plus simple, sans toucher au couplage réserve→exil que poste5 vient de stabiliser,
corrige le cas mesuré aujourd'hui. (b) reste la vraie solution si un modèle futur dépasse aussi (a).
