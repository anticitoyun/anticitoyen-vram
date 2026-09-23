# Verdict — cellules acvram vs TensorRT-LLM (Laure, 22/09)

Qwen3-Coder-30B-A3B, RTX 5090, décodage soutenu (1024 jetons/séquence, fenêtre
20 s), même client que la cellule officielle (`banc-llamacpp-16-09.py decode`,
invites en ids, `/v1/completions` SSE, `ignore_eos`). Fenêtres intercalées
A B B A A B ; 5/6 valides (3 acvram, 2 trtllm ; 3B ratée — trtllm n'a pas rouvert
son port au rechargement consécutif B→B). Client validé par calibration acvram
seul le même jour (1 615,8 / 390,7, cf. laure-chaine-trtllm-reparee-22-09.md).

## Les 7 lignes

1. **b=12 : trtllm 1 997,9 t/s (fp8) > acvram 1 638,1 t/s (int8)** — trtllm devant de **+22 %** (2 fenêtres trtllm 1987,6/2008,1 ; 3 acvram 1634,4/1638,1/1685,5, médianes).
2. **b=1 : acvram 390,5 t/s (int8) ≫ trtllm 46,3 t/s (fp8)** — acvram devant de **×8,4** ; trtllm s'effondre à b=1 (2 fenêtres 46,3/46,4, j/jeton 7,18 contre 0,44 acvram).
3. **b=1 trtllm = 46 est reproductible, pas un artefact** : deux fenêtres concordantes ICI + le run charge.py de Manon (05 h) avait déjà donné 46 — deux harnais indépendants, et le même client donne 390 sur acvram, donc la mesure est juste ; c'est un comportement réel de trtllm-serve à b=1 (à expliquer côté config trtllm, hors de cette cellule).
4. **Scellé Laure réfuté des deux côtés** : b=1 420 [380-470] → 46 (très en dessous) ; b=12 1 700 [1 550-1 850] → 1 998 (au-dessus). Publié tel quel (mesure reproductible).
5. **Note de comparabilité — KV** : acvram **INT8** ; trtllm **`dtype='auto'`** (LLM Args / KvCacheConfig — PAS fp8 confirmé ; l'étiquette « fp8 » du premier jet était erronée, 'auto' est résolu par trtllm, format exact non tranché). Écart de format KV, non neutre — nommé, non corrigé.
6. **Note de comparabilité — régime commun** : graphes CUDA actifs des DEUX côtés (acvram sampler=graphe ; trtllm cuda_graph_config batch 1..32,64,128). Exclusions de quantification : acvram NVFP4 exclut MoE gates + lm_head ; trtllm charge le point de contrôle FP4-hub (exclusions propres au paquet, non re-vérifiées ici). Contexte : acvram --max-model-len 2304 ; trtllm max_input_len 1024 / max_num_tokens 8192 — l'invite (256) + 1024 décodés tient des deux côtés.
7. **Lecture** : trtllm gagne le débit agrégé à forte concurrence (b=12, +22 %), acvram gagne franchement la latence à requête unique (b=1, ×8,4) — les deux moteurs occupent des régimes opposés ; aucun ne domine l'autre sur les deux points. b=1 trtllm à confirmer/expliquer avant tout usage décisionnel.

**Comparabilité b=12 (relecture Laurine)** : horloge acvram ~2456 MHz vs trtllm ~2946 MHz (+20 %), mais la PUISSANCE médiane par fenêtre départage — acvram 390,1 W, trtllm 378,2 W, plafond 400 W : **les deux bras sont au plafond de puissance**, donc l'écart d'horloge = densité par cycle différente (acvram plus dense : NVFP4 + KV INT8), pas un avantage de budget de trtllm. Le débit au plafond est la mesure juste, et le J/jeton net confirme (trtllm 0,155 < acvram 0,196) : trtllm devant à b=12 en débit ET en énergie. Cellule b=12 comparable, publiée avec W et MHz.

**b=1 — indice de puissance** : trtllm 399,8 W (AU PLAFOND) pour 46 t/s ; acvram 241 W pour 390 t/s. Le GPU trtllm est saturé en puissance à b=1 → calcul massif par jeton, PAS un overhead hôte/streaming (qui laisserait le GPU à basse conso). La cause b=1 est côté calcul GPU trtllm à faible batch (cf. scellé b=1 dans laure-chaine-trtllm-reparee-22-09.md).

## Données (cellules.tsv)

| moteur | kv | b=1 t/s | b=12 t/s | W b=1 | W b=12 | MHz b=12 |
|---|---|---|---|---|---|---|
| acvram | int8 | 390,6 / 390,2 / 390,5 | 1 634,4 / 1 638,1 / 1 685,5 | 241 | 390,1 | ~2456 |
| trtllm | auto | 46,3 / 46,4 | 1 987,6 / 2 008,1 | 399,8 | 378,2 | ~2946 |

Toutes fenêtres `fenetre_valide=true`. Plafond 400 W. J/jeton net b=12 : acvram
0,196, trtllm 0,155. 3B ratée (rechargement trtllm consécutif) — 5/6, comparatif
conservé.
