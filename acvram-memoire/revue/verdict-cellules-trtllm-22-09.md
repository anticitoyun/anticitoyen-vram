# Verdict — cellules acvram vs TensorRT-LLM (poste3, 22/09)

Qwen3-Coder-30B-A3B, RTX 5090, décodage soutenu (1024 jetons/séquence, fenêtre
20 s), même client que la cellule officielle (`banc-llamacpp-16-09.py decode`,
invites en ids, `/v1/completions` SSE, `ignore_eos`). Fenêtres intercalées
A B B A A B ; 5/6 valides (3 acvram, 2 trtllm ; 3B ratée — trtllm n'a pas rouvert
son port au rechargement consécutif B→B). Client validé par calibration acvram
seul le même jour (1 615,8 / 390,7, cf. poste3-chaine-trtllm-reparee-22-09.md).

## Les 7 lignes

1. **b=12 : trtllm 1 997,9 t/s (fp8) > acvram 1 638,1 t/s (int8)** — trtllm devant de **+22 %** (2 fenêtres trtllm 1987,6/2008,1 ; 3 acvram 1634,4/1638,1/1685,5, médianes).
2. **b=1 : acvram 390,5 t/s (int8) ≫ trtllm 46,3 t/s (fp8)** — acvram devant de **×8,4** ; trtllm s'effondre à b=1 (2 fenêtres 46,3/46,4, j/jeton 7,18 contre 0,44 acvram).
3. **b=1 trtllm = 46 est reproductible, pas un artefact** : deux fenêtres concordantes ICI + le run charge.py de poste2 (05 h) avait déjà donné 46 — deux harnais indépendants, et le même client donne 390 sur acvram, donc la mesure est juste ; c'est un comportement réel de trtllm-serve à b=1 (à expliquer côté config trtllm, hors de cette cellule).
4. **Scellé poste3 réfuté des deux côtés** : b=1 420 [380-470] → 46 (très en dessous) ; b=12 1 700 [1 550-1 850] → 1 998 (au-dessus). Publié tel quel (mesure reproductible).
5. **Note de comparabilité — KV** : acvram **INT8**, trtllm **FP8** (KvCacheConfig dtype ; trtllm n'offre pas INT8). Écart de format, pas neutre sur le débit et la qualité — NON corrigé, nommé.
6. **Note de comparabilité — régime commun** : graphes CUDA actifs des DEUX côtés (acvram sampler=graphe ; trtllm cuda_graph_config batch 1..32,64,128). Exclusions de quantification : acvram NVFP4 exclut MoE gates + lm_head ; trtllm charge le point de contrôle FP4-hub (exclusions propres au paquet, non re-vérifiées ici). Contexte : acvram --max-model-len 2304 ; trtllm max_input_len 1024 / max_num_tokens 8192 — l'invite (256) + 1024 décodés tient des deux côtés.
7. **Lecture** : trtllm gagne le débit agrégé à forte concurrence (b=12, +22 %), acvram gagne franchement la latence à requête unique (b=1, ×8,4) — les deux moteurs occupent des régimes opposés ; aucun ne domine l'autre sur les deux points. b=1 trtllm à confirmer/expliquer avant tout usage décisionnel.

## Données (cellules.tsv)

| moteur | kv | b=1 t/s | b=12 t/s | fenêtres |
|---|---|---|---|---|
| acvram | int8 | 390,6 / 390,2 / 390,5 | 1 634,4 / 1 638,1 / 1 685,5 | 1A 4A 5A |
| trtllm | fp8 | 46,3 / 46,4 | 1 987,6 / 2 008,1 | 2B 6B |

Toutes fenêtres `fenetre_valide=true`. Horloge b=12 : acvram ~2440 MHz, trtllm
~2946 MHz. 3B ratée (rechargement trtllm consécutif) — 5/6, comparatif conservé.
