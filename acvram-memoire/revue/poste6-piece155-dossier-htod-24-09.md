# Dossier à sec — pièce 155 : 361 copies hôte→carte paginées par préfill (70 ms, 27 % du TTFT de 512 sur gemma4 31B) — poste6, 24/09

Source : profil torch d'un préfill de 512 jetons en processus (`scratchpad/poste6-p147-24-09/profil-gemma-512-{A,B}.txt`, 11 h 56) :
`Memcpy HtoD (Pageable -> Device)` **n = 361, 70,9 ms** (A) / 361, 69,9 ms (B) — identique dans les deux bras, donc hors des
projections ; `aten::_to_copy` 551, `aten::to` 1 440 (A) / 2 897 (B, dont les `.to(dtype)` no-op du dépaquetage). 361 = 6 × 60 couches + 1.
Un `Memcpy HtoD` paginé = un tenseur créé sur l'hôte (liste Python → `torch.tensor(...)` sans `device=`) ou un `.to(device)` d'un
tenseur non épinglé ; il est SYNCHRONE (pageable), donc chaque copie coupe le flux : 361 coupures ≈ 70 ms.

## Sites examinés (fichier:ligne) et verdict à sec
| site | ce qu'il fait | par préfill | verdict |
|---|---|---|---|
| `engine/runner.py:1455-1471` (`_build_batch`) | `tokens`, `positions`, `slot_mapping` et `block_tables[i]` créés sur l'HÔTE par `torch.tensor(liste)` | 1 lot | source des tenseurs, copiés ensuite |
| `engine/lot.py:55-59` (`positions_on`) et `:117` (`slot_mapping_on`) | cache par carte : `self.positions.to(device, non_blocking=True)` UNE fois par lot ; `non_blocking` sur du paginé reste synchrone | 2 par lot | pas les 6 par couche |
| `engine/model.py:140` `batch.tokens.to(embed.device)` | 1 par lot | 1 | le « + 1 » |
| `engine/attention.py:496` `batch.block_tables[i].to(q.device)` | seulement si `offset > 0` (préfixe en cache) — `--no-prefix-cache` : non pris | 0 | non |
| `engine/attention.py:569` (idem, chemin `gather` de secours) | décodage sans noyau paginé | 0 au préfill | non |
| `engine/layers.py:705-728` (RoPE `_ensure`) | tables `cos/sin` reconstruites seulement si `seq_len > _cache_len` ou autre carte | 0 après chauffe | non |
| `quant/calibrate.py:186-200` (`ChannelScaler.apply`) | échelle convertie UNE fois (`_au_dtype`), tenseur sur la carte (`layers.py:458`) | 0 | non (c'était 337 noyaux `unrolled_elementwise` avant, déjà corrigé) |
| `engine/couches.py:544` `x * self.out_scale.to(x.dtype)` (Gemma) | conversion de dtype d'un tenseur carte | 0 HtoD | non |
| `engine/layers.py:1037` `torch.tensor(lens, device=device)` (`decode_attention_fixed` de secours) | liste → carte | décodage seulement | non |
| `engine/moe.py:193,271,617` `torch.tensor([...], device=...)` | MoE ; gemma 31B est dense | 0 | non (à revoir pour le Coder : 3 par couche MoE ?) |

**Aucun site examiné ne rend « 6 par couche » sur un modèle dense au préfill** : la lecture à sec ne suffit pas, et je ne veux
pas nommer une cause par élimination. Ce qui reste plausible sans preuve : les normes RMSNorm × 4 par couche Gemma
(`rmsnorm_bf16(x, self.weight, eps)` — `eps` est un flottant Python, pas une copie) ; les `q_norm`/`k_norm` par tête ;
l'attention SDPA (`_scaled_dot_product_efficient_attention`, 60 appels) et son masque/`attn_bias` — si `cu_seqlens` ou un biais
sont construits par liste à chaque appel dans `layers.py:983-1000`, ce sont des HtoD par couche (à vérifier : lignes 983-993
créent sur `q.device`, mais `decode_attention_fixed`/`prefill_attention` ont plusieurs chemins).

## Ce qui tranche (première prise de la 155, ≈ 1 min de carte, sans code moteur)
`torch.profiler` avec `with_stack=True` sur un préfill de 512 (Qwen3-4B-srcgguf-nvfp4, 36 couches, chargement 10 s : le motif
« 6 par couche » doit y donner 6 × 36 + 1 = 217 copies), `key_averages(group_by_stack_n=8)` filtré sur `Memcpy HtoD` → les
piles Python exactes (fichier:ligne) des 6 sites, puis la même chose sur gemma pour confirmer (60 s). Remède attendu : créer
ces tenseurs sur la carte (ou en mémoire épinglée, une copie asynchrone par lot), gain ≈ 70 ms par préfill de 512 (27 % du
TTFT) et ≈ 0,19 ms par copie évitée à toute longueur — **sur le DÉFAUT**, au bit (mêmes valeurs, autre emplacement).
Prédiction à sceller avant : TTFT A à 512 sur gemma 262 → ≤ 200 ms ; à 2 048 : 949 → ≤ 890 ; J/préfill −25 J (repos × 70 ms).
