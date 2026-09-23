# Pièce 103 — lecture NInfer (clone superficiel github.com/Neroued/ninfer, Apache-2.0), à sec

Clone lu : clone superficiel de github.com/Neroued/ninfer du 23/09 (hors dépôt). Aucun code exécuté.

## 1. `src/ops/linear_topk` — lm_head + top-16 fondus, jamais de logits denses en HBM

`include/ninfer/ops/linear_topk.h:24-50` : projette `hidden [5120,U]` à travers la tête vocabulaire complète
(`[248320,5120]`, Q8 ou FP8) et rend directement les 16 meilleurs (score, id) par colonne — « no dense
vocabulary-logit tensor is an observable intermediate ». Une variante tête réduite 131072 lignes existe aussi
(l. 52-60).

**Chez nous** : `model.py:262-303` (`_tete`) calcule les logits COMPLETS (int8, nvfp4 ou fp32 selon le chemin),
puis `_logits_finaux` (l. 300+) masque le rembourrage, puis `runner.py:1887` (`_sample_only`) fait l'argmax/topk
séparément — deux passages HBM sur le tenseur de logits complet (écriture puis lecture), jamais fondus.

**Gain d'octets/pas, Qwen3-Coder-30B-A3B** (vocab 151 936, tête tenue en int8/fp32 sortie) : logits complets à
b=12 = 151 936 × 12 × 4 o (fp32) ≈ 7,29 Mo écrits + 7,29 Mo relus par le sampler ≈ **14,6 Mo/pas évités** (b=1 :
≈1,22 Mo/pas). Le calcul de la GEMM/GEMV de tête n'est PAS réduit (mêmes lectures de poids) — le gain est
uniquement le trafic HBM du tenseur intermédiaire de logits, pas la tête elle-même.

## 2. `src/ops/attn_input_proj` — Q/K/gate/V à partir de deux poids parents, aucun buffer empilé

`include/ninfer/ops/attn_input_proj.h:14-35` : un seul appel calcule q, gate, k, v depuis deux poids parents
(RowSplit), écrit directement les quatre destinations finales — « no packed parent output is materialized ».

**Chez nous** : `attention.py:110,147-166` empile déjà q/k/v (`self.qkv_proj`) en UN poids, une seule GEMM, puis
`torch.split` (l. 260) — sur un tenseur contigu selon la dernière dimension, `torch.split` est une VUE, pas une
copie : le trafic HBM est déjà équivalent à un fondu quatre-sorties pour q/k/v. **La différence réelle** : NInfer
fond aussi la porte (« gate ») dans le même appel à deux poids parents. Nous avons DÉJÀ testé cette extension
exacte — `attention.py:82-83` : « palier 2 (multi-projection q/k/v, qkv/gate/α/β) : témoin nommé, jamais défaut —
0,88 To/s pondéré au banc, qkv_multi 0,52 pas mieux que kv seule ». **Rien à retirer chez nous : la fusion plus
large a été mesurée et écartée, pas oubliée.**

## 3. `src/ops/sparse_moe` (+ `small_t`) — bloc MoE entier en un appel, geometry FERMÉE et incompatible Coder

`include/ninfer/ops/sparse_moe.h:48-58` : « Closed sparse-MoE Op for the exact future 35B-A3B geometry » — 256
experts routés, top-8, un expert partagé toujours actif, hidden 2048 : router + sélection + SwiGLU routé + SwiGLU
partagé + down + fusion + résidu, TOUT dans un seul appel. `small_t/sparse_moe_small_t_plan.cpp:23-33` confirme :
seuls les profils codec Q4/Q5/Q6-G64 (« main ») ou Q8-G32 (« mtp ») sont acceptés, sur cette même géométrie
figée — aucun paramètre de nombre d'experts.

**Incompatibilité de forme, nommée** : Qwen3-Coder-30B-A3B a **128 experts routés**, pas 256 (config.json :
`num_experts: 128`, `num_experts_per_tok: 8`, `hidden_size: 2048`). Le noyau tel qu'écrit ne s'applique pas à
Coder sans réécriture de la géométrie — **gain d'octets/pas non calculable en l'état**, à ne pas chiffrer pour ne
pas publier un nombre reconstruit (REGLES point 10).

**Chez nous** : `moe.py:1514-1579` fond déjà routeur + sélection top-k en un lancement Triton (`route_prep.route_fusee`)
distinct des GEMM expertes (`gemm_experts_tensor`, `moe.py:1939`) et de la fusion finale — plusieurs lancements,
pas un seul bloc MoE monolithique. Ce que NInfer ferait EN PLUS, à géométrie égale : fondre aussi les deux SwiGLU
(routé + partagé) et le résidu dans le même appel — mais Coder n'a pas d'expert partagé toujours actif dans notre
config actuelle à vérifier avant tout chiffrage (à nommer, pas mesuré ici).

## 4. Stockage KV `K8V4` (K FP8 row-scale, V NVFP4) — recoupe directement la Q(18) posée à duck.ai

`include/ninfer/ops/kv_cache_append.h:52-53` : « K8V4 uses the existing 256-byte row-scaled FP8 K code plus one
FP16 scale, and the 144-byte NVFP4 representation for V » — pour une ligne D=256 : K ≈ 258 o (1,008 o/élément),
V ≈ 144 o (0,5625 o/élément, blocs de 16 avec échelle UE4M3).

**Chez nous** (`kvcache.py:276-289`, `kv_canal.py:11-27`) : K ET V en int8 (`dtype="int8"` par défaut), K à
échelle PAR CANAL sur bloc de 16 jetons (128 o code + 8 o échelle amortie/jeton + 2 o `ks` demi ≈ 1,078 o/élément
pour D=128), V à échelle PAR JETON (128 o + 2 o ≈ 1,016 o/élément pour D=128) — symétrique, pas de K8V4.

**Gain d'octets/pas prédit, Qwen3-Coder-30B-A3B** (head_dim 128, num_key_value_heads 4, num_hidden_layers 48,
forme différente de NInfer D=256 — comparaison à l'élément, pas à la ligne) :
* K : 1,078 → 1,008 o/élément × 128 = 138,0 → 129,0 o/(jeton,tête,couche) — gain ≈ 9 o (6,5 %)
* V : 1,016 → 0,5625 o/élément × 128 = 130,0 → 72,0 o/(jeton,tête,couche) — gain ≈ 58 o (44,6 %)
* Total par (jeton, tête, couche) : 268,0 → 201,0 o, gain 67 o (25,0 %)
* Sur 4 têtes KV × 48 couches : 51 456 → 38 592 o/jeton, **gain ≈ 12,6 Kio/jeton (25 %) de trafic KV en écriture
  ET lecture** — c'est V (passage int8 → nvfp4) qui porte tout le gain, K est déjà proche de l'équivalent FP8.

## 5. Question duck.ai — coût de précision V int4 (group=32, échelle half) vs V int8, K int8 inchangé

Réponses complètes : `~/Bureau/Vibe/duck-reponses-23-09.md` § Pièce 103. Trois modèles en désaccord marqué :
**Luna** 1-3 % typique (0-1 % favorable, 3-5 % si couche sensible, cite KIVI arXiv 2402.02750 qui valide notre
choix K-par-canal / V-par-jeton) ; **gpt-oss** 3-7 % (≈4 % en pratique), répartit la cause granularité≈50 % /
clipping échelle fp16≈30 % / accumulation le reste ; **Gemma** < 1 %, argument que l'accumulation sur V est un
moyennage (pas de divergence exponentielle comme le softmax des scores), donc stable à 8192 jetons.

**Recoupement avec le § 5 de la pièce 104 de poste5** (scellé géo(k8v4/int8) + 2 SE ≤ +0,30 %, prédiction propre
de poste5 +0,10 à +0,25 %) : **aucun des trois modèles ne s'aligne sur ce seuil**. Luna et gpt-oss, s'ils ont
raison, réfutent le scellé avant toute mesure carte ; seul Gemma (< 1 %) est cohérent avec l'ordre de grandeur de
poste5, et son mécanisme (moyennage sur V) corrobore son raisonnement propre (« V par jeton portait un tiers du
coût int8, l'int4 par groupe en rajoute ≈2× sur cette moitié »). Aucun modèle ne connaissait le détail exact de
notre implémentation (bloc 16 pour K, groupe 32 pour V, pas de fenêtre résiduelle FP16) — le désaccord reflète de
la littérature générale, pas notre géométrie précise. **La mesure de la pièce 104 (KL b=1 5/5, PPL 3×12 fenêtres
à 8192+512) tranche, le vote entre modèles ne tranche pas** : si la PPL mesurée dépasse +0,30 %, ça confirme
Luna/gpt-oss et réfute le scellé tel quel (v1b min/max ou alternative B, comme prévu par poste5) ; si elle reste
sous ce seuil, ça confirme Gemma et le mécanisme de moyennage.

## Note de méthode

Aucune mesure carte, aucun code écrit. Les deux fusions (1) et (4) sont des leviers réels et chiffrés ; (2) est
déjà testée et écartée chez nous ; (3) ne s'applique pas à Coder sans adaptation de géométrie — nommé, non
chiffré. La question duck.ai (§5) est un recoupement de littérature, pas une preuve — le scellé de la pièce 104
reste seul juge.
