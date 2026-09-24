# Verdict — 104 repli (1) : puits d'attention gardés en int8 (P = 16), reste du KV en k8v4 — 24/09 01 h 1x (poste1)

* **instrument** : `outils/gpu/mesure/ppl-decode-kv.py` (128 tranches `scratchpad/corpus-prive/tranches-128`, préfixe 8 192 + 512 notés), `scratchpad/poste1-p104s5-23-09/kl-b.py` (KL_B=1, 5 dumps HF), `outils/gpu/mesure/frontiere-pas.py` ; prises `prise-puits-1.sh` et `prise-puits-2.sh`
* **commit** : KL 217145c8 ; PPL et frontière 0d7bdf28 (poste1-mtp ; code des puits de poste5 b6118755 fusionné)
* **régime** : Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c ; témoin `ACVRAM_KV_FORMAT=int8`, candidat `k8v4` + `ACVRAM_KV_PUITS=16` — la ligne de régime imprime `kv=k8v4+puits16` dans chaque bras candidat (config prise). **cpu-safe=off** : la prise KL (00:50:37 → 00:54:24) chevauche le bridage processeur (00:51:59 → 00:54:35) ; ce sont des valeurs numériques, donc valides (chef). PPL (00:54:50 →) et frontière : max_perf_pct 100 au début et à la fin.
* **scellé** : `scratchpad/poste1-p104s5-23-09/scelle-puits.md` (commit 8b09381d, écrit avant le code et la mesure) : n = 128 fixé ; géo + 2 SE ≤ +0,30 % ; KL b=1 ≤ 0,74 et |ΔKL| ≤ 0,10 ; frontière ≥ 0,4 ms de gain
* **mesuré** :
  * PPL : n = **128**, géo(puits16/int8) **+0,335 %**, SE 0,071 %, **géo + 2 SE = +0,477 %** ; 79/128 tranches positives, médiane +0,287 %, étendue −1,76 à +2,89 % ; σ par tranche 0,80 % (prévu 0,83 : la résolution scellée est tenue, 2 SE = 0,142 %).
  * KL b=1 : int8 et puits16 kl_max global 0,5193, 5/5 ≤ 0,74 ; ΔKL par invite 0 / 0 / 0 / **+0,0352** / 0 (k8v4 seul : +0,1173 en prise A).
  * Frontière (ctx 8 704, invite 8 192, ACVRAM_BUDGET_JETONS=2048, -lgc 2700) : rc 0 dans les deux bras mais **0/300 pas à lot plein** (lots vus : 4 en int8, 3 en puits16) — le budget KV de 12 × 8 704 ne tient pas à côté de llama-server (5,6 Gio résidents) ; aucun temps de pas à b=12.
* **verdict** :
  * **PPL : FAUX** (+0,477 > +0,30), à résolution suffisante. La prédiction (géo +0,05 à +0,15 %) est réfutée : garder les 16 puits en int8 ne rapproche pas k8v4 de int8. L'écart n'est pas porté par les puits. Le niveau ne se compare pas à la prise B (+0,211 %, autres textes).
  * KL b=1 : tenue (ΔKL +0,035 ≤ 0,10) ; les puits corrigent l'invite 3, pas la PPL longue.
  * Frontière : **non mesurée** (lot jamais plein) — la mesure à ctx 8 k, b=12 demande la carte sans llama-server, ou un contexte plus court ; décision au chef.
* **durée** : prise 1 (KL) 00:50:37 → 00:54:24 ; prise 1 bis (PPL int8) 00:54:50 → 01:00:35 ; prise 2 (PPL puits16 + frontière) → 01:11:18 ; toutes < 30 min, compute-apps début = fin (llama-server 4627 seul). Première prise 1 : PPL int8 arrêtée à la tranche 1 (LISEZMOI.txt pris par `--dossier`, motif corrigé), aucune donnée.

Suite scellée (scelle-puits.md, issues) : repli (2) **V avec point zéro**, sous un nouveau scellé écrit avant le code. Ce repli est d'autant mieux indiqué que les puits ne portent pas l'écart : K étant en 8 bits dans les deux bras, il vient vraisemblablement du V 4 bits sur tout le contexte (inférence, non mesurée).

**ERRATUM 24/09 02 h 3x (poste1)** : llama-server (PID 4627) tourne sur la RTX 3080 Ti (GPU 1), pas sur la 5090 (`nvidia-smi --query-compute-apps=gpu_uuid`). L'OOM et le lot jamais plein à ctx 8 k viennent de NOTRE processus (31,32 Gio sur la 5090 : poids + KV de 12 × 8 704 + activations du préfill), pas d'un voisin. La consigne « ne pas toucher au llama-server » reste juste, mais sa cause était fausse : la frontière à 8 k, b=12 ne tient pas sur 32 Go avec ce plan mémoire, llama-server ou non.
