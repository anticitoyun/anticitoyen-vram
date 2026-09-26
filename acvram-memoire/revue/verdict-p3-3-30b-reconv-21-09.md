# P3 (3) Qwen3-VL-30B reconverti (correctif experts groupés, carte visible) — 21/09 (poste2)

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-3.sh` (worktree poste2-w-21-09), référence `ref-30b-bf16.pt` réutilisée (sha256 08733794711e, non recalculée)
* commit : 02a68005 ; référence = `Qwen3-VL-30B-A3B-awq-dequant-bf16` (AWQ déquantifié, biais nommé au script : aucun Qwen3-VL-30B bf16 officiel sur disque)
* régime : `ctx_tenu=non-chauffe deepstack=3 eco=2700(2692) graphes=on(hybrides≤4) kv=int8 masque_images=causal mrope=[24,20,20](interleaved) vision=bf16(eager,transformers=5.17.0)`
* scellé (poste7 20/09) : « non établi pire que 3 % à 2 SE » — tenu ssi `haut_2se_pct` ≤ 3 %. Issue nommée d'avance (H1 poste1) : « > 3 % = plafond de la source AWQ, à écrire, pas à diagnostiquer »
* mesuré : 3 images, `greedy8_identiques=0/3`, `premier_jeton_identique=2/3`, `ppl_desc_geo_pct=-14,42 %`, **`haut_2se_pct=26,095 %`**
* verdict : **RÉFUTÉ, nettement** (26,1 % contre seuil 3 %, ×8,7) — et **PIRE que l'alias cassé du 20/09** (21,8 %, `verdict-p3-3-30b-665eeacc-21-09`) malgré le format nvfp4 désormais correct (`plan.tiers[].weight_format=nvfp4` confirmé) : cohérent avec l'alarme calibration — `experts_sans_stats=18432/18432` (100 % des tenseurs d'experts en repli échelle identité, aucune statistique AWQ collectée, `cli.py:654` « statistiques relevees pour 0 tenseurs », toutes couches en échec `setStorage`). La cause n'est donc PAS le plafond de la source AWQ nommé d'avance, mais l'absence totale d'AWQ sur ce converti précis — à distinguer.
* durée : 20 s (18:29:55–18:30:15), `nvidia-smi` propre après

## Suite
Pièce (1) de la chef close : reconversion réussie côté format (nvfp4, garde convert.py:2280 corrigée) mais la qualité reste bloquée par l'échec de calibration (0 tenseur avec stats, `acvram/quant/collect.py`/`setStorage`) — hors mon domaine, à nommer par poste1 avant tout nouveau chiffre de qualité sur cet alias. P3 (4) TENU (seuils H1), P3 (3) RÉFUTÉ. Pointeur transmis à la chef.
