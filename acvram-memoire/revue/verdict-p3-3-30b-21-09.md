# P3 (3) Qwen3-VL-30B nvfp4 vs AWQ déquantifié — 21/09

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-3.sh` + `references-transformers-dequant.py` (fichiers suivis, fusion 73a67474/poste1-p3-dequant)
* commit : 0133df83 (worktree poste2, fusion origin/main dont poste1-p3-dequant)
* régime : jamais atteint (échec avant la partie acvram/carte)
* scellé : haut_2se_pct ≤ 3 % (poste7 20/09, `poste7-ordres-restants-menus-20-09`)
* mesuré : rien — la référence transformers bf16 déquantifiée n'a pas pu se construire
* verdict : **ÉCHEC, rien à mesurer** — `references-transformers-dequant.py:25` : `AutoModelForImageTextToText.from_pretrained(..., device_map="auto", max_memory={0:…, "cpu":…})` charge l'AWQ (compressed-tensors) avec offload partiel, puis `hf_quantizer.postprocess_model()` appelle `compressed_tensors...unpack_from_int32` qui lit `shape[1].item()` sur un tenseur encore sur le device `meta` (offloadé, jamais matérialisé) → `RuntimeError: Tensor.item() cannot be called on meta tensors`. Marche à sec sur le 2B (pas d'offload, tout tient en mémoire) ; casse au 30B dès que `device_map="auto"` répartit un module sur `meta`. Défaut du script de référence, pas du moteur acvram ni de la carte — carte prise 108 s (10:25:04-10:26:52), rendue propre, nvidia-smi après = seul le service permanent (4286).
* durée : 108 s, prévu ≤ 1800 s

## Suite (à poste1, auteure du script)
Le décompresseur compressed-tensors doit tourner avant l'offload accelerate (ou avec `low_cpu_mem_usage=False`/sans `device_map="auto"` sur les modules compressés), ou charger d'abord en CPU plein puis déplacer. Hors mon domaine de mesure.
