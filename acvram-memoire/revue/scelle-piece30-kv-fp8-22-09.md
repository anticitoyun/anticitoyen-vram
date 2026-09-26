# Scellé — pièce 30 : cellule TRT-LLM vs acvram à KV égalisé fp8 (poste3, 22/09, avant mesure)

Après le test b=1 et la KL relative. But : refaire la cellule A B B A A B avec le
cache KV **égalisé en fp8 des deux côtés**, pour retirer le facteur KV de l'écart
b=12 (décomposition poste1 : « KV FP8 0,1-0,2 »). Publiée À CÔTÉ de la cellule KV
natif (int8 acvram / auto trtllm), **jamais confondue** avec elle.

## Réglages (vérifiés dans le code servi)

- **acvram KV fp8** : `ACVRAM_KV_FORMAT=fp8_e4m3` (opt-in). Source :
  `acvram/memory/tiering.py:48` (`_KV_FORMAT = os.environ.get("ACVRAM_KV_FORMAT","")`),
  validé `tiering.py:50` (attendu int8 | fp8_e4m3 | fp16 | bf16 | lm4 | lm3 | lm2) ;
  le type par défaut est `int8` (`kvcache.py:276`). La ligne de régime imprime
  `kv=<format>` (vu `kv=int8` au run de calibration) → **contrôle : la cellule kv-fp8
  n'est valide que si le régime imprime `kv=fp8_e4m3`** (sinon refus, cellule nulle).
- **trtllm KV fp8** : `KvCacheConfig(dtype='fp8')` explicite (au lieu de 'auto').
  À passer à `trtllm-serve` (flag kv cache dtype) ; vérifier dans les LLM Args du log
  que `kv_cache_config … dtype='fp8'`.

## Prédiction (avant mesure)

- Débit b=12 : à KV fp8 des deux côtés, l'écart trtllm−acvram se resserre un peu
  (la part « KV FP8 0,1-0,2 » de la décomposition disparaît) mais reste en faveur de
  trtllm (l'essentiel vient de Marlin/étroites/service). Prédit : écart b=12 **+15 à
  +20 %** (contre +22 % en KV natif). acvram fp8 : débit b=12 à ±3 % de son int8.
- b=1 : inchangé sur le fond (l'effondrement trtllm b=1 ne vient pas du KV — GPU au
  plafond en calcul, cf. scelle-b1) ; prédit trtllm b=1 toujours ≈ 46 t/s.
- KL relative en fp8 (kl-api, des deux côtés en fp8) : proche du natif (le format KV
  pèse peu sur la distribution de sortie à 8 pas) ; prédit KL max fp8 ≈ KL natif ± 0,2.

Cellule kv-fp8 étiquetée distinctement dans le TSV (kv=fp8 des deux côtés), verdict
séparé, jamais agrégée avec la cellule KV natif.
