# Verdict — `ACVRAM_DISABLE_KERNELS=1` rendait une PPL de 10⁸ : le jumeau torch de la pile d'experts lisait les échelles E4M3 comme des octets (17/09)

- **régime** : à sec ; reproduction sur GLM-4.7-Flash-vllm-direct chargé sur processeur, 2 couches,
  16 jetons, `ACVRAM_DISABLE_KERNELS=1` avec et sans `ACVRAM_DISABLE_CPU_KERNELS=1`.
- **scellé** (poste7) : reproduire, corriger, test qui casse si la faute revient ; inventaire des tests
  passés qui dépendaient de ce repli sur un converti en passage direct.
- **mesuré** : sur processeur, logits sains et identiques dans les deux régimes (max 13,3, argmax
  7265) — le processeur ne construit PAS les piles (experts exilés 64/64, repli boucle par expert),
  donc il ne reproduit pas. Lecture du chemin qui tourne sur carte sans extension :
  `MoEBlock._pile_bf16` (model.py:863-880) empile les `block_scale` des experts **en uint8**
  (`torch.stack([w.block_scale.view(torch.uint8) …])`, :771) et appelle `kernels.nvfp4_dequant` ;
  le noyau CUDA décode ces octets, mais sans extension le repli appelait `dequantize_nvfp4`
  (nvfp4.py) qui faisait `block_scale.to(float32)` : la VALEUR DE L'OCTET (0-255) à la place de
  l'E4M3 → poids ×100 → PPL 10⁸. Second défaut du même repli : déquantification en bf16 puis
  remultipliée par l'échelle de l'expert (double arrondi, 1 ulp d'écart avec le noyau).
- **verdict** : cause nommée, indépendante du passage direct (tout converti à piles y passait ;
  seul le régime « piles construites + noyaux coupés » le révélait — sur -k48 ou -vllm-direct
  pareil). Correctif (poste4) : `dequantize_nvfp4` voit les échelles uint8 comme E4M3 ; la pile
  garde la vue E4M3 ; le repli de `nvfp4_dequant` avec `gscale_rows` passe par
  `global_scale_rows` de la référence (échelle de ligne × échelle de bloc en fp32, un seul
  arrondi) = l'arithmétique du noyau. Tests `tests/test_repli_torch_pile_nvfp4.py` : octets
  décodés comme E4M3 (témoin de faute qui diverge), repli sur une pile à 3 experts == référence
  expert par expert au bit (extension absente par monkeypatch).
- **inventaire** (`git log -S passage_direct -- tests`, `-S DISABLE_KERNELS -- tests`) : aucun test
  passé ne dépendait de ce repli sur un converti en passage direct — le format a deux jours
  (70b5279, 8164622) et ses tests ne coupent pas les noyaux. Prédiction de poste7 tenue.
- **reste** : rejouer la PPL sous `ACVRAM_DISABLE_KERNELS=1` sur carte (poste3) : attendue égale à
  la PPL noyaux au bruit près (le repli est maintenant le jumeau).
