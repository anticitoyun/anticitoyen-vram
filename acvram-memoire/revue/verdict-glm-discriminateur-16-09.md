# Verdict — discriminateur précision vs noyau, GLM MMA=0/1 (16/09)

poste1, à sec (aucun GPU, `CUDA_VISIBLE_DEVICES=""`), ordre poste7
(`revue/poste7-glm-mma0-verdict-16-09.md` §3, main f48c0f7).

## Protocole

`acvram.quant.fakequant_activation.fake_quantize_nvfp4_activation` +
`acvram.quant.nvfp4.{quantize_nvfp4,dequantize_nvfp4}`, PyTorch pur, aucun
noyau CUDA. Couche MoE 1 (`first_k_dense_replace=1`), 8 experts
(0-7), `gate_proj`.

- **Poids** : bf16 réels de `GLM-4.7-Flash-bf16` (source, non quantifiés).
- **Échelle AWQ** : table `[E,K]` réelle lue dans le converti alpha-commun
  `GLM-4.7-Flash-srcbf16-nvfp4` (`model.layers.1.mlp.experts.{e}.gate_proj.weight.act_scale`),
  pas une échelle synthétique.
- **Activations** : forward réel (CPU, mini-répertoire 2 couches converti en
  bf16 pur, `--no-awq --quant-device cpu --host-exec cpu`, même méthode que
  `outils/equivalence-glm-2couches.py`), 16 jetons synthétiques (mêmes que
  l'équivalence GLM du 15/09), hook sur l'entrée de `MoEBlock` — x réel
  `[16, 2048]`, pas une sonde diagonale.
- **Quatre bras**, `err = ‖y − y_ref‖ / ‖y_ref‖` (norme poolée sur les 8
  experts, `y_ref = x_fp32 @ w_fp32.T`) :
  - (i) W4A16 sans échelle
  - (ii) W4A16 avec échelle (`w·s` fake-quant NVFP4, `x/s`)
  - (iii) W4A4 sans échelle (`x` fake-quant NVFP4 directement)
  - (iv) W4A4 avec échelle (`x/s` fake-quant NVFP4)

## Mesuré

| bras | err |
|---|---|
| (i) W4A16 sans échelle | 0,093196 |
| (ii) W4A16 avec échelle | 0,097602 |
| (iii) W4A4 sans échelle | 0,131145 |
| (iv) W4A4 avec échelle | 0,139417 |

**err(iv)/err(iii) = 1,0631**

Détail par expert dans `/tmp/glm-discriminateur-mma0/resultat.json` (script
`outils/discriminateur-glm-mma0-16-09.py`) — hétérogène (expert 3 : 1,215 ;
expert 2 : 1,000, échelle quasi identité ; expert 5 : 1,152), pas un artefact
d'un seul expert extrême.

## VERDICT : ≤ 1,1 → **(1a) application**

Seuil scellé franchi sans zone grise (1,0631, pas de recours au témoin ulp).
L'arithmétique exacte (fake-quant, aucun noyau) dit que l'échelle AWQ
alpha-commun **n'aggrave presque pas** l'erreur en W4A4 (+6,3 % relatif, du
même ordre que son effet en W4A16 : +4,7 % en (ii) contre (i)) — l'hypothèse
(1b) de poste7 (« l'échelle élargit l'étendue intra-bloc, le noyau est
innocent ») est **écartée par la mesure**. La pile réelle (`model.py:983`,
division `xs / awq["gate_proj"][e_sorted]`, ou l'ordre `route+pack`) fait
autre chose que ce calcul exact.

## Conséquence

Pas de correction du noyau/de la métrique de calibration. Le bogue est dans
l'application de l'échelle sur le chemin MMA réel — poste4, comme prévu par
poste7 pour le cas (1a) : correctif sur `_forward_prefill_grouped`/MMA (piste
déjà notée dans `poste1-equivalence-pile-gateup-16-09.md` : coupure nette à
la position 8/préfixe 9 jetons, frontière `_MOE_DECODE_MMA_MIN_T`), test
d'équivalence pile/boucle **avec la table réelle du converti** dans le même
commit, re-PPL du même converti scellé ≤ 1,010 (réfuté > 1,015). poste2
attend (§5 de la note de poste7).
