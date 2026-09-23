# Verdict — déquant/requant NVFP4 modelopt, bit à bit (16/09)

Océane, à sec, ordre Sage (`revue/sage-convertisseur-formats-16-09.md`,
main 1cf933b, § doute nommé + ordre point 1).

## Protocole

3 tenseurs réels d'un NVFP4 modelopt (`Qwen3.8-27B-NVFP4`,
`/mnt/4TO_SATACMR_2022/Modeles/models_vllm/`, `quant_method=modelopt`
confirmé) : `lm_head.weight`, `layers.0.mlp.gate_proj.weight`,
`layers.0.mlp.down_proj.weight`. Lus bruts (`weight` u8, `weight_scale`
e4m3 par bloc de 16, `weight_scale_2` f32 scalaire), déquantifiés par
`HFQuantCheckpoint._modelopt_nvfp4` (le chemin réel du convertisseur),
requantifiés par `quantize_nvfp4` sans échelle AWQ. Codes (`qweight`),
échelle de bloc (`block_scale`) et échelle globale (`global_scale`)
comparés bit à bit aux originaux. Script
`outils/verif-requant-nvfp4-bitabit-16-09.py`.

## Mesuré

| tenseur | codes différents | échelle de bloc différente | échelle globale |
|---|---|---|---|
| lm_head.weight | 45,5 % des octets | 42,4 % des blocs | **IDENTIQUE** |
| gate_proj (couche 0) | 51,7 % | 49,7 % | **IDENTIQUE** |
| down_proj (couche 0) | 51,5 % | 49,7 % | **IDENTIQUE** |

`global_scale` (le scalaire `amax_tenseur / (448×6)`, `nvfp4.py:237`)
tombe pile sur `weight_scale_2` — au bit — sur les trois tenseurs : la
formule de l'échelle GLOBALE est bien celle décrite (« amax/(448·6) »).

`block_scale` diverge sur ~50 % des blocs, et pas d'un ULP E4M3 (bruit
de rounding attendu) : écart relatif **médian 33 %, jusqu'à 84 %**
(exemples mesurés : 30↔20, 40↔26, 36↔24 — la nôtre systématiquement
PLUS PETITE). Les codes E2M1 suivent (médiane 1 niveau d'écart sur les
blocs qui divergent, cohérent avec une échelle elle-même fausse).

## VERDICT : DIFFÉRENT — pas du bruit, une échelle de bloc qui n'est pas
un simple `amax/6`

Le taux (~50 %) et l'amplitude (jusqu'à ×1,8) excluent un arrondi E4M3
différent (qui donnerait un écart de 1 ULP sur une minorité de blocs
proches d'une frontière, pas la moitié du tenseur à 30-80 %). La lecture
la plus probable : **modelopt calibre son échelle de bloc** (clipping
ou une statistique autre que le max brut — typique d'une quantification
qui protège contre les valeurs aberrantes), alors que
`quantize_nvfp4(no_awq)` prend le max brut du bloc **déquantifié**, qui
n'est pas nécessairement retrouvable sans les données de calibration
d'origine. `global_scale` (bornée par le max du TENSEUR entier, un seul
nombre) survit à ce ré-échantillonnage ; l'échelle par bloc, elle, encode
une décision propre à chaque groupe de 16 qu'une redéquantification ne
peut pas reconstituer.

## Conséquence

Le trou 1 (`sage-convertisseur-formats` § 3.1) **reste un vrai chantier** :
pas de raccourci `--no-awq` + garde manifeste. Laurine écrit le passage
direct — mapper `weight`/`weight_scale`/`weight_scale_2` vers
`NVFP4Tensor` SANS repasser par un dequant/requant, comme prévu au § 3.1
d'origine (~200 lignes). Réfutation prévue par Sage pour ce passage :
`dequantize_nvfp4` du tenseur mappé directement contre la déquantification
`hfquant.py` du même tenseur — égalité bit à bit (celle-ci DOIT réussir,
puisqu'aucun nouvel arrondi n'intervient), puis PPL acvram vs vLLM sur
le même checkpoint, écart ≤ 0,004.
