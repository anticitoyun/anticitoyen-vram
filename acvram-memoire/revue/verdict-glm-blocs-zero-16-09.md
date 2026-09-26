# Verdict — contrôle blocs-à-zéro (plancher dénormal E4M3), GLM (16/09)

poste1, à sec (aucun GPU), ordre poste7 (main 86b152b,
`revue/poste7-glm-pile-correctif-16-09.md`), suite au discriminateur
(`revue/verdict-glm-discriminateur-16-09.md`) qui utilisait à tort
`quantize_nvfp4`/`dequantize_nvfp4` (poids, échelle globale, `nvfp4.py:237`)
pour fake-quantifier l'activation — le noyau réel `nvfp4_quant_act_kernel`
(`acvram/kernels/acvram_kernels.cu:2021-2050`) n'a AUCUNE échelle globale :
`s = amax/6` arrondi en E4M3, et si `s < 2⁻⁹` (plancher dénormal E4M3),
`sdec == 0.f` et **le bloc de 16 entier est écrit à zéro**
(`if (sdec > 0.f)` garde l'empaquetage, ligne 2042).

## Protocole

Mêmes 8 experts, même `x` réel (16 positions, `[16, 2048]`, capturé à
l'entrée de la couche MoE 1) que le discriminateur précédent. Table AWQ
réelle de `up_proj` (converti alpha-commun `GLM-4.7-Flash-srcbf16-nvfp4`).
Fraction de blocs de 16 dont `amax < 6×2⁻⁹ = 0,01171875` (zéro forcé), sur
`x` brut contre `x / s_up[e]` (poolé sur les 8 experts, 16384 blocs).

## Mesuré

- fraction(x brut) = **0,000000** (0/2048)
- fraction(x/s_up, poolée) = **0,000000** (0/16384) — nulle pour chacun des
  8 experts individuellement
- `s_up` observées : min 0,053 (expert 3) à max 2,111 (expert 4) — la
  plupart des canaux DIVISENT par une échelle proche de 1-2 (donc réduisent
  peu), et les échelles < 1 (jusqu'à 0,053) **amplifient** x plutôt que de
  le rapprocher du plancher

Détail par expert : `/tmp/glm-discriminateur-mma0/resultat-blocs-zero.json`,
script `outils/controle-blocs-zero-16-09.py`.

## VERDICT : cause ÉCARTÉE

Ratio techniquement infini (dénominateur nul) mais le critère scellé exige
**ratio ≥ 3× ET fraction(x/s) ≥ 1 %** — la seconde condition échoue
franchement (0 % ≪ 1 %). Sur cet échantillon réel, aucun bloc, brut ou
mis à l'échelle, n'approche le plancher dénormal E4M3 : les magnitudes
d'activation en jeu (couche 1, 16 jetons synthétiques) restent bien
au-dessus de 0,0117 même après division par l'échelle AWQ la plus faible
mesurée (0,053, qui amplifie plutôt que réduit dans ce sens de division).
Le plancher dénormal existe dans le noyau (lu au bon endroit, formule
vérifiée ligne à ligne) mais **n'explique pas** l'écart mesuré par le
discriminateur sur cet échantillon.

## Conséquence

Le verdict (1a) application reste debout (`verdict-glm-discriminateur-16-09.md`),
mais ce mécanisme précis n'en est pas la cause. La piste du plancher
dénormal n'est pas à exclure en général (elle pourrait se manifester à
d'autres positions/couches avec des activations de magnitude plus faible),
mais elle n'est pas ce qui explique la coupure nette observée à la
position 8/préfixe 9 jetons (`poste1-equivalence-pile-gateup-16-09.md`).
poste4 : chercher directement dans `model.py:983`
(`xs / awq["gate_proj"][e_sorted, :xs.shape[1]]`) ou l'ordre `route+pack`,
pas dans une hypothèse de précision du noyau d'activation — un échec est
un résultat, pas un chiffre à reconstruire pour combler ce trou.
