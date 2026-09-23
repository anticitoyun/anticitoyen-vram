# 1aj décodage — projections d'attention et lm_head en NVFP4 (à sec, 14/09, Laurine)

Ordre de Sage (`revue/sage-point0-ordre-14-09.md`) : poids q/k/v/o (et lm_head,
à part) en NVFP4 au décodage, GEMV existant. Seuils scellés : PPL Coder-30B
≤ int8-promu × 1,01 (réfuté > 1,015 → repli lm_head seul, −0,15 Go) ; J/jeton
b=1 ≤ 1,10 (réfuté ≥ 1,25) ; ms/jeton b=1 ≤ 3,6 (contrôle du point 0 : s'il
rend ≥ 4,0, les octets n'étaient pas le goulot). Issue nommée : q/k refusent
le W4 → test par projection.

## État lu dans le manifeste (`Qwen3-Coder-30B-A3B-nvfp4/acvram_manifest.json`)

193 tenseurs `int8`, tous `promoted_from: nvfp4` sous `snr_floor=25` :
les 48 × (q, k, v, o) et le lm_head — **rien d'autre n'est promu** (experts
nvfp4, routeur/normes/plongements bf16). Par jeton à b=1 (ncu, point 0) :
projections 0,93 Go, lm_head 0,32 Go, MoE 1,02, total 2,30 (llama.cpp 1,95).

| classe | Mparam | int8 (Go) | NVFP4 (Go, 0,5625 o/param) | gain |
|---|---:|---:|---:|---:|
| q_proj + o_proj | 805 | 0,82 | 0,45 | −0,37 |
| k_proj + v_proj | 101 | 0,10 | 0,06 | −0,05 |
| lm_head | 311 | 0,32 | 0,17 | −0,14 |
| **tout** | 1 217 | 1,24 | 0,68 | **−0,56 (−24 % des octets du jeton)** |

## Ce qui est déjà mesuré : la variante « tout en NVFP4 » est A6

Manon, 14/09 (`verdict-a6-int8-snrfloor0-14-09.md`) : reconversion
`snr_floor=0` = exactement ces 193 tenseurs en NVFP4 (le manifeste n'a pas
d'autre promotion) → **PPL 10,1574, +8,79 %** (seuil 1aj : ≤ +1,0 %).
**La variante complète est réfutée avant d'être écrite.** Ce qui reste
ouvert est le test par projection : laquelle des quatre classes porte les
+8,79 % (l'issue de Sage : q/k). Le débit −93 % d'A6 était un autre défaut
(`_try_build_stacks` refusait la pile hétérogène), sans rapport.

## Outil livré : `acvram convert --promotion-classes q_proj,k_proj`

`ConvertOptions.promotion_classes` (`acvram/quant/convert.py`) : seules
les classes nommées restent promouvables sous `snr_floor` ; les autres
restent NVFP4 même sous le plancher. Quatre conversions à faire (Manon,
carte : calibration), même corpus/régime PPL que l'étalon :

| variante | promouvables (int8) | en NVFP4 | octets/jeton | ms/jeton b=1 prédit (borne octets, 550 Go/s) | PPL prédite |
|---|---|---|---:|---:|---|
| A | q_proj, k_proj | v, o, lm_head | 2,30 − 0,33 = 1,97 | **3,6** | ≤ ×1,01 si l'issue de Sage est la bonne |
| A' | q_proj, k_proj, lm_head | v, o | 2,11 | 3,8 | ≤ ×1,01 |
| C | v_proj, o_proj, lm_head | q, k | 1,88 | 3,4 | > ×1,015 (q/k refusent) |
| B | aucune (= A6) | tout | 1,74 | 3,2 | **×1,088 mesuré — réfutée** |

Réfutation de l'issue « q/k refusent » : A > ×1,015 ET C ≤ ×1,01 (c'est v/o).
Si A et C dépassent tous deux : aucune projection ne tient le W4 seule →
repli lm_head seul (−0,14 Go, −6 %, ≈ 3,9 ms/jeton), et 1aj décodage se
ferme sur les octets ; le levier restant à b=1 est alors le KV (int8 →
fp8 ne change pas les octets) — rien.

## Deux issues gênantes à nommer avant la carte

1. **b=12 : le GEMV NVFP4 dense n'a pas de NV=12.** `nvfp4_gemv`
   (`acvram_kernels.cu:1229-1240`) va jusqu'à NV=8 : un lot de 12 (ou 16
   en godet) fait deux lancements et relit les poids (le L2 les sert,
   leçon du 14/09 — mais le calcul est doublé). C'est le motif du
   184,9 → 173,5 t/s de `FEUILLE-DE-ROUTE.md:833` à b=1 (767 contre 1 780
   Go/s demandés). À b=1 sans objet ; à b=12 il faudra NV=16 comme pour
   l'int8 (une heure) ou la MMA (1aj « vraie » : W4A4 sur tuile m16).
2. **ms/jeton ≤ 3,6 suppose la borne octets à 550 Go/s** (notre débit
   mesuré à b=1) ; si le GEMV NVFP4 dense tourne à 767 Go/s demandés
   (bloc 4 lignes, un chargement par fil), les 0,45 Go de q/o en NVFP4
   prendront ≥ 0,6 ms — le gain de temps serait nul même avec la PPL
   tenue. Le contrôle : profil eager b=1 de la variante, ms de
   `nvfp4_gemv` sur q/o contre 2,62/12 ms d'int8 aujourd'hui.

Rien de ceci n'a besoin de la carte avant les quatre conversions de Manon.
