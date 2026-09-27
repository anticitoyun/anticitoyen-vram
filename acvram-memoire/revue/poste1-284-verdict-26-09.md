# 284 — préfill du mixte à l'échelle : la cause est la FRONTIÈRE D'INSTANTANÉ des hybrides, pas l'absence de lot GDN (poste1, 27/09)

* instrument : `scratchpad/poste1-p284-27-09/` — `profil-mixte-284.py` + nsys + `fenetres-270.py` (fenêtres de temps, REGLES § 4) ;
  `client-carte-274.py` (TTFT servi, flux, invites réelles distinctes ~470 jetons, sans lecteur /metrics) ; prises
  `prise-nsys-284.sh`, `prise-ttft-284.sh`, `prise-cache-284.sh`
* commit : poste1-284 61bb7825f (nsys, TTFT LOT), abf098e43 (cache) ; base origin/main 0.7.4 + code de la 245
* régime : Qwen3.8-27B-unsloth-mixte-i8c, -lgc 2700, ACVRAM_ECO=off, `serve --max-batch 16 --max-model-len 4096 --speculative none`
* scellé : `scratchpad/poste1-p284-27-09/scelle.md` (4d2279233, à sec) + addenda 1-2 (avant chaque prise)
* mesuré : nsys p1/p12 × LOT 0/1 ; TTFT b=1/12 LOT 0 1 1 0 ; TTFT b=1/12 cache de préfixe D C C D
* verdict : H1 (lot GDN, 245) **FAUX** comme prédit ; H2 (projections dominantes) **TENU** — mais c'est la DÉQUANTIFICATION qui
  domine, refaite par séquence et par passe ; sans cache de préfixe, TTFT b=12 **5,47 → 1,92-2,04 s (−63 %)**
* durée : prévu 15 + 20 + 15 min ; tenu 17:1x-17:38 (nsys + TTFT), cache 265 s (journal `tenue=`)

## Ce que dit la trace (p1 = 1 × ~470, p12 = 12 × ~470 ; noyaux = 95-97 % du mur)
| famille (p1) | ms | part | noyau principal |
|---|---|---|---|
| déquant int8 → bf16 | 207,1 | 44,6 % | `int8_dequant_kernel` (525 lanc., 395 µs) — étiqueté « int8_cublas » par ma regex, c'est une déquant |
| GEMM bf16 | 148,7 | 32,0 % | cutlass s16816 bf16 |
| déquant nvfp4 (Marlin → bf16) | 52,8 | 11,4 % | `depaqueter_marlin_kernel` |
| GDN (fla, conv, portes) | 8,3 | 1,8 % | — |
p12 : mêmes parts ; `int8_dequant` relancé × 12,9 (6 784 lanc., 2,53 s) → aucun partage B′. LOT=1 : mêmes lancements, même mur.

## Cause (lue)
`Engine.step` (`runner.py:1583-1584`) ne groupe le préfill que si AUCUNE séquence n'a de `_frontiere_insta` ; sur un hybride
avec cache de préfixe ON (défaut), toute invite plus longue que le pas d'instantané en a une (`runner.py:1238-1252`) → boucle
« une par une », chaque séquence coupée en DEUX passes → la déquantification de chaque linéaire est refaite 2 × par séquence.
Les invites de 78 jetons du banc chat ne franchissent pas la frontière : elles passent en lot, d'où un défaut jamais vu.

## TTFT servi (ms, p50 ; deux passages)
| | b=1 | b=12 | débit b=12 (t/s) |
|---|---|---|---|
| D (défaut, cache de préfixe ON) | 466 / 482 | **5 471 / 5 461** | 118 |
| C (`--no-prefix-cache`) | 300 / 319 | **2 040 / 1 923** | 241 / 240 |
| LOT=1 (245), cache ON | 480 / 481 | 5 646 / 5 536 | 121-122 |
Prédit C b=12 ≤ 2,0 s : 1,92 tenu, 2,04 au-dessus de 2 % — seuil FAUX (3,5 s) loin. C b=1 ≤ 0,40 s : tenu.

## Suite proposée (chef tranche)
1. **Préfill groupé COUPÉ à la frontière** (cache de préfixe gardé) : deux passes GROUPÉES (toutes les séquences jusqu'à leur
   frontière, instantanés, puis le reste) au lieu de 2 × b passes → deux déquantifications par pas quel que soit b. Prédit :
   TTFT b=12 ≤ 2,4 s (C + une déquant), b=1 inchangé. Qualification : la numérique est celle du chemin groupé déjà servi aux
   invites courtes ; KL contre le chemin actuel scellée avant code. FAUX si b=12 > 1,2 × C dans les deux paires.
2. Ou, plus court : cache de préfixe coupé par défaut sur les hybrides à la demande — à peser contre ce qu'il rapporte en
   conversation multi-tours (non mesuré ici).
3. Levier suivant (plus grand, plus long) : GEMM int8 W8A16 sans déquantification préalable (la déquant fait 56 % du p1).
Reste de la 274 : mixte contre NInfer — le serveur NInfer n'a pas pu lier son port (acvram du passage précédent encore
dessus) ; à rejouer avec un port par passage.
