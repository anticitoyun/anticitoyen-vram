# Scellé — pièce 125 ter : (a) prix du W8A8 sur invites longues ; (b) bisection de l'excédent du préfill (poste6, 24/09, AVANT la mesure)

Feu de chef sur la 125 bis (43c7c97b) : pas de défaut servi (W8A8 gardé, ×1,3 au préfill) ; une prise ≤ 15 min, scellée.

## (a) Prix du W8A8 sur des invites ≥ 512 jetons — information pour le README, pas un verdict
* Invites : les 5 questions de kl-gabarit précédées d'un contexte de 480 jetons du même texte (`scratchpad/poste6-p107-23-09/
  prose-fr.txt`, prose française) → 512-560 jetons d'invite ; réponse forcée = les 32 jetons de HF de l'invite courte i (mêmes
  positions de jugement). **Sans référence HF** (un forward de 550 jetons du 30B sur processeur × 5 ne tient pas dans la prise) :
  le prix est mesuré CONTRE LE TÉMOIN W8A16 (`ACVRAM_INT8_GEMV_MAX=4096`, GEMV), comme en 125 bis — KL(témoin ‖ cublas) par
  position de réponse, argmax égaux, L2 relative par couche sur les positions d'invite. Étiquette obligatoire dans tout
  chiffre publié : « relatif au témoin W8A16, pas à HF ».
* Prédit : KL_moy 0,01-0,04 nat, KL max ≤ 0,5, argmax égaux ≥ 90 %, L2 d'invite en fin de pile 0,05-0,15.
* Alarme d'avance : le GEMV n'est exercé qu'à n ≤ 80 en production ; à n = 550 c'est LUI qui sort de sa plage. Si KL_moy > 0,2
  ou argmax < 80 % sur une invite, je dis « témoin hors plage », pas « prix du W8A8 ».

## (b) Bisection de l'excédent des invites 1-2 (préfill ×2,1 et ×3,35 le forcé, à projections identiques)
Invites courtes (dumps HF du 02 h 03), Coder i8c, mode `prefill`, un bras par variable, tous dans la même prise, défaut rejoué :
| bras | variables | ce qu'il coupe | preuve dans le processus |
|---|---|---|---|
| D | défaut | — | `chemins_moe` : `mma` (W4A4 sur MMA native, moe.py:788-919 : `nvfp4_quant_act` sur gate/up ET sur l'entrée de down) |
| B1 | `ACVRAM_MOE_MMA=0` | le W4A4 des experts au préfill → groupé W4A16 (`nvfp4_gemm_grouped`, par_expert ≤ 48) ou Marlin | `chemins_moe` sans `mma` ; ligne de régime |
| B2 | `ACVRAM_PREFILL_DEQUANT=1` | tout chemin direct des experts → déquant bf16 (le chemin d'avant) | `chemins_moe` ; ligne de régime |
| B3 | `ACVRAM_PREFILL_COMPACT=0 ACVRAM_GLUE_COMPACT=0` | la glue C15 (résidu différé, épilogues fusionnés, fusion attn) | ligne de régime |
L'attention de préfill n'a pas de témoin par variable (SDPA bf16 dans tous les cas, attention.py:465-467) : le bras `force` du
02 h 04 reste son seul témoin (tout par le décodage). Le point (b) ne peut donc isoler que les experts ou la glue.
* **Un bras isole la cause** si, sur les invites 1 ET 2, KL_moy(bras) ≤ 1,25 × KL_moy(forcé) (soit ≤ 0,0045 et ≤ 0,029) alors
  que D rejoué reste ≥ 2 × forcé sur les deux ; « partiel » si une seule des deux ; sinon « cause hors des trois ».
* Sur les 3 autres invites le bras ne doit pas dégrader : KL_moy(bras) ≤ 1,25 × KL_moy(D) — sinon le bras remplace un défaut
  par un autre et ne prouve rien.
* Prédit : **B1 et B2 isolent** (le W4A4 des experts — activation E2M1 sur gate/up et sur down — est la seule quantification
  d'activation que le décodage ne paie pas, une fois le W8A8 hors jeu sous 80 lignes ; le commentaire moe.py:775 « coupé par
  défaut » est faux : `MOE_MMA=1` par défaut) ; B3 au bit avec D (fusions « au bit » par construction).
* FAUX si ni B1 ni B2 ne ramènent les invites 1-2 sous 1,25 × forcé → cause hors des trois (attention SDPA, ordre du lot,
  routage) ; je le dis sans mesurer plus.
* Ce qui me gênerait : B3 change les chiffres — les fusions C15 ne seraient pas au bit ; je le dirais et ce serait une pièce.
* Verdict seulement (ordre) : aucun défaut n'est déclaré ici ; le W4A4 de préfill a son histoire (poste2 13/09 : +2,58 % de PPL
  sans lissage ; poste7-w4a4-clos-w4a8-porte-19-09) que chef relit avant de décider.

## Durée et régime
(a) 2 bras longs (14 s + 5 × ~2 s) ≈ 1 min ; (b) 4 bras courts ≈ 1 min 30 ; invites longues à sec avant la prise (tokeniseur seul).
Une prise ≤ 5 min (budget 15). cpu-safe=off, relevés début/fin, compute-apps début = fin. Preuves : `prefill-<bras>-…-preuve.json`.
