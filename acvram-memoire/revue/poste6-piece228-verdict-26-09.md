# Verdict — pièce 228 : qualité de la 209 sur le Coder-30B-A3B-nvfp4 PUR (décodage b=8, texte réel forcé) — KL TENUE (0,560 ≤ 2 × 0,460), argmax TENU (0,9727 = T1), PPL +0,33 % sous la résolution de l'instrument ; la 209 reste au défaut pour ce modèle (poste6, 26/09)

* **instrument** : `scratchpad/poste6-p228-26-09/kl-decode-228.py` (= `kl-decode-209.py` de la 209 c, kl2 de la 195 : texte RÉEL forcé — README × 3,
  jetons imposés, aucune génération libre —, 8 séquences × 32 pas à b=8, deux processus, logits A sauvés hors git), `chaine-228.sh` ;
  `kl-{A,B}.log`, `kl-Qwen3-Coder-30B-A3B-nvfp4.json`.
* **commit** : 188d5bdb7 (poste6-228 = origin/main 1f72f910f + scellé + instrument) ; noyau servi inchangé.
* **régime** : carte 0, -lgc 2700, llama-server 4219 (tiers) ; **A = `ACVRAM_MARLIN_PAR_LIGNE=0`** (témoin : 4 piles refusées « sous-normales »,
  naturel + decode_mma W4A4) ; **B = 1** (défaut : Marlin-w13 par colonne, W4A16 exact au poids). Preuve du bras sur les couches : **A 4/48
  refus, B 0/48** ; rejeu A = 0, rejeu B = 0.
* **scellé** : `scratchpad/poste6-p228-26-09/scelle.md` (188d5bdb7) — seuils de la 209 c : KL_AB max ≤ 2 × max(T1, T2), argmax AB ≥ T1 − 0,5 pt.
* **mesuré** : **KL_AB max 0,560** (moy 0,0090, p99 0,054 ; par séquence 0,052 / 0,087 / **0,560** / 0,053 / 0,027 / 0,056 / 0,023 / 0,031) ;
  **T1 max 0,460** (moy 0,0109 ; séquence 3 : 0,460 — la même que l'AB max), T2 = 0 → **seuil 0,919** ; **argmax AB 0,9727 = argmax T1/A 0,9727** ;
  PPL A 12,613 · **B 12,655** · T1 12,577 ; NLL par fenêtre (8 × 2) : B hors de l'intervalle [A, T1] sur 9 fenêtres sur 16, de ± 0,05 nat.
* **verdict** : **KL TENUE** (0,560 ≤ 0,919, la séquence la plus sensible l'est aussi pour le témoin T1), **argmax TENU** (égal au témoin) ;
  PPL_B − PPL_A = **+0,33 %** : prédiction « PPL_B ≤ PPL_A » FAUSSE (sur i8c la 209 c donnait −1,5 %), mais 256 jetons notés résolvent
  ± 1 % environ (REGLES § 3 : 3 × 512 jetons = ± 0,6 % SE) — l'écart est sous la résolution, ni preuve ni réfutation, dit tel quel.
  Aux deux critères de la 209 c, **la 209 est qualifiée sur le Coder pur : elle reste au défaut**.
* **durée** : A 8 min 3x (dont file 1 045 s), B 8 min 1x (file 1 202 s) ; prévu 2 × ≤ 6 min sous prise (`ACVRAM_DUREE_MAX=600`) : tenu.

## Lecture
* Les 4 couches basculées sont les mêmes que sur i8c (mêmes tenseurs d'experts, 209 a) : l'écart A/B a la même origine — W4A4 (activations
  E2M1) contre W4A16 exact — et la même ampleur que le témoin de lot T1 (b=1 contre b=8), séquence par séquence (corrélation visible :
  la séquence 3 domine les deux).
* PPL : A 12,613 contre B 12,655 contre T1 12,577 — les trois dans 0,6 % ; pour trancher le signe il faudrait `ppl-decode-kv` (3 × 512 jetons,
  ± 0,6 %) sur les deux bras : 2 × 3 min, à la demande de chef seulement.

## Suite (à chef)
Rien à changer : PAR_LIGNE=1 reste au défaut sur le Coder pur. Si le signe de la PPL importe : `ppl-decode-kv` A/B (2 × 3 min).
