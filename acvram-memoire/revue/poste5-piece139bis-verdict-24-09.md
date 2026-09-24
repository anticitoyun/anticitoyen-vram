# Verdict — pièce 139 bis : PROJ_MARLIN sur l'alias mixte unsloth (poste5, 24/09)

* **instrument** : `scratchpad/poste5-p139-24-09/prise-cbis.sh` + `banc-chat-openai.py` (banc de la 102), A B B A A B B A
  A B par b, serveur neuf par passe ; dépouillement `resume-cbis.py` ; NVML `energie.py`, carte 0 seule
* **commit** : 06432321 (branche poste5), alias `Qwen3.8-27B-unsloth-mixte-i8c`
* **régime** : A défaut, B `ACVRAM_PROJ_MARLIN=1 ACVRAM_PROJ_MARLIN_DOUBLES= ACVRAM_GEMV_MARLIN_V2=1
  ACVRAM_GEMV_MARLIN_TPB=1 ACVRAM_GEMV_MARLIN_S=0` — relevé 5/5 sur chaque b : `marlin(doubles=0,seuls=112,0.00Go,kv=…)`,
  A 5/5 « sans marlin » ; `prefill_int8=cublas+bf16(origine fp8 ×233)` des deux côtés ; -lgc 2700, plafond 400 W ;
  cpu-safe 100 ; `--max-model-len 4096` (aucun refus, repli 2048 non utilisé) ; 10/10 fenêtres valides par b ; load1 ≤ 4,1
* **scellé** : `revue/poste5-piece139bis-scelle-24-09.md` (écrit avant) — b=8 +12 à +30 % (FAUX < +8 ou > +40), b=1 −3
  à +5 %
* **mesuré** : b=1 A 58,7 / B 58,7 t/s (−0,07 %), J/jeton −3,0 % ; b=8 A 247,2 / B **296,3** t/s (**+19,85 %**, Welch
  t +51,5), J/jeton **−22,8 %** (1,234 → 0,952)
* **verdict** : **TENU** aux deux b. À b=8, horloge B 2 634 contre A 2 554 MHz (+80, au-delà de 30) : B consomme moins
  (358 contre 381 W) et bride moins. Gain réel **en service**, pas à horloge égale.
* **durée** : prévu 2 × ~9 min / tenu b=1 10:28-10:37, b=8 10:38:57-10:47:50 (`carte.sh`)

## Lecture

* Ma décomposition à sec du scellé prédisait ≈ 5,7 ms/pas gagnés à b=8. Mesuré : 32,4 → 27,0 ms/pas, **5,4 ms**. La part
  des MLP nvfp4 à 0-55 se retrouve à 5 % près.
* Face à NInfer (139 c, même banc, autre séance) : b=8 **296,3 contre 463,3 t/s (−36 %)**, contre −47 % sans Marlin.
  Reste 9,7 ms/pas d’écart (27,0 contre 17,3). Les int8 (≈ 5,8 ms d'après la décomposition du scellé) en sont maintenant la plus grosse
  part isolable. Le levier suivant est un GEMV/GEMM 8 bits plus rapide sur ces 10,63 Go, ou leur passage en format
  Marlin 8 bits.
* J/jeton b=8 : 0,952 contre NInfer 0,699 (+36 %) ; b=1 : 5,15 contre 4,35 (+18 %, l'int8 lu à 1,1 To/s).
