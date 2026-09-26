# Verdict — 130 : GEMV Marlin v2 (opt-in) à la parité du naturel à b=1 ; témoins KL et KL des chemins Marlin — 24/09 04 h 0x (poste1)

* **instrument** : `scratchpad/poste1-p129-24-09/prise-130.sh` (tests carte, `kl-chemins.py` × 5 bras, `banc-gemv-marlin-v2.py`) ; `casser-130.sh` (bras cassants, cache de compilation séparé)
* **commit** : e714483b (prise 130), fa0f4b80 (bras cassants) — poste1-mtp
* **régime** : Qwen3.8-27B-nvfp4 (KL, b=1, 5 invites neuves `invites-neuves-129/`, 8 pas) ; banc à -lgc 2700 ; cpu-safe=off (100/100) ; compute-apps début = fin (llama-server sur la 3080 Ti) ; dumps KL hors git, sha256 16 : A 0dcba53a…, T1 9365830…, T2 492431c…, B aee04e6…, B2 930327b…
* **scellés** : `scelle-temoin-kl.md` (KL ≤ 2 × max des témoins) ; dossier 130 § 4 (projection b=1 ≤ +3 % ; réfuté si gate‖up > +8 % → ncu)
* **mesuré** :
  * Tests : 60 passed (v2 au bit de v1 à S égal pour tpb 1/2/4, contre le naturel ≤ 2⁻⁷, K = 17 408 en un lancement, 129 (2), GEMV Marlin v1) ; bras cassants **2/2 ROUGES** (ordre : v2 ≠ v1 au bit, 70 colonnes ; index x : 4 952 colonnes hors 2⁻⁷).
  * **Témoins KL** : T1 (`ACVRAM_DENSE_NVFP4=gemv`) **0 exactement** (prédit : il ne touche pas b=1) ; T2 (`ACVRAM_NVFP4_GEMV_MAX=0`, cuBLAS contre GEMV sur tous les appels) **0,002455** → seuil **0,00491**. Chemins Marlin : B (mixte, v1) **0,000675**, B2 (mixte, v2) **0,002428** ; argmax 40/40 partout.
  * **Banc b=1** (µs par appel ; projection au pas b=1, naturel 10,118 ms) : v1 11,47 (**+13,4 %**) ; **v2 réglage unique TPB = 1, S auto : −0,58 %** ; meilleur par forme −0,73 %. gate‖up 63,81 → 64,36 (**+0,9 %**), down 36,64 → 34,40 (−6,1 %, un lancement), tête 523,7 → 431,5 (−17,6 %), q +1,5 %, GDN qkv +1,6 %, gate +2,3 %, **GDN out et o_proj +10,2 %** (5 120 × 6 144, S = 4). Justesse v2 : erreur relative ≤ 0,0017 (naturel ≤ 0,0039).
* **verdict** :
  * **130 TENUE** : −0,58 % ≤ +3 %, gate‖up à +0,9 % (réfutation non atteinte, pas de ncu). La disposition Marlin UNIQUE est viable à b=1 : plus de 8,5 Gio doublés.
  * **Ma prédiction s'est trompée de levier** : j'attribuais gate‖up (+15 %) à l'accès à tuile unique (levier 1), or TPB = 1 est le meilleur réglage. Le gain vient du levier 2 : x lu en global, sans copie en mémoire partagée ni barrière. La queue du split-K (levier 0) garde GDN out et o_proj à +10 % : un reste nommé, 0,16 ms/pas.
  * **KL tenue** pour les deux chemins Marlin MIXTES (≤ 0,00491). **Bras UNIQUE + v2 (B3, mesuré en tête de l'ABBA, 04:0x) : 0,005452 > 0,00491 → FAUX sur la KL** (argmax 40/40, `marlin(doubles=0,seuls=305)`). En unique, gate‖up, down et GDN out passent AUSSI au préfill par la GEMM Marlin et à M = 1 par le v2 : l'écart au défaut s.additionne sur les 64 couches, et dépasse de 11 % le seuil de 2 × témoin. La vitesse de l'ABBA reste mesurée ; le défaut unique n'est pas qualifié tel quel.
* **durée** : prise 130 03:51:07 → 03:54:47 ; bras cassants 03:55:19 → 04:00:30

Suite : ABBA b=8 puis b=1, bras B = disposition unique + v2 (addendum au scellé, avant les lots).
Note : le commentaire de `mb_splitk` (acvram_kernels.cu, « =0 (défaut) : S = 1 ») contredit son code (défaut 1 = automatique). Le code fait foi ; le commentaire est à corriger.
