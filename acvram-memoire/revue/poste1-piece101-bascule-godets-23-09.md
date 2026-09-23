# Pièce 101, étape 1 — bascule par godet des projections nvfp4 : le port du Marlin DENSE est nécessaire au-delà du godet 2 — 23/09 (poste1)

* **instrument** : `outils/gpu/mesure/banc-proj-nvfp4.py`, BANC_LOTS=1,2,4,8,12, cinq bras dans UNE prise (L2 froid, graphe, mur/48, médiane de 30).
* **commit** : 0e563dd8 ; -lgc 2700 ; seul llama-server 4627 au début et à la fin ; prise de 13 s (17:42:21-34).
* **scellé** : `scratchpad/poste1-p101-23-09/scelle-banc.md`, commité avant.
* **mesuré**, qkv + o en µs par couche (détail par forme dans `prise-banc.log`) :

| godet | int8 servi | Marlin dense vLLM | marlin_moe_E1 | gemv_marlin_E1 | nvfp4 acvram (2ᵉ disposition) |
|---|---|---|---|---|---|
| 1 | 16,21 | 15,68 | 22,72 | 14,03 | **12,79** |
| 2 | 19,82 | 15,94 | 22,47 | **14,50** | 16,92 |
| 4 | 20,14 | **16,00** | 28,44 | 19,28 | 20,72 |
| 8 | 20,72 | **16,11** | 46,50 | 30,88 | 21,34 |
| 12 | 21,59 | **17,38** | 46,53 | 48,12 | 22,14 |

  * Justesse contre fp64 : bras Marlin MoE et dispositions acvram à 1,7e-3 ; `gemv_marlin_E1` à 8e-8, soit le déquantifié exact ; Marlin vLLM à 2,4-3,2e-3.
  * Chemin int8 : `gemv` à M=1, `etroit_triton` à M ≥ 2, comme prévu.
* **verdict, par les règles du scellé** :
  1. Règle 3 : **port du Marlin dense requis.** Notre Marlin MoE à E=1 est de 40 % (godet 2) à 190 % (godets 8-12) plus lent que le Marlin dense. Il ne parallélise que par blocs d'experts, qui se réduisent ici à un seul bloc. Prédiction ratée : 7-9 µs sur qkv prédit, 9,6-18,4 mesuré.
  2. Règle 4, sans port : le débit est perdu aux godets 8 et 12. Le meilleur bras à disposition Marlin y fait 30,9 et 46,5 µs, contre 20,7 et 21,6 pour l'int8. **Avec le port, le débit est gagné à tous les godets** : −21 % à −27 % contre l'int8.
  3. Règle 1, noyau par godet :
     * godet 1 : `nvfp4_gemv` sur la disposition nvfp4 (règle 2 tenue : −8,8 % contre `gemv_marlin_E1`, qui ferait 14,03) ;
     * godet 2 : `gemv_marlin_E1` ;
     * godets ≥ 4 : **Marlin dense porté**.
  4. Deux dispositions des poids q/k/v/o au godet 1 (+283 Mo, acceptés par le chef ; à relever sur carte à l'intégration). Si la VRAM venait à manquer, `gemv_marlin_E1` y ferait 14,03 au lieu de 12,79, soit +0,06 ms/pas à b=1.
* **Prédiction d'énergie inchangée**, avec des temps de projection de −21,6 % (godet 1) et −19,5 % (godet 12) contre l'int8 :
  * b=1 : −0,16 ms/pas (−5 %), J/jeton −5 à −8 % ;
  * b=12 : −0,20 ms/pas (−3,3 %), J/jeton −3 à −5 %.
* **Suite (étape 2)** : port du Marlin dense (marlin.cu, marlin_template.h, kernel.h et instanciation NVFP4 depuis `csrc/libtorch_stable/quantization/marlin/`, v0.29.0 ; repack déjà porté), intégration opt-in avec bascule par godet, capture, ulp contre témoins.
* **durée** : ≈ 40 min ; carte 13 s.
