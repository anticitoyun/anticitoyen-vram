# Verdict — duel GLM-4.7-Flash prise A : vLLM NVFP4 = 4,1 × notre débit à b=12 (796 contre 195 t/s), 0,28 × notre J/jeton ; scellé de poste7 (0,7-1,3×) RÉFUTÉ, ma prédiction (1,3-1,9×) RÉFUTÉE

- **instrument** : acvram `certifie-b12-15-09.py` (rondes ctx 2048 / invite 256 / ≥ 20 s, `energie.py` b11b8a3 + chrono hôte, écart ≤ 0,02 s, ×2 par lot) ; vLLM `decode-glm-vllm-16-09.py` (mêmes ctx/invite, fenêtre 20 s, NVML) ; prefill pp2048 : `prefill-glm-acvram-15-09.py` bras `w4a16`, `prefill-glm-vllm-15-09.py` (kv fp8, backend par défaut) ; sorties `scratchpad/duel-glm-16-09/`
- **commit** : travail/poste3 **42c24fb** (code `acvram/` = main ce71723) ; protocoles 9998eca + 2cb4e22 + f07ece8 ; vLLM 0.29.0, `/opt/ia/vLLM/.venv`
- **régime** : acvram `-k48`, **prefill W4A16 (`ACVRAM_MOE_MMA=0`), décodage MMA=1 MIN_T=5** (relu au JSON), NOMINAL 0/47 exilées, 0/2944 experts, piles_ok, graphes on ; vLLM GadflyII NVFP4 (compressed-tensors), KV fp8, TRITON_MLA `num_stages=1` ; une carte, -pl 400, horloge libre 2 940-2 985 MHz, 31-51 °C ; un drapeau « bridage puissance » transitoire sur b12-p2 (309 W moyens)
- **scellé** : poste7 `poste7-objectif-14-09` : vLLM 0,7-1,3 × notre débit b=12. Moi (f07ece8) : nous 205-222 t/s b=12, vLLM 1,3-1,9 × ; b=1 22-27 ms ; prefill acvram 8-12 k j/s > vLLM 6-10 k ; fp8 dyn OOM
- **mesuré** : b=12 acvram **56,19 / 56,10 ms · 194,8 / 195,1 t/s · 1,575 / 1,586 J** — vLLM **796,3 t/s · 0,445 J** (net 0,354) → **×4,08 débit, ×0,28 J**. b=4 : 124,7 contre 421,1 (×3,38) ; b=1 : 78,6 contre 153,7 (×1,96). Prefill pp2048 : acvram **4 401 j/s** (465 ms), vLLM **26 732 j/s** (77 ms, ×6,1). fp8 dyn : OOM au chargement (29,96 Gio), résultat
- **verdict** : **RÉFUTÉ** des deux côtés ; le duel est perdu 4× au décodage b=12 et 6× au prefill, à PPL comparable (colonne § 1). Ce n'est pas le MoE : c'est le MLA (§ 2)

## 1. Colonne PPL avec le régime (référence bf16 8,1427)
```
acvram -k48   prefill W4A16 (régime du duel)   8,1569   1,0018   verdict-ppl-k48-regime-duel
acvram -k48   prefill W4A4 par ligne           8,2913   1,0183   non pris au duel
vLLM GadflyII NVFP4                            non mesurée ici (même corpus à faire : 10 min de carte)
```
Le décodage MMA W4A4 (godet 12) n'est pas couvert par la PPL de prefill ; le coût E2M1 y reste (poste7 § 8.3).

## 2. Où sont les 56 ms
```
                          b=1        b=4        b=12       J/jeton b=12   prefill pp2048
acvram -k48 (ms/pas)      12,7       32,1       56,1       1,58           4 401 j/s
vLLM NVFP4 (t/s → ms)     153,7→6,5  421→9,5    796→15,1   0,445          26 732 j/s
acvram Coder-30B 0.6.6    4,48       7,85       13,60      0,494          (autre modèle)
```
- GLM (64 experts top-4, `moe_intermediate` 1 536, 46 couches MoE) lit par jeton **les mêmes octets d'experts** que Coder-30B (128 top-8, 768) : à b=12 nos 13,6 ms sur Coder deviennent **56,1 ms sur GLM, ×4,1** — l'écart est celui de l'attention **MLA** (`kv_lora_rank` 512, 20 têtes), déjà mesurée à **32,8 ms de noyaux `mla_*` à slots=12** (`part-reelle-mla-sous-graphes-13-09`), plus 2,0 Go d'int8 denses (q/kv/o, `first_k_dense`) lus à chaque pas (≈ 1,1 ms). vLLM tient 15 ms au même lot : son MLA Triton (absorption, KV fp8 compact) vaut ~4 ms. **Le duel se joue sur le MLA, pas sur le NVFP4.**
- b=1 : 12,7 ms chez nous (Coder : 4,5) — même lecture ; vLLM 6,5 ms.
- Prefill : nos 4 401 j/s en W4A16 (déquantification + boucle `fpg` 322 / `pile` 966) contre 16 938 en W4A4 m64e4 le 15/09 sur l'ancien converti ; le régime W4A16 coûte ×3,8 au prefill et le duel en hérite ; vLLM 26,7 k j/s.
- Énergie : vLLM 0,445 J/jeton brut à 355 W contre 1,58 J à 308 W chez nous — le rapport suit le temps, pas la puissance.

## 3. Ce qui manque pour que le chiffre soit complet, et ce que je propose
Prise B (ncu octets par jeton, `campagne-duel-glm-ncu-15-09.sh`) dira si vLLM lit moins d'octets ou les lit plus vite ; PPL vLLM même corpus (10 min) pour la colonne ; un nsys d'un pas b=12 GLM (`nsys-rejeu-b12-15-09.py`, 5 min) pour découper les 56 ms en MLA / MoE / trous — c'est la mesure qui nomme le chantier. Je ne lance rien : poste7 tranche l'ordre.


## Réserve (poste7 § 6, 16/09)

bras vLLM GadflyII à +5,4 points de PPL (1,056), hors seuil 1,02, non apparié : sa vitesse ne se classe pas — sans en faire une excuse, mêmes octets NVFP4 des deux côtés, notre ×4 est réel et se lit dans 15 534 lancements/pas. Bras vLLM apparié = conversion ModelOpt par nous (poste2, dans le comparatif 5 moteurs, pas avant).

## ERRATUM FORMEL (16/09, poste7 § 10, cause : verdict-glm-slots12-erratum-16-09)

**b=12 RETIRÉ.** Le régime mesuré (HYBRID_SLOTS=12) calcule une mauvaise attention
après un prompt réel — arbitre prefill 58/84 (défaut ET boucle, identiques) contre
81/84 à 4 créneaux. Le chiffre ×4,08 (56,1 ms) ne se lit plus comme une comparaison
valide. b=1 et b=4 du duel : à revérifier par l'arbitre à leur configuration exacte
avant republication.

## PUBLICATION FINALE (16/09, poste7 § 14, après correctif tête fp32 et prise B)

Chiffre définitif : `verdict-duel-glm-prise-b-16-09` (52c019c). Nous/vLLM = 0,70
(b=1) / 0,79 (b=4) / 0,65 (b=12) — scellé 0,8-0,9 RÉFUTÉ DE PEU à b=12, réserve
de qualité inchangée (vLLM PPL 1,056, non apparié). Régime W4A16 prefill +
décodage, arbitré SLOTS=b. **MLA : parité atteinte sur son poste** — chantier
clos. Postes restants (poste4, poste2) : projections int8, tête GEMV, MoE MMA.
