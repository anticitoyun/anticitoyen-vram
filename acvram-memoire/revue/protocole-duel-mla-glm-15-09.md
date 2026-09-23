# Protocole — duel MLA apparié GLM-4.7-Flash, acvram contre vLLM

Laure, 15/09/2026, avant mesure. Ordre : Sage
[`sage-objectif-14-09.md`](sage-objectif-14-09.md) (livrable 48 h), transmis
par Jérôme ; GLM-4.7-Flash-srcbf16-nvfp4 validé par Manon (21c121f, PPL
8,1275 vs bf16 8,1427, ratio 0,998). Carte après ma recourbe MIN_T.

## Bras

    acvram   /mnt/2TO_2023_980PRO/Modeles/models_acvram/GLM-4.7-Flash-srcbf16-nvfp4
             (source bf16 zai-org, experts NVFP4, MLA/lm_head selon manifeste),
             main du jour (v0.6.5 défaut : MMA au godet ≥ 12 ou seuil recourbé,
             route+pack), commit relevé au journal
    vLLM     /mnt/4TO_SATACMR_2022/Modeles/models_vllm/GLM-4.7-Flash-NVFP4
             (GadflyII, compressed-tensors NVFP4 : experts + MLP dense E2M1,
             attention MLA / lm_head / routeur bf16, 20,4 Go), vLLM 0.29,
             TRITON_MLA avec num_stages=1 dès BLOCK_DMODEL ≥ 512 (patch de
             Laurine, processus seul), KV fp8, graphes CUDA en décodage
    apparié  même source bf16 ; PAS le même format d'attention (nvfp4/int8
             chez nous, bf16 chez vLLM) — à dire dans le verdict, c'est la
             réserve principale ; le fp8 dynamique de la prédiction de Sage
             est impossible (31 Go), remplacé par ce checkpoint NVFP4

## Montage

    décodage   b=1 / 4 / 12, ctx 2048, invite 256, ≥ 20 s au compteur
               (acvram : rondes `certifie-b12-15-09.py`, 2 passes par b ;
               vLLM : `outils/banc_decode_vllm_glm.py`, lots de 1 024
               jetons jusqu'à 20 s), repos 30 s / 8 s avant, une carte
               (CUDA_VISIBLE_DEVICES=0 des deux côtés), -pl 400, horloge
               libre, température en en-tête ; J/jeton brut (et net côté
               vLLM, son banc le publie)
    prefill    pp2048, même dénominateur (L / durée d'un max_tokens=1
               complet, invite différente par répétition, 2 chauffes + 7
               répétitions, médiane) : copies paramétrées de
               `banc_prefill_chaud.py` (modèle par env) et
               `banc_prefill_vllm.py` (+ patch MLA)
    octets     prise séparée : `outils/ncu_instr_par_octet.sh` acvram / vllm,
               b=12, un pas, `--cache-control none` (L2 chaud), dram bytes
               read+write par pas, plafond 2 000 lancements
    sortie     `revue/duel-mla-glm-15-09.md` ; données
               scratchpad/duel-glm-15-09/

## Seuils de Sage (scellés)

vLLM ≤ 0,8 × notre débit b=12 avec ≥ 1,8 × nos octets, et ≥ 1,3 × notre
J/jeton ; **réfuté si vLLM ≥ notre débit à b=12 avec ≤ 1,5 × nos octets**
— alors « B se réduit à (c) et il faut le dire ».

## Ma prédiction (scellée, contraire à Sage)

Les experts sont NVFP4 des deux côtés : les octets d'experts sont égaux.
vLLM lit son attention MLA et son lm_head en bf16 (≈ 2× nos octets sur
cette part, qui est < 30 % du pas), soit **1,15-1,4 × nos octets par pas**,
pas 1,8. Sur Coder-30B vLLM fait 2,1 × notre débit à b=12 (1 437 vs 674) ;
le MLA Triton non natif de sm_120 lui coûte, mais pas un facteur 2,6 :
**vLLM b=12 = 1,3-1,9 × notre débit** (nous : 450-600 t/s, lui 750-1 000),
J/jeton vLLM 0,6-0,8 × le nôtre. À b=1 : nous 150-190 t/s, vLLM 120-180,
J proches (± 15 %). Prefill pp2048 : vLLM 1,5-2,5 × nous. **Verdict
attendu : réfuté au sens de Sage (vLLM ≥ nous avec ≤ 1,5× octets).**
Réfuté (mon bord) si vLLM b=12 ≤ 0,8 × nous (le MLA Triton est bien pire
que je crois) ou si ses octets ≥ 1,8 × (l'attention bf16 pèse plus que
30 % du pas). Alarme d'avance : si un bras ne charge pas (OOM, graphes
refusés, backend absent), c'est un résultat publié tel quel, pas un
chiffre reconstruit.

## Complément du 16/09 (à sec, avant le go) — bras vLLM

* Bras acvram servi depuis **mon worktree figé** (`ACVRAM_ARBRE`), jamais
  l'arbre main ; dossier acvram = celui que Manon livrera après reconversion
  (formats homogènes par couche — la garde `piles_ok` du régime NOMINAL est
  la preuve, écrite au journal), chemin et manifeste au verdict.
* Bras vLLM **NVFP4** : GadflyII (20 Go, présent), `scratchpad/decode-glm-vllm-16-09.py`
  = copie du banc de Laurine + `BANC_QUANT`/`BANC_GPU_UTIL`, energie.py corrigé.
* Bras vLLM **fp8 dynamique** (prédiction de Sage) : même script,
  `BANC_QUANT=fp8` sur la source bf16 (49 shards), `gpu_memory_utilization`
  0,97, b=1 et 12. Attendu : **OOM au chargement** (31 Go de poids fp8 sur
  32) — c'est la moitié « ne sert pas ce modèle » de la prédiction ; l'OOM
  est publié comme résultat avec le message, pas contourné. S'il charge
  (KV fp8 2048 sur < 1 Go), la mesure vaut et le seuil de Sage s'applique.
