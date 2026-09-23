# Verdict — la pire fenêtre GLM relue jeton par jeton : instabilité du modèle, cause = absence de `[gMASK]<sop>`

instrument : `scratchpad/sonde-fenetre-{hf,acvram,llamacpp}-17-09.py` (pertes par jeton : HF bf16 `device_map=auto` ; acvram par capture des `cross_entropy(reduction="none")` d'`evaluate.perplexity` ; llama.cpp par `--kl-divergence-base`, log-probs uint16 à plancher max−16) — journaux `scratchpad/sonde-fenetre-17-09/`
commit : arbre laure 2f998da + sondes (ce commit)
régime : corpus privé, tranche 0 (tokenizer GLM), fenêtre 8 = jetons 16 384..18 431, cibles 1024..2047 pour acvram/llama.cpp, 1..2047 pour HF ; acvram `-k48` W4A16 (`ACVRAM_MOE_MMA=0`), llama.cpp Q4_K_M unsloth, HF bf16
scellé (Sage § 9.2) : première perte > 20 nats à la MÊME position dans les trois → instabilité du modèle ; un seul bras → bogue de ce moteur
mesuré : PPL de la fenêtre 8 (cibles 1024..2047) : HF bf16 **110 219**, acvram **145 435**, llama.cpp **98 228** (KL, plancher) — les trois d'accord ; HF vs acvram jeton par jeton : |Δ| médian **0,2 nat**, 92,6 % d'accord sur « > 10 nats » ; première perte > 20 nats : HF **cible 53** (HF seul note la 1re moitié), acvram **1045**, llama.cpp **1045** (premières cibles de leur plage ; HF y vaut 30,5) ; HF par blocs de 128 dès le début : 8,7 / 10,3 / 10,2 / 11,0 / 11,1 / 11,4 / 11,3 / 10,8 nats — toute la fenêtre est à ~11 nats (≈ hasard sur 155 k jetons), pas un jeton fautif
verdict : **instabilité du modèle, identique dans les trois moteurs, sur toute la fenêtre.** Cause trouvée par témoin : la même fenêtre seule (2 048 jetons, min-ctx 0) donne PPL **104 637** ; préfixée de `[gMASK]<sop>` (2 jetons, non notés) : **31,9** (pertes par bloc 5,3 / 4,1 / 3,4 / 3,4 / 3,6 / 3,1 / 3,6 / 3,1). GLM-4.7-Flash n'a pas de puits d'attention sans son préfixe de début de séquence ; selon le premier jeton de la fenêtre (' 0' ici ; '#' pour les fenêtres 0-1 qui vont bien) il s'effondre ou non. **Aucune PPL GLM publiée jusqu'ici (wiki-gptq comprise, donc le 0,9956) n'est un classement : elles mesurent des effondrements différents par fenêtre, engine-indépendants mais numériquement différents une fois effondrés.**

## Ce que ça impose (à Sage)
1. Cadrage GLM : chaque fenêtre commence par `[gMASK]<sop>` (2 jetons, exclus des cibles ; 2 046 jetons de texte), pour TOUS les bras — HF/acvram/vLLM/TRT-LLM reçoivent les identifiants ; llama.cpp : corpus pré-découpé en fenêtres textuelles préfixées (`parse_special`), retokenisation à vérifier à 2 048 exactement. Coder n'a pas de préfixe (`add_bos_token: false`, fenêtres régulières 6-23) : rien à changer.
2. Refaire les PPL GLM (public et privé) sous ce cadrage ; le 0,9956 se rejuge là.
3. La leçon vaut pour le comparatif de temps : les invites de jetons tirés sans préfixe mettent GLM dans le régime effondré — les t/s ne dépendent pas des valeurs, mais un bras qui sample (llama.cpp sans `ignore_eos`) l'a déjà montré.

## Point 1 de Sage (médianes, PPL par fenêtre) : en cours
Instruments à compléter (HF, vLLM, TRT-LLM : PPL par fenêtre dans le JSON ; acvram : sonde par capture) puis Coder recalculé en médianes (attendu < 0,005 d'écart) — même passage que la refonte GLM.
