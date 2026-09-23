# Verdict — refonte des PPL (Sage § 9-10) : GLM avec `[gMASK]<sop>`, PPL par fenêtre, médianes ; Coder recalculé

instrument : `scratchpad/ppl-{hf,acvram,llamacpp,vllm,trtllm}-17-09.py` + `ppl_commun_17_09.py` (fenêtres de 2 048, cibles 1024..2047, préfixe non noté, PPL par fenêtre, géo + médiane, explosées = > 10× la médiane du bras, 4 premiers ids) ; llama.cpp : `llama-perplexity-ps` (même commit 4c9233c, `parse_special=true`, `bin-ps/` à part) via `--kl-divergence-base` (log-probs uint16, plancher 16 nats) ; chaîne `scratchpad/ppl-refonte-17-09/chaine.sh` — JSON et journaux dans `scratchpad/ppl-refonte-17-09/`, PPL par fenêtre dans `ppl-par-fenetre-refonte-17-09.md`
commit : arbre laure 59cd2ca pour les prises (acvram GLM sur 59cd2ca, avant fusion de 505f31c) ; protocole `protocole-ppl-corpus-prive-17-09.md` + Sage § 9-10
régime : GLM : préfixe = `Tokenizer.prefixe_gabarit` du moteur (505f31c) = `[gMASK]<sop>` = ids [154822, 154824] ; fenêtres fabriquées par `encode_brut` (`fabrique-corpus-encode-brut-17-09.py`) — **fichiers identiques octet pour octet** aux fichiers préfixés de la chaîne (`cmp`, 6/6), donc les prises acvram sont celles d'`encode_brut` sans relance ; HF/vLLM/TRT-LLM reçoivent les mêmes ids ; llama.cpp les retokenise (jetons 24 576 = 24 576, ids prouvés). Coder : sans préfixe (`add_bos_token: false`), ids [2, 2823, 329, 220]. Corpus privé `corpus-revue-5909d27` (sha256 `76088761d41b2abc1b00eda66e0c37c91009c911fdf3dcc9b2137d5c0404f0a2`) et public wiki-gptq (tranches 0/24 576/49 152), 3 tranches × 12 fenêtres
scellé : 0 fenêtre explosée dans bf16 (sinon le préfixe n'a pas pris) · classement GLM = médianes ≤ 1,02 ET explosées ≤ bf16 · Coder : écart géo/médiane < 0,005
mesuré : **0 fenêtre explosée, tous bras, tous corpus** (bf16 GLM privé par fenêtre : 7,1 … 26,7 ; la fenêtre 8 de la tranche 0 passe de 110 219 à 25,9) · preuve du préfixe : `premiers_ids_fenetre_0 = [154822, 154824, …]` dans les 24 JSON GLM · Coder géo/médiane : llama.cpp 1,0103/1,0125, acvram 1,0270/1,0285 (écarts 0,002 et 0,0015 : tenu) ; vLLM 1,155/1,134, TRT-LLM 1,231/1,237
verdict (× bf16, moyenne géométrique des 3 tranches ; classement sur la **médiane**, privé) :
    modèle  corpus   bras / converti (calibration)                 ×bf16 géo   ×bf16 médiane   explosées   classé (méd ≤ 1,02)
    GLM     privé    acvram -k48 W4A16 (défaut collect.py)          1,0156      **1,0289**      0           non
    GLM     privé    llama.cpp Q4_K_M unsloth (imatrix non publié)  1,0248      **1,0152**      0           oui
    GLM     privé    vLLM GadflyII NVFP4 KV fp8 (non publié)        1,0734      1,0779          0           non
    GLM     public   acvram -k48                                    1,0040      **0,9972**      0           oui (et sous bf16 en médiane : signalé, non interprété)
    GLM     public   llama.cpp Q4_K_M                               1,0397      1,0266          0           non
    GLM     public   vLLM GadflyII                                  1,0740      1,0640          0           non
    Coder   privé    llama.cpp Q4_K_M (imatrix non publié)          1,0103      1,0125          0           oui
    Coder   privé    acvram nvfp4 (aucune ; MMA=1 = W4A4 prefill)   1,0270      1,0285          0           non (Hadamard le rejugera)
    Coder   privé    vLLM ModelOpt (non publié)                     1,1554      1,1335          0           non
    Coder   privé    TRT-LLM ModelOpt, KV fp8 (non publié)          1,2313      1,2367          0           non
    bf16 (géo/méd des tranches) : GLM privé 13,08/14,51, GLM public 8,58/8,18, Coder privé 11,74/11,55.
**GLM se juge enfin : llama.cpp classé sur le privé (1,015), acvram non (1,029) — géo et médiane s'inversent entre ces deux bras (acvram 1,016 en géo), l'écart tient à trois fenêtres ; sur le public acvram reste sous bf16 en médiane (0,997) et à 1,004 en géo — divergence privé/public d'acvram 0,032 > 0,02, signalée sans interprétation. Coder inchangé : llama.cpp seul classé.**

## Bornes
- llama.cpp : 1 023 cibles par fenêtre (KL : `first = n_ctx/2`, cibles 1025..2047) contre 1 024 ailleurs ; log-probs quantifiés uint16 (plancher 16 nats : 12 pertes saturées sur 12 288 dans la tranche 0 GLM privé) — Coder llama.cpp 1,0103 ici contre 1,0126 par `llama-perplexity` v1 hier (même binaire, autre lecture) : le KL sous-estime légèrement les pertes ≥ 16 nats.
- « Publié tel quel dans le journal » (Sage) : complétion brute GLM effondrée depuis toujours, silencieuse — le gabarit de chat met `[gMASK]<sop>`, `encode(add_special_tokens=True)` ne le met pas (vérifié) ; correctif serveur 505f31c (Laurine).
- Arbitre prefill / PPL en mode décodage / duel : à refaire avec `encode_brut` par séquence (Sage § 10.2) — hors de ce verdict.
