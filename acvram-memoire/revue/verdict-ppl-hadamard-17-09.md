# Verdict — converti Hadamard (poste2, `-hadamard512`) : PPL réfutée (privé 1,042, public 0,986), temps 22,6 ms, capture rétablie

instrument : `ppl-acvram-17-09.py` (prefill, fenêtres `encode_brut`, médiane, 4 ids) ; `ppl-decode-prefixe-17-09.py` (12/12, graphes) ; `certifie-b12-15-09.py` ; `sonde-capture-17-09.py` (captures/rejeux) ; seconde référence bf16 : vLLM délesté (`ppl-vllm-17-09.py`, `cpu_offload_gb=42`, KV bf16) sur les 6 tranches — journaux `scratchpad/ppl-hadamard-17-09/`
commit : prefill sur l'arbre 40ec541 (avant le correctif 59f2595, bit-identique selon poste4) ; décodage, temps, capture sur 5e10c17 (= main a311da2) ; protocole `protocole-ppl-hadamard-17-09.md`
régime : `GLM-4.7-Flash-srcbf16-nvfp4-hadamard512` (AWQ + rotation experts 512, NOMINAL 0 exilé), W4A16 prefill+décodage, MIN_T=5, SLOTS=12, graphes ; préfixe `[gMASK]<sop>` (ids [154822, 154824, …] dans chaque JSON) ; corpus privé 5909d27 et public wiki-gptq, 3 tranches, cibles 1024..2047 (prefill) / 1..2047 (décodage)
scellé : PPL médiane ≤ 1,010 × bf16 (privé ET public) · temps b=12 ≤ 19,0 ms · capture 10/10 · mes prédictions : privé [1,010 ; 1,022], public [0,995 ; 1,005], décodage ≈ prefill
mesuré : **prefill privé 1,0421** (× bf16 HF, médiane ; 1,0311 × bf16 vLLM), **public 0,9856** (0,9894 × vLLM) ; **décodage privé 1,0342** (× bf16 HF min-ctx 0, 12/12, 24 564 cibles) ; **temps 22,569 / 22,602 ms**, 485 t/s, 0,783 J/j (bridage puissance) ; capture : `warm_graphs` 3 + 1 = 4 clés, **66 rejeux pour 63 pas, aucun repli eager**, régime NOMINAL graphes=on
verdict : **le converti Hadamard est réfuté sur les deux scellés : PPL privée 1,042 (pire que le `-k48` à 1,029) avec public 0,986 (sous bf16, plus que le `-k48` à 0,997) — la rotation ÉLARGIT l'inversion privé/public (écart 0,057 contre 0,032) ; temps 22,6 ms contre 21,0 (la rotation des activations à l'exécution coûte +1,6 ms). La capture des graphes est rétablie par 59f2595 (100 % des pas rejoués). Ma prédiction privée [1,010 ; 1,022] est réfutée aussi (par le haut).**

## Ce que ça dit de l'inversion (piste RTN experts)
    converti              AWQ    rotation   privé (méd ×bf16 HF)   public   écart
    -sansawq              non    non        1,0207                 1,0055   0,015
    -avant-alpha (AWQ hors experts)          1,0239                1,0027   0,021
    -k48                  oui    non        1,0289                 0,9972   0,032
    -hadamard512          oui    512        1,0421                 0,9856   0,057
L'écart croît avec ce qui adapte les poids à une distribution : RTN seul 0,015, AWQ (défaut `collect.py`, 230 jetons d'anglais) +0,017, rotation +0,025 de plus. Lecture la plus simple : l'AWQ et la rotation sont réglées sur l'anglais du texte de calibration par défaut — elles améliorent wikitext (jusqu'à passer SOUS bf16) et dégradent le français technique. Ce n'est pas prouvé ; le test qui prouverait : le même converti calibré sur un texte français (ou mixte, ou sur le privé lui-même — interdit pour l'évaluation, autorisé pour le test) : si l'inversion s'inverse, c'est la calibration. À poste2/poste7.

## Bornes
- Les deux références bf16 concordent par fenêtre (tranche 0 privée : ± 1 %) mais leurs **médianes** diffèrent de 1,1 % sur le privé (géo 0,1 %) : la médiane de 12 fenêtres saute d'une fenêtre à l'autre, elle est plus bruyante que la moyenne géométrique — un scellé à ± 0,01 sur la médiane est à la limite de ce que l'étalon tient. Les deux verdicts (réfuté) tiennent avec l'une comme l'autre référence.
- Prefill mesuré sur 40ec541 (avant 59f2595) : le correctif ne touche que la copie hôte→carte d'un scalaire dans `fwht_activations`, bit-identique (test capturé == eager de poste4) ; non remesuré.
- La sonde de capture compte 4 clés pour un décodage de 64 jetons à b=12 sous `max_len=1024` (pas 10 godets : les godets dépendent des longueurs atteintes) ; la preuve est « aucun repli, rejeux ≥ pas ».
