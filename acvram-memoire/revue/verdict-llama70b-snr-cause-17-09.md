# Verdict — Llama-3.3-70B, écart PPL 22,05 contre 11,65 (+89 %) : conversion NON en cause, deux vérifications directes, à sec

Manon, 17/09. Suite de `verdict-palier2-llama-70b-17-09.md` (Laure) : « à
Manon (SNR du converti) avant toute lecture ». Deux hypothèses nommées par
Laure — la conversion `srci1_Q4_K_M → NVFP4`, ou la voie d'exil — vérifiées
séparément, à sec, aucune carte.

## 1. SNR du converti : signature normale, pas d'anomalie

`Llama-3.3-70B-Instruct-heretic-srci1_Q4_K_M-nvfp4/acvram_manifest.json`,
options `awq: false` (round-to-nearest, comme le `--no-awq` de Qwen3.8) :

```
561 tenseurs avec out_snr_db : min 19,95 / médiane 20,44 / max 30,72 dB
aucun tenseur sous 15 dB
```

C'est EXACTEMENT la signature round-to-nearest déjà vue sur Qwen3.8-27B
`--no-awq` (20,5 dB moyen) — qui n'a coûté que +3 % de PPL, pas +89 %. Le
SNR de l'étape de quantification NVFP4 ne montre rien d'anormal : si
c'était la seule cause, l'écart serait du même ordre que Qwen3.8, pas 30×
plus grand.

Mais `out_snr_db` mesure le bruit NVFP4 contre son parent IMMÉDIAT (le
tenseur déquantifié du GGUF), pas contre les poids bf16 d'origine — un
défaut de la lecture GGUF resterait invisible à cette mesure. D'où la
vérification 2.

## 2. Lecture GGUF (Q4_K, Q5_K, Q6_K) : bit-exacte contre la référence officielle

Comparaison directe, sur le VRAI fichier de la mesure de Laure
(`Llama-3.3-70B-Instruct-heretic.i1-Q4_K_M.gguf`), entre
`acvram/quant/gguf.py::_dq_q4_k/_dq_q5_k/_dq_q6_k` et
`gguf.quants.dequantize` (dépôt officiel `llama.cpp`, `gguf-py`, cloné à
`/mnt/AI_GENERATOR/llamacpp/officiel-src`) — les trois types réellement
présents dans ce fichier (441 Q4_K, 81 Q6_K, 40 Q5_K, 162 F32 pour les
normes) :

```
token_embd.weight (Q4_K, 1 050 673 152 valeurs)  : écart max 0,0 — identique au bit
blk.0.attn_v.weight (Q6_K)                        : écart max 0,0 — identique au bit
blk.0.ffn_down.weight (Q6_K)                      : écart max 0,0 — identique au bit
blk.10.attn_v.weight (Q5_K)                       : écart max 0,0 — identique au bit
```

Les formules de `_kscales`/`_dq_q4_k` correspondent bit à bit à
`get_scale_min_k4`/`dequantize_row_q4_K` de référence (vérifié aussi par
la mesure, pas seulement par lecture de code — REGLES §7).

## Verdict : les deux étapes de conversion sont innocentées, la cause reste à chercher côté exil

Ni la quantification NVFP4 (SNR normal, même signature qu'un converti qui
ne coûte que 3 %) ni la lecture GGUF (bit-exacte sur les trois types
réellement utilisés) n'expliquent un écart de +89 % entre le converti et
sa source, sur les mêmes fenêtres. Reste l'hypothèse que Laure nomme en
second : la voie d'exil (57/80 MLP en hôte sous l'instrument PPL) — déjà
le siège d'un bogue distinct et déjà nommé par elle (`loader.py:1117`,
budget KV borné à 0 après exil, garde 410dd07). Un exil qui corrompt ou
mélange les poids MLP hôte↔carte, ou qui lit la mauvaise couche exilée,
expliquerait un écart de cet ordre sans que la conversion elle-même soit
en cause.

Pas de carte utilisée. Prochaine étape (hors de ce contrôle, à décider
par Sage/Jérôme) : instrumenter la voie d'exil elle-même — comparer,
couche par couche, les poids réellement lus pendant le prefill PPL contre
le manifeste (résidentes vs exilées), avant de soupçonner un mécanisme
plus profond.
