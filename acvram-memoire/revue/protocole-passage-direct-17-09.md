# Protocole — passage direct NVFP4 (Laurine 70b5279) : PPL acvram vs vLLM sur les MÊMES poids (scellé 2)

Instrument : `acvram convert --passage-direct --no-awq` sur `GadflyII/GLM-4.7-Flash-NVFP4` (compressed-tensors) → `GLM-4.7-Flash-vllm-direct` ; PPL `ppl-acvram-17-09.py` (préfixe `encode_brut`, cibles 1024..2047, géo + médiane) en W4A16 (`ACVRAM_MOE_MMA=0`) et W4A4 (`ACVRAM_MOE_MMA=1`) ; vLLM `ppl-vllm-17-09.py` sur le checkpoint source, **KV auto (bf16)** — le fp8 du duel est un régime de décodage, pas de PPL prefill —, mêmes ids (préfixe, 4 premiers publiés) ; corpus privé 5909d27 et public, 3 tranches chacun · commit : arbre laure a0f0776 (= main 565d686) · régime : une carte, prefill seul.

## Scellés
- S2 (Laurine/Sage) : |PPL acvram W4A16 / PPL vLLM − 1| ≤ 0,004 sur chaque corpus (moyenne géométrique des 3 tranches). Tenu → le format est le même et le noyau W4A16 est exact ; > 0,004 → l'écart est dans le noyau (W4A16), pas dans les poids — résultat valable.
- S2' (mien) : W4A4 (MMA=1) − vLLM ≤ +0,015 (le duel W4A4 coûtait +1,4 % de PPL) ; les deux bras acvram diffèrent entre eux de > 0,004 (bras qui doit différer : sinon MMA n'a pas pris).
- Témoin de format : `premiers_ids_fenetre_0` identiques dans les trois bras ; les couches en clair (attention MLA, routeur, tête) suivent le plan acvram (int8/bf16) — c'est la seule différence de poids restante, à dire dans le verdict avec les comptes du manifeste.
