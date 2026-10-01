# Réponses duck.ai — 01/10/2026 — p81 (poste1) : CAKE, arXiv 2608.12629

poste4 · Question p81 · source primaire arXiv (HTML lu directement) + recoupement Gemma 4 31B
(web search, alphaxiv.org + notes tierces) — concordants, pas de désaccord à signaler.

## Réponses, dans l'ordre

**(1) Code publié ?** Oui, deux PR GitHub distinctes, citées texto par le papier :
- prefill KDA : `github.com/flashinfer-ai/flashinfer/pull/4262`
- decode KDA : `github.com/flashinfer-ai/flashinfer/pull/4279`

« The generated CUDA is available in FlashInfer PR #4262 [...] so downstream users take on no
dependency on Cake. »

**(2) GPU mesuré ?** **B200 seulement** pour le ×2,05 — « across six B200 BF16 shapes ». Le
papier dit cibler « NVIDIA GPUs from Ampere through Blackwell » en général (portée du
compilateur), mais ce chiffre précis n'est mesuré que sur B200. **Rien trouvé sur sm_120/RTX
5090** — ni dans le papier ni dans les sources tierces recoupées par Gemma.

**(3) Prefill, décodage, ou les deux ?** Le ×2,05 porte **sur le prefill par blocs
uniquement** : « A FlashKDA-compatible prefill covers fixed, packed-variable, and tail inputs and
reaches a 2.05× geometric-mean speedup over that baseline across six B200 BF16 shapes. » Le
décodage est un chiffre **séparé et plus faible** : « Separate decode paths reach a 1.14×
geometric mean over upstream FlashInfer across 30 public-API shapes. »

**(4) Comparé à quoi ?** Les deux chiffres n'ont **pas le même témoin** : prefill ×2,05 contre
**FlashKDA officiel** (« Official FlashKDA is used only as a black-box timing baseline; its
source and generated code are not provided to the agent »). Decode ×1,14 contre **FlashInfer
amont** (upstream), pas contre FlashKDA. **fla (chunk_kda/fused_recurrent_kda) n'est mentionné
nulle part** dans le papier pour KDA.

**(5) Au bit, ou tolérance ?** « It is bitwise correct on its validation contract and was
verified in end-to-end Kimi-K3 serving under SGLang. » — au bit sur son contrat de validation,
plus un contrôle de bout en bout en service réel (Kimi-K3/SGLang), pas seulement un test isolé.

## Verdict sur les critères de réfutation posés par poste1

- Pas de code → **réfuté** : code publié (PR #4262, #4279).
- Hopper/B200 seulement → **réfuté à moitié, à nuancer** : le papier ne teste QUE B200 pour ce
  chiffre ; aucune mesure sm_120/RTX 5090 trouvée nulle part (ni papier ni sources tierces) —
  **le levier n'est pas démontré transférable à notre carte sans mesure locale**.
- Aucun chiffre en décodage → **réfuté** : 1,14× décodage existe, mais c'est un chiffre
  nettement plus faible que le ×2,05 (qui est prefill seul) et contre un témoin différent
  (FlashInfer, pas FlashKDA) — **à ne jamais citer le ×2,05 pour un gain de décodage**.

**RESTE** : rien en cours après cette réponse. Pointeur envoyé à chef et poste1.
