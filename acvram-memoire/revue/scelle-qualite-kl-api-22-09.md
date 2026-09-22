# Scellé — qualité W4A4 par KL relative, par API (poste3, 22/09, avant mesure)

Demande poste1 (dbf5032a) : PPL/KL relative de TRT-LLM (W4A4) avant de courir après
son débit. Par API sur les serveurs OpenAI-compatibles : `/v1/completions` avec
`logprobs` (+ `echo`), invites decode-pas TEXTE, 8 pas gloutons, KL max par pas.

## Instrument (à sec, testé)

`scratchpad/trtllm-cellules-22-09/kl-api.py` : (1) génère N=8 jetons gloutons sur la
référence ; (2) teacher forcing par `echo=True`, `logprobs=K`, `max_tokens=0` sur
chaque serveur, mêmes jetons ; (3) KL(ref‖moteur) par pas sur l'union des top-K,
KL max. Testé à sec sur faux serveurs logprobs (biais 0,4 → KL 0,1534 cohérente).

## Prédiction (chef)

W4A4 (trtllm) : **KL max 0,5-2 nat** contre nvfp4 W4A16 (acvram) **~0,9**.
Repères scellé E (gemma-31B, réf bf16 HF local) : A=0,867, B=1,394, témoin cassé 2,12.

## Points de conception à TRANCHER avant la mesure carte

1. **Référence bf16** : le scellé E était gemma-31B via HF LOCAL. Par API il faut un
   serveur bf16 du MÊME modèle que les cellules (**Qwen3-Coder-30B**). L'alias bf16 30B
   servi comme témoin (60 Go) est-il celui-là ? Sinon, sur quel modèle juger.
2. **Modèle** : cellules = Qwen3-Coder-30B ; scellé E = gemma-31B. La qualité doit se
   juger sur le MÊME modèle (Qwen3-Coder-30B : trtllm W4A4 vs acvram nvfp4 vs bf16).
3. **Support echo/logprobs/token_ids** : non garanti que acvram ET trtllm rendent
   `token_ids` + `top_logprobs` avec `echo=True, max_tokens=0`. Le client a un garde-fou
   (erreur nommée si absent) ; à vérifier au 1er contact serveur.
4. **Limite nommée** : KL sur top-K tronqué (K≤5, plafond OpenAI /v1/completions) =
   borne INFÉRIEURE de la vraie KL. Suffit pour classer aux seuils 0,5/1,0/1,2.

Rien sur carte tant que 1-2 ne sont pas tranchés (quel bf16, quel modèle).
