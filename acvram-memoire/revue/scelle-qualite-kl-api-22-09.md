# Scellé — qualité W4A4 par KL relative, par API (Laure, 22/09, avant mesure)

Demande Océane (dbf5032a) : PPL/KL relative de TRT-LLM (W4A4) avant de courir après
son débit. Par API sur les serveurs OpenAI-compatibles : `/v1/completions` avec
`logprobs` (+ `echo`), invites decode-pas TEXTE, 8 pas gloutons, KL max par pas.

## Instrument (à sec, testé)

`scratchpad/trtllm-cellules-22-09/kl-api.py` : (1) génère N=8 jetons gloutons sur la
référence ; (2) teacher forcing par `echo=True`, `logprobs=K`, `max_tokens=0` sur
chaque serveur, mêmes jetons ; (3) KL(ref‖moteur) par pas sur l'union des top-K,
KL max. Testé à sec sur faux serveurs logprobs (biais 0,4 → KL 0,1534 cohérente).

## Prédiction (Maîtresse)

W4A4 (trtllm) : **KL max 0,5-2 nat** contre nvfp4 W4A16 (acvram) **~0,9**.
Repères scellé E (gemma-31B, réf bf16 HF local) : A=0,867, B=1,394, témoin cassé 2,12.

## Décision (Maîtresse 22/09)

- Qualité jugée sur **Qwen3-Coder-30B-A3B** (le modèle des cellules).
- **Aucune source bf16 du Coder sur disque** au moment du run → KL **RELATIVE**
  TRT-LLM (W4A4 hub) ↔ acvram (**nvfp4-qkvo-i8c**, celui des cellules), teacher
  forcing sur les **8 jetons gloutons d'acvram** (acvram = pivot `--ref`), **sans
  référence absolue** — limite écrite EN TÊTE du verdict.
- 3e bras optionnel : llama.cpp (GGUF de T1) s'il est servi dans le même créneau
  (pas une référence non plus).
- **Référence absolue bf16** : download 60 Go `Qwen/Qwen3-Coder-30B-A3B-Instruct`
  lancé en fond (feu utilisateur), vers `…/Qwen3-Coder-30B-A3B-Instruct-bf16-hub/`.
  Une fois là, servi par acvram (étagé/exil) = 3e/4e bras, référence absolue de la KL.
- Corpus : `invites-kl-texte.txt` (5 invites de programmation, texte seul).

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

## BLOCAGE (22/09, 08:20) — acvram serve ne renvoie pas de logprobs

Tentative KL par API arrêtée dès la 1re capture : acvram serve rend « token_ids/echo
non rendus ». Cause vérifiée dans le code (pas le client) :
- `acvram/server/app.py:1116-1117` : la réponse `/v1/completions` non-stream construit
  `CompletionChoice(text=text, finish_reason=reason)` **sans jamais peupler `logprobs`**.
- `app.py:1111` : `echo` se contente de préfixer le texte du prompt (`text = prompt_text + text`),
  il ne retourne AUCUN logprob par position.
- Le champ `logprobs` existe au schéma (`protocol.py:302`) mais n'est jamais rempli.

Donc **la KL par API (teacher forcing echo + top-K logprobs) est impossible côté
acvram** : le serveur ne calcule/expose pas les logprobs. Ce n'est pas résoluble par
le client. Options (choix Maîtresse) :
- **(a) KL en LOCAL** via `decode-pas.py` (accès direct aux logits d'acvram, comme le
  scellé E) : donne acvram-vs-bf16 nativement ; trtllm-vs-bf16 demanderait l'API python
  TRT-LLM (`generate` avec logits de génération), un instrument séparé.
- **(b) Ajouter les logprobs à acvram serve** : peupler `CompletionChoice.logprobs`
  (+ `echo` renvoyant les logprobs du prompt), avec test — modif serveur, rend la KL
  par API possible pour tous les moteurs OpenAI.
- **(c) Renoncer à la KL par API** ; juger la qualité W4A4 par PPL locale ou decode-pas.

Le débit (cellule + b=1 + débit(b)) est complet et publié ; seule la qualité KL bute
sur cet instrument. kl-api.py + kl-3bras.sh restent prêts pour l'option (b).
