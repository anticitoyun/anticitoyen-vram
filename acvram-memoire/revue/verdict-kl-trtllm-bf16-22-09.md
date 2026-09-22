# Verdict — KL(bf16 ‖ trtllm W4A4), qualité, PARTIEL (poste3, 22/09)

Bras TRT-LLM/bf16 du croisement qualité (poste1 dbf5032a) : KL par pas contre le
bf16 (dumps PHASE_HF de poste2, teacher forcing sur les 8 gloutons bf16), par l'API
python TRT-LLM (`return_context_logits`), instrument `kl-trtllm-local.py`.
Complément du bras acvram/bf16 (poste2 : TENU 4/5, kl_max=0,519).

## État : PARTIEL — 1 invite sur 4

- **invite0** : kl_par_pas [0,011 · 0,0002 · 0,015 · **0,965** · 0,173 · 0,243 · 0,029 · 8e-5], **kl_max = 0,965**.
- **invite1-3** : `IndexError` — trtllm renvoie sur certaines longueurs un
  `context_logits` plus court que `len(prompt)−1`, l'indexation `n_prefix−1+k`
  dépasse (index 35, size 35). Pas un problème carte/trtllm : calage d'offset.

## Lecture (indicative, 1/4 — NON concluant)

Sur invite0, trtllm-W4A4 : kl_max 0,965 — **sous** le seuil scellé E (≤ 1,2) mais
**au-dessus** du bras acvram-nvfp4 (0,519, TENU 4/5). Sur ce seul point, W4A4
diverge davantage du bf16 que nvfp4. Une seule invite ne tranche rien : le verdict
qualité attend les 4.

## Correctif (à sec, poussé) — non relancé sans feu

- Offset calé par la fin des context_logits quand `L < n_prefix−1+N`, `L`/prompt/offset imprimés.
- **Contrôle qui rend faux** (chef) : sur les positions de préfixe, l'argmax des
  context_logits doit retrouver le token suivant du prompt (taux ≥ 0,90). Un offset
  décalé effondre ce taux (~0) ; en dessous du seuil, l'alignement est SUSPECT et le
  kl_max de l'invite est EXCLU de l'agrégat. Le taux `argmax_prefixe` est imprimé par invite.

## Verdict FINAL de l'instrument (3e essai, 15:57) : NON CONCLUANT

Le discriminant d'alignement a fait son travail — il a rendu FAUX. Sur les 4 invites,
le taux argmax(context_logits[i]) = prompt[i+1] est **≤ 0,31 pour tous les décalages
d ∈ {−2..+2}** (invite0 : 0,00/0,00/0,31/0,00/0,00 ; invite1-3 : 0,00 partout) —
très en dessous du 0,4-0,7 attendu d'un bon alignement. **Toutes les invites SUSPECT,
exclues ; kl_max_global = 0.** Les kl_max bruts (2,35 · 28,5 · 38,6 · 37,3) sont des
ARTEFACTS d'un alignement faux, **non publiables**. Le contrôle a empêché un faux
verdict « W4A4 diverge énormément ».

**Cause probable** : trtllm renvoie des `context_logits` tronqués/désalignés — 3/4
invites ont L < len(prompt) (35<38, 39<47, 37<40), signature du **prefix caching**
(`kv_cache_config enable_block_reuse=True` dans les LLM Args) : les positions dont
les blocs KV sont réutilisés ne sont pas recalculées, donc `context_logits` ne couvre
pas tout le prompt et l'indexation position→token est fausse. Même invite0 (L=38=prompt,
non tronquée) n'atteint que 0,31 : l'ordre ou la sémantique des context_logits n'est
pas [position i prédit token i+1] comme supposé.

**Correctif à tester (sur feu)** : relancer trtllm avec `enable_block_reuse=False`
(SamplingParams/LLM args) pour que context_logits couvre tout le prompt ; le
discriminant restera le juge (≥ 0,4 et écart ≥ 0,3). Si le taux ne monte pas, la
sémantique des context_logits diffère et il faudra une autre voie (logits de génération
d'une suite forcée jeton-à-jeton, ou la pièce 36 côté acvram une fois les logprobs servis).

**Bras trtllm/bf16 : OUVERT** — aucune KL trtllm publiable par cet instrument en l'état.
Le débit (cellule b=12, b=1, débit(b)) reste complet et publié.
