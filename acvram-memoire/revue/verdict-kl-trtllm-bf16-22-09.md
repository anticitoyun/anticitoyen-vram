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

## Suite

Relance ≤ 10 min au créneau après la 5e chaîne de poste2 (invite4 + chaîne ~35 min),
sur son « carte rendue » et `nvidia-smi` propre (seul le llama-server 4453 de
l'utilisateur). Si l'alignement passe (≥ 0,90) sur les 4 : verdict qualité complet
(kl_max W4A4 vs 0,519 nvfp4 vs seuil 1,2). Sinon : l'instrument est déclaré non
concluant et le bras trtllm/bf16 reste ouvert.
