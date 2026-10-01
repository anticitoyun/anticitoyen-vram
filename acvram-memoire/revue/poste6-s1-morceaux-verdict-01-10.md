# kv31b levier 2, étape 1 — preuve carte S1 : les morceaux ne sont PAS au bit du seul tenant en service (poste6, 01/10, branche poste6-reserve-attention)

instrument : `outils/gpu/mesure/prise-s1-morceaux-kv31b.sh` (un processus par bras, HEAD asserté, relevés nvidia-smi début/fin, journal lu par ses lignes `[acvram]`) ; `s1-morceaux-comparer.py` (ids par sha256, Δ des logprobs ; texte jamais lu, § 6) ; `/v1/completions` glouton 32 jetons, `logprobs=10`, `ACVRAM_SAMPLER_LENT=1`, `--speculative none`, ctx 10 240
commit : 72521cb1b (S1 principale) ; 168e20156 et 462805072 (bras d'isolement : `OPTIONS_SERVE`, `REQUETES=2`) — moteur inchangé entre les trois
régime : RTX 5090 seule (le llama-server 4436 vit sur la 3080 Ti, bus 02:00.0) ; 29,5 Gio libres lus par le plan ; gemma-4-31B : 8/60 MLP exilés (régime DÉGRADÉ, graphes off), kv=int8 ; Devstral-24B : 0 exilé, graphes on, kv=int8 ; CPU : rtk 33 %, charge 0,46
scellé : `poste6-s1-morceaux-scelle-carte-30-09.md` (P1-P6, bras A1/A2/B)
mesuré : invite `87d8bfe0a:README.md` (sha 0640377149ab8d46…), 7 953 jetons gemma / 7 865 Devstral ; 3 bras × 5 chaînes (S1 à 63fb2efe8, 0838f4ca8, 72521cb1b ; Devstral ; gemma sans cache de préfixe) + 2 témoins « reprise » + 2 chaînes KV bf16 refusées
verdict : **S1 FAUX (P3)** — avec un témoin A1/A2 au bit (Δ = 0 sur 350 valeurs, P2 tenu), les morceaux divergent dès le premier jeton généré : gemma Δ logprob 0,60 (5/10 top-10 communs, argmax changé) ; sans cache de préfixe Δ 0,088 (Δmax 0,29 sur 32 pas, ids identiques) ; Devstral dense Δ 0,041 (10/10 communs, argmax changé sur une quasi-égalité à −3,19/−3,15), soit 10 × le témoin reprise de Devstral (0,004) — au-delà des 2 × du scellé. Deux défauts de l'étape 1 trouvés et corrigés en route (ci-dessous) ; l'option reste OFF
durée : 5 chaînes de 3 prises + 4 prises isolées, 13-36 s chacune (prêt en 21-48 s), 04:57-05:09 ; prévu ≤ 8 min par prise, tenu par `carte.sh`

## Ce que la première prise a révélé (63fb2efe8 : P1 faux, `prefill_morceaux` = 0 en B)
1. **La passe jusqu'à la frontière d'instantané contournait les morceaux** (runner.py:1909) : gemma-4 est `est_hybride` (couches typées) et, cache de
   préfixe actif, le préfill est coupé à 7 936 sur 7 953 — presque toute l'invite passait d'un seul tenant. Corrigé 0838f4ca8 : cette passe aussi par
   `_prefill_morceaux` ; test jouet (0 avant, 1 après, au bit sur processeur).
2. **Poids en flux (MLP exilés) × `forward_tranches`** : la seconde tranche d'une couche reprenait un créneau du bassin de flux déjà réattribué —
   `down_proj` servi avec un poids de forme gate/up (`mat1 4096×21504, mat2 5376×21504`, chauffe B à 0838f4ca8, serveur mort). Corrigé 72521cb1b :
   `tranches_possibles` rend faux dès qu'un `QuantLinear.streamed` existe → une tranche après l'autre par `forward` ; test.
   Conséquence : sur gemma (8 exilés) la preuve porte le repli séquentiel ; sur Devstral (0 exilé) le chemin `forward_tranches` — les deux divergent.

## Chiffres (B contre A1 ; A1 = A2 partout)
| chaîne | modèle | cache préfixe | `prefill_morceaux` B | ids 32 | Δ lp pos 0 | top-10 communs pos 0 | Δmax lp |
|---|---|---|---|---|---|---|---|
| S1 72521cb1b | gemma-4-31B, 8/60 exilés | on (frontière 7 936) | 3 (2 de chauffe + 1) | diffèrent dès 0 | 0,601 | 5 | — |
| gemma sans cache | idem | off | 2 | identiques | 0,088 | 8 | 0,29 sur 32 |
| Devstral-24B | dense, 0 exilé | on | 4 | diffèrent dès 0 | 0,041 | 10 | — |
| témoin reprise Devstral (même requête 2 fois, 9 712 jetons servis par le cache) | | | 0 | diffèrent dès 0 | 0,004 | 9 | — |
| témoin reprise gemma | | | 0 | identiques | 0,000 | — | 0 |
P1 : prise prouvée (B > 0, régime `(morceaux@4096)`), mais le compte « = 1 » du scellé ignorait la chauffe (2 essais à 10 240 → 2 morceaux) : faux
à la lettre, dit. P4 tenu (7 953 / 7 865, égaux). P5 tenu (même exil dans les 3 bras ; le comparateur somme les lignes « de plus » (23), le
régime dit 8/60 — à corriger dans l'outil). P6 tenu.
Non exploitables : KV bf16 refusé sur gemma ET Devstral (« 640 blocs KV nécessaires, 322-325 en tout » : à 10 240, le budget bf16 plafonne à
5 200 jetons alors que 7 900 jetons de Devstral en bf16 font 1,2 Gio — défaut de plan à nommer, hors pièce) ; témoin reprise gemma : le cache de
préfixe n'a servi aucun jeton à une requête identique (`cached_prompt_tokens` 0, `cache_prefixe=0.000`) — sur un modèle `est_hybride` la reprise
ne passe que par les instantanés, et il n'y en a pas sans état récurrent : à nommer aussi.

## Lecture
* Le jouet était au bit avec un cache bf16 et à 6-9e-3 avec un cache int8 (4 couches). En service le cache est int8 : les morceaux relisent les
  K/V quantifiés de 4 096 jetons pour 3 769 lignes de requête, et l'erreur se compose couche après couche (40-60) — 0,04 sur Devstral est
  compatible avec cette seule cause ; le témoin reprise (0,004) ne relit l'int8 que pour UNE ligne, il n'est pas la même quantité d'erreur.
* gemma : 0,088 sans cache, 0,60 avec — l'écart avec cache dépasse ce que l'int8 explique ; l'interaction frontière d'instantané (7 936) +
  morceaux reste suspecte (le B avec cache rend exactement les ids du A sans cache : d3449ae4…).
* Rien de tout cela ne se tranche sans un bras KV bf16 sur carte (refusé par le budget) : issue (a) du scellé — « morceaux non au bit sur
  carte, cause non séparée » — c'est l'issue qui me gêne, et c'est celle qui sort.

## Reste
* Levier 2 étape 1 : OFF par défaut, inchangé ; les deux correctifs (frontière, flux) sont nécessaires quelle que soit la suite.
* Pour séparer int8 et chemin : (1) corriger le budget KV bf16 (325 blocs à 10 240) puis rejouer Devstral B/A en bf16 — au bit attendu si
  l'int8 est seul en cause ; (2) jouet processeur int8 à 4, 8, 16 couches : l'écart doit croître avec la profondeur ; (3) gemma avec cache :
  bras `ACVRAM_INSTA_PAS` grand (frontière au-delà de l'invite) pour isoler la frontière.
* Levier 2 étapes 2-4 (anneau) : suspendues tant que l'étape 1 n'est pas au bit ou acceptée « comme une reprise » par chef.
* Outil : comparateur à lire `couches_exilées=` du régime plutôt que la somme des « de plus ».
