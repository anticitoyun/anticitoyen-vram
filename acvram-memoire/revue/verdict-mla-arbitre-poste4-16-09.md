# Verdict — arbitre prefill sur poste4 f4607c2 : C 81/84, D (fp8) 81/84, boucle 80/84 → CONFORME (≥ 80, cos ≥ 0,9999 en médiane) ; fp8 : logits du décodage identiques à C, donc sans effet sur ce que le chemin mesuré lit

- **instrument** : `arbitre-prefill-mla-16-09.py` (prefill W4A16 rejoue prompt 128 + k jetons greedy du bras, logit comparé au logit du décodage à k ; k ∈ {1,2,4,8,16,32,63}, 12 séq = 84 points) ; ties W4A16 `MOE_DECODE_MMA=0` (régime GLM de poste7 § 9) ; sorties `scratchpad/arbitre-poste4-16-09/`
- **commit** : arbre mesuré **travail/poste3-qa @ f4607c2** (poste4 : batch, une passe, prep, latent fp8 OFF) ; bras **C** défaut, **D** `ACVRAM_MLA_LATENT_FP8=1`, **B0** `ACVRAM_MLA_BATCH=0` (boucle, témoin sur le même arbre)
- **régime** : `-k48`, b=12, prefill W4A16, décodage W4A16, graphes actifs pour les ties, eager pour l'arbitre
- **scellé** (poste7 § 9) : ≥ 80/84 top-1 et cos ≥ 0,9999 ; témoins main : boucle 83, défaut 81
- **mesuré** : **C 81/84**, cos min 0,99997 méd 1,000000, |Δ| méd 1,13 ; **D 81/84**, identique à C point par point (les 84 |Δ| et top-1) ; **B0 80/84**, cos min 0,99905 méd 1,000000, |Δ| méd 1,17
- **verdict** : **CONFORME** — C ≥ 80 et au niveau du défaut de main (81) ; la boucle sur cet arbre est à 80 (cos min 0,99905 < 0,9999 sur un point : la boucle est le bras le plus éloigné du prefill, pas le plus proche). **Fusion possible, 17 ms acquis** (les temps de mla1-3 sont ceux d'un chemin aussi juste que le défaut de main). D : à **trancher NON par la PPL mais par le fait** que le fp8 ne change aucun logit du décodage mesuré (§ 1)

## 1. Point (1) de poste7 — le fp8 s'applique-t-il à ce que le noyau lit ? Ce que j'ai, sans conclure
- Faits : `mla_1p_kernel<FP8=true>` lancé 89/pas en D (nsys), cache `DecoderBlock.statics[0]` uint8 bien écrit (échelles, |déq| = |bf16| à 10⁻³) ; **logits du décodage bit-identiques C/D** (ties 768 positions, arbitre 84 points, PPL 3 tranches) — un E4M3 par ligne réellement lu ne peut pas rendre cela.
- `DecoderBlock.statics` : **1 seul créneau par bloc**, `len` 511 sur les 47 blocs, après 12 séquences × 100 pas — alors que `graphs.py:520-537` lie un créneau par slot (`static_bind(slot, sid, …)` pour `slot in range(b)`). Les 12 séquences ne vivent donc pas dans les objets que j'ai inspectés : soit `GraphRunner.model` n'est pas `loaded.model` (à vérifier : `runner.py:327`, `graphs.py:164`), soit les couches MLA de GLM ne passent pas par `_la_decode` et le noyau `mla_1p` que je vois est appelé d'ailleurs.
- Je n'ai pas trouvé en 10 min à sec **d'où** `decode_static_batch_complet` reçoit ses 12 créneaux ; c'est la question exacte pour poste4 : *quel objet porte les 12 `statics` du lot, et `new_static` y est-il appelé avec `_MLA_LATENT_FP8` lu ?* Tant que ce n'est pas répondu, « E4M3 sans effet à ctx 300 » n'est pas un résultat de qualité, c'est un chemin non exercé — et le contrôle à ctx 2 048 ne dirait rien de plus.

## 2. Ce que je retiens pour la suite
Le scellé de commit (REGLES § 4 bis) : arbitre prefill ≥ 80/84 sur prompt réel + un bras qui DOIT réagir. Ici le bras D aurait dû réagir et n'a pas réagi : c'est lui le drapeau, pas la PPL.
