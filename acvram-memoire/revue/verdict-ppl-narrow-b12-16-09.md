# Verdict — voyant `narrow_gemm` b=12 : PPL graphes RÉFUTÉE (de peu), eager PASSE, divergence CONFORME

poste2, 16/09. Ordre poste7 (`poste7-narrow-b12-16-09.md`), amendant le voyant
b=1 (`verdict-ppl-narrow-16-09.md`, chemin jamais pris). Montage : 12
séquences en DÉCODAGE (un jeton/séquence/pas, jamais un prefill),
12 × 2047 = 24 564 jetons teacher-forcés (`outils/ppl-narrow-b12-
coder30b.py`), quatre bras : A/B sous graphes CUDA, A-eager/B-eager
(témoin et compte exact).

## Piège trouvé et corrigé AVANT la mesure valide

Premier essai : `load_model(max_model_len=2112)` réplanifie avec
`PlannerOptions.max_concurrent_seqs` par défaut (**8**, `tiering.py:152`,
utilisé ligne 447 : `kv_per_tok * max_model_len * max_concurrent_seqs`) —
budget KV dimensionné pour 8 séquences, pas 12. Résultat : 5 des 12
séquences finies `"length"` (`_grow` épuisant l'allocateur,
`loader.py:1013`) bien avant 2047 jetons (1408 à 2047, `n_jetons_notes`
22 872 au lieu de 24 564) — **silencieux, aucune erreur, un chiffre qui
aurait pu se publier faux**. Corrigé en construisant le plan nous-mêmes
(`auto_plan(..., PlannerOptions(max_concurrent_seqs=12))`) avant
`load_model(plan=...)` — `kv_max_tokens=25344 ≥ 24576` vérifié par
assertion. Les quatre mesures ci-dessous sont sur le plan corrigé,
`n_jetons_notes=24564/24564` sur les quatre bras.

## En-tête de mesure

Instrument : `outils/ppl-narrow-b12-coder30b.py`, une carte
(`CUDA_VISIBLE_DEVICES=0`), preuve de configuration publiée par bras
(`ACVRAM_NARROW_GEMM`, `eager`, `kv_max_tokens`, `lancements_narrow_gemm`).
Modèle `Qwen3-Coder-30B-A3B-nvfp4`, corpus `wiki-gptq.txt`.

## (1) PPL teacher-forcing — seuil scellé (chef) 1,000 ± 0,002

```
A       (graphes, NARROW=0) : ppl=8.498852
B       (graphes, NARROW=1) : ppl=8.518279   ratio B/A = 1,002286  RÉFUTÉ (> 1,002)
A-eager (NARROW=0)          : ppl=8.511008
B-eager (NARROW=1)          : ppl=8.505987   ratio B/A = 0,999410  PASSÉ
```

**Sous graphes CUDA (le chemin RÉEL de production), le seuil est dépassé
de 0,000286** — marginal, mais un seuil dépassé de peu reste dépassé
(REGLES : « un seuil qui laisse passer sa thèse de peu n'est pas un
seuil »/10-09). **En eager, PASSÉ avec marge** (0,00059 sous le seuil).
Écart non expliqué ce soir : narrow_gemm lui-même est cohérent en eager,
mais quelque chose dans l'interaction narrow_gemm × graphes CUDA (capture/
rejeu) coûte 0,29 % de PPL supplémentaire par rapport à eager — à poste4.

## (2) Preuve que le chemin a été pris

```
A/A-eager (NARROW=0) : 0 lancement narrow_gemm (attendu)
B (graphes, NARROW=1) : 1551 lancements — capture SEULEMENT (un noyau
    baké dans un graphe rejoué n'est PAS ré-invoqué par Python à chaque
    pas ; 1551 ≈ 8 passes de capture × 193 tenseurs int8 éligibles au
    format, PAS une mesure par pas)
B-eager (NARROW=1) : 196 512 lancements = EXACTEMENT 2047 pas × 96
    tenseurs éligibles PAR PAS (48 couches × 2 projections — pas les 193
    tenseurs int8 du manifeste : q_proj/o_proj qualifient, k_proj/v_proj
    non, cause non investiguée ce soir)
```

Le compte par pas exact (196 512, divisible net par 2047) est la preuve
demandée par poste7 ; sous graphes le compte ne peut être qu'un indicateur
de sélection à la capture, jamais un compte par pas — noté pour ne pas le
sur-interpréter la prochaine fois.

## (3) Taux de divergence top-1 — seuil scellé taux(B) ≤ 1,2 × taux(témoin)

```
B vs A (graphes)        : 1846/24564 = 0,075151
témoin A-graphes/A-eager : 1930/24564 = 0,078570
seuil (1,2 x témoin)     : 0,094284
VERDICT CONFORME (B est même EN DESSOUS du bruit graphes/eager)
```

**Règle le s3@3 de poste3** (`verdict-narrow-voyants-15-09.md` § (2)) : sur
24 564 positions au même critère top-1, narrow_gemm ne diverge pas plus
que le bruit graphes/eager déjà présent sans lui.

## Synthèse

| # | mesure | verdict |
|---|---|---|
| 1 | PPL graphes (production) | **RÉFUTÉ** (1,002286 > 1,002) |
| 1 | PPL eager | PASSÉ (0,999410) |
| 2 | preuve de sélection | oui (0 → 1551/196512) |
| 3 | divergence top-1 | CONFORME |

**Pas un feu vert net** : narrow_gemm est cohérent en isolation (eager,
divergence), mais son interaction avec les graphes CUDA dépasse le seuil
PPL de production, de peu. Rendu à chef/poste7 — décision narrow ON/OFF en
0.6.6 pas prise par moi.
