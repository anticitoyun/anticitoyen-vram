# 268 — /metrics hors de la boucle HTTP : TTFT à 12 sous lecteur 20 Hz 0,78 → 0,24 s (poste1, 26/09) — VRAI

* instrument : `tests/test_metrics_hors_boucle_268.py` (cassants boucle, compte, parcours, gabarit ; identité ; jetons au bit) ;
  `scratchpad/poste1-p268-26-09/` — `prise-tests.sh` (nouveau code, puis ancien e3c345bf7 en worktree jetable), `prise-ttft.sh`
  (`client-ttft-262.py`, serveur neuf par bras, 5 tours à 12, VEILLEUR = GET /metrics toutes les 50 ms comme le duel)
* commit : poste1-268 306e3f58b (tests, TTFT) ; étape 1 = ec81b6987 (+ 562abe0cd), étape 2 = f1177799c ; ancien = e3c345bf7
* régime : Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4, -lgc 2700, ACVRAM_ECO=off, spéculation coupée, max-batch 16, cpu-safe 100
* scellé : `scratchpad/poste1-p268-26-09/scelle.md` (4562e92aa, avant le code) + errata 1-3 et addendum 1 (avant chaque prise)
* mesuré : tests nouveau 64 passed ; ancien 4 cassants ROUGES ; TTFT A B T U U T B A
* verdict : **VRAI** — B/T = 0,97 et 0,98 (seuil ≤ 1,10), B ≤ 0,266 s ; /metrics 343 → 60 ms. Étape 2 (gabarit hors boucle) : neutre
* durée : prévu ≤ 20 min ; tenu (`carte.sh` journal `tenue=`) tests 14 s, TTFT 663 s

## Chiffres (mur du tour à 12, médiane de 5 ; deux passes par bras)
| bras | code | lecteur /metrics | /metrics médian | mur |
|---|---|---|---|---|
| A | ancien e3c345bf7 | 20 Hz | 334 / 343 ms | **0,770 / 0,787 s** |
| B | étape 1 | 20 Hz | 61 / 60 ms | **0,240 / 0,242 s** |
| T | étape 1 | aucun | — | 0,247 / 0,246 s |
| U | étape 2 | aucun | — | 0,255 / 0,249 s |

## Ce qui a changé (sortie de /metrics identique champ par champ, test d'identité)
1. `app.py` `metrics` : `def` au lieu de `async def` → pool de fils FastAPI, la boucle HTTP reste libre.
2. `engine.regime()` UNE fois par /metrics (il y était appelé 7 fois).
3. `Engine.regime()` : un seul parcours de `model.modules()` (il en faisait 5 : 4 dans le corps, 1 dans `_regime_echelle_awq`,
   plus un 6e dans `_couverture_experts` sur un modèle sans MoE) ; les deux aides reçoivent les blocs collectés, même ordre.
   Pas de cache entre appels : `streamed` change en service (`_promote_expert`).
4. Étape 2 : `render_chat` et `_encode` de `/v1/chat/completions` via `asyncio.to_thread`, même ordre ; jetons au bit (test).

## Lecture
* Le reste de /metrics (60 ms) est le sous-processus nvidia-smi de `capteurs.nvidia()` (41 ms à vide) et regime() : hors boucle
  et sous-processus (GIL rendu), il ne coûte plus rien au service (B = T à 3 % près, du bon côté).
* Étape 2 : U − T = +8 / +3 ms, dans l'étendue des tours (± 15 ms) — ni gain ni perte mesurables à 12 requêtes de 470 jetons.
  Elle libère la boucle pendant le rendu Jinja (cassant gabarit : 0,259 s → < 0,1 s), ce qui compte pour de longs gabarits (outils) ;
  la 269 (poste6, fenêtre d'admission) dira si elle resserre les arrivées. Proposé : la garder (au bit, sans coût mesuré).
* Erratum de ma 262 : j'y comptais six appels regime() et quatre parcours ; c'était sept et cinq (six sur un modèle dense).

## Suite proposée (chef)
* Fusion de poste1-268 (0.7.1). Requalifier au README tout TTFT « à 12 » mesuré par duel-moteurs.py avant ce correctif.
* Optionnel : `capteurs.nvidia()` par NVML (pynvml) au lieu du sous-processus — 41 ms → < 1 ms par /metrics, sans enjeu de service.
