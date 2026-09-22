# Scellé — cause de l'effondrement TRT-LLM à b=1 (poste3, 22/09, avant mesure)

Fait mesuré : trtllm b=1 = 46,3 t/s (21,6 ms/jeton) à **399,8 W (plafond 400 W)** ;
acvram b=1 = 390,5 t/s à 241 W. Le GPU trtllm est **saturé en puissance** à b=1 mais
improductif → beaucoup de watts, peu de jetons. b=1 est reproductible (2 fenêtres +
run charge.py) et non biaisé par le comptage (1024 jetons réellement décodés, 1 requête).

L'indice de puissance (400 W = GPU qui calcule, pas qui attend) oriente vers un
**calcul GPU massif et sous-productif par jeton à faible batch**, pas vers un
overhead hôte/réseau (qui laisserait le GPU à basse conso).

## 3 hypothèses, ligne de config portante, prédiction chiffrée

**H1 — GEMM/MoE sous-alimenté à M=1 (le plus probable).**
Ligne : `moe_config=MoeConfig(backend='AUTO')` + `nvfp4_gemm_config(allowed_backends=['cutlass','cublaslt','cuda_core'])`. À b=1, M=1 : les tensor cores FP4 sont alimentés par une seule ligne, l'occupation utile est minuscule alors que le kernel choisi pour le régime tourne à pleine puissance → 400 W, FLOPs utiles ridicules. À b=12, M=12 remplit les tuiles → débit ×43.
**Prédiction** : courbe débit(b) fortement SUR-linéaire — débit PAR SÉQUENCE croît nettement de b=1 à b=12 (b=1 ≈ 46, b=2 ≈ 90-140/séq, b=4 ≈ 200-300/séq…). Réfutée si débit/séquence ≈ constant.

**H2 — overhead fixe par tour de l'executor / overlap scheduler (RPC).**
Ligne : `disable_overlap_scheduler=False` + RPC orchestrator (`RPCServer ... ipc://` dans le log). Un coût fixe par itération de décodage non amorti à b=1.
**Prédiction** : si H2 dominait, la conso serait BASSE à b=1 (GPU en attente entre les tours) — CONTREDIT par les 400 W observés. Donc H2 improbable ; le test la départage : débit TOTAL ~plat de b=1 à b=4 signerait H2, ce qu'on n'attend pas.

**H3 — streaming SSE par jeton (`stream_interval=1` + client `stream=True`). RÉFUTÉE d'avance par la puissance** : un goulot réseau/SSE laisse le GPU idle (~100 W), or il est à 400 W. Contrôle négatif : si (contre toute attente) H3, `stream=False` remonterait le débit ET ferait CHUTER la conso. Prédiction : aucun changement.

## Test carte (≤ 10 min, 1 seul chargement trtllm)

`BANC_SLOTS=1,2,4,8,12` en un appel `decode` contre trtllm, relever débit/séquence
et watts par b. La FORME tranche :
- sur-linéaire (débit/séquence ↑ avec b) → **H1** (GEMM/MoE sous-alimenté à faible M) ;
- débit total plat de 1 à 4 → **H2** (overhead par tour) ;
- H3 déjà réfutée par les 400 W (confirmable en repassant `stream=False` sur b=1 si doute).

Verdict : si trtllm remonte à ≥ 300 t/s dans une variante praticable, la cellule b=1
se rejoue ; sinon on publie 46 t/s AVEC la cause (H1 attendue : régime à faible batch
de trtllm, intrinsèque, hors ressort du client). Le prefill reste hors chaîne (dit au verdict).

## Résultat débit(b) 08:03-08:16 — H1 INFIRMÉE, scheduler à batch plein

| b | total t/s | t/s/séq | watts | MHz | lots | durée |
|---|---|---|---|---|---|---|
| 1 | 46,3 | 46,3 | 399,7 | 2533 | 1 | 22,1 s |
| 2 | 49,1 | 24,6 | 399,4 | 2462 | 1 | 41,7 s |
| 4 | 50,6 | 12,7 | 399,7 | 2393 | 1 | 80,9 s |

b=8/12 NON atteints : à ~50 t/s, un lot de b=4 (4096 jetons) dure déjà 81 s, la
fenêtre de 20 s est dépassée par un seul lot ; b=8/12 auraient dépassé DUREE_MAX.

**Lecture** : le débit TOTAL est PLAT (~50 t/s) de b=1 à b=4, à 400 W constants ;
chaque lot de N requêtes de 1024 jetons dure ~N×22 s → trtllm **sérialise** les
requêtes à faible charge (elles ne se recouvrent pas). **H1 (GEMM/MoE sous-alimenté,
sur-linéaire dès b=2) est INFIRMÉE** : à bas batch le débit ne monte pas du tout.
**H2 partiellement confirmée** (overhead/sérialisation par tour), MAIS le b=12 = 1 997 t/s
de la cellule montre que trtllm bat efficacement PRÈS de `max_batch_size` (12) :
le débit n'explose qu'à batch quasi plein (transition entre b=4 et b=12, à confirmer
par b=8). Cause de l'effondrement b=1 : **le scheduler de trtllm-serve
(GUARANTEED_NO_EVICT, `max_batch_size=12`) ne recouvre pas les petites charges** —
il est taillé pour le débit à batch plein, pas la latence à requête unique. Ce n'est
pas un défaut du client (acvram, même client, fait 390 t/s à b=1).

**Suite** : confirmer la transition avec b=8 (fenêtre courte : BANC_JETONS=256 pour
que les lots tiennent, ou mesurer le temps du 1er lot seulement). La cellule b=1 = 46
se publie AVEC cette cause (scheduler à batch plein), reproductible sur 3 instruments
(débit-b, cellule, charge.py).
