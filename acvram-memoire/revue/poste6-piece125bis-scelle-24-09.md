# Scellé — pièce 125 bis : le W8A8 du préfill est-il ce qui rend le préfill du Coder i8c moins juste ? (poste6, 24/09, AVANT la mesure)

Feu de chef sur le verdict 125 (fedef7e9) : sur l'alias servi, KL(HF ‖ préfill) 0,054 contre KL(HF ‖ forcé) 0,038, préfill
le moins juste sur 3/5 invites (×3,35 sur l'invite 2). Candidat nommé : les projections q/k/v/o int8 passent au préfill
en **W8A8** (`ACVRAM_PREFILL_INT8=cublas`, activation quantifiée int8 par jeton, défaut depuis poste7-p2-au-defaut-19-09),
au décodage en W8A16 (GEMM étroit, activation bf16).

## Instrument
`outils/gpu/mesure/kl-chemins-p125.py` (commit dans le verdict), mode `prefill`, Coder i8c, **deux bras dans la même prise** :
`ACVRAM_PREFILL_INT8=cublas` (défaut, `P125_SUFFIXE=-cublas`) puis `ACVRAM_PREFILL_INT8=bf16` (témoin : déquant entière +
cutlass, `P125_SUFFIXE=-bf16`) ; mêmes dumps HF (`4b`-style : `coder/invite{i}.pt` du 02 h 03), mêmes ids ; le bras forcé du
02 h 04 (91593bf6, même code de décodage) sert de repère, pas de bras. Chaque bras imprime la ligne de régime du processus :
**le bras bf16 doit y porter `ACVRAM_PREFILL_INT8=bf16`, le bras cublas « défaut »** — sinon la variable n'a pas pris, prise nulle.
Comparaison à sec : KL(HF ‖ préfill) par position de réponse, L2 relative du flux résiduel par couche (positions d'invite)
contre HF, et KL(cublas ‖ bf16) directement.

## Seuils, fixés ici
* **S0** : ligne de régime conforme sur les deux bras ; compute-apps début = fin ; max_perf_pct début = fin.
* **Plancher de quantification** = KL(HF ‖ préfill bf16) par invite (le préfill sans quantification d'activation : il ne reste
  que les poids). **FAUX (W8A8 non coupable)** si KL_moy(cublas) ≤ 1,25 × KL_moy(bf16) sur ≥ 2/5 invites — l'écart W8A8
  est sous le plancher.
* **COUPABLE** si KL_moy(cublas) ≥ 1,25 × KL_moy(bf16) sur ≥ 4/5 ET L2 d'invite (dernière couche) cublas ≥ 1,10 × bf16 sur ≥ 4/5.
* Entre les deux : « partiel », dit tel quel, sans verdict de culpabilité.
* Secondaire (attendu si coupable) : KL_moy(préfill bf16) ≤ KL_moy(forcé du 02 h 04) sur ≥ 4/5 — le préfill sans W8A8 redevient
  au moins aussi juste que le décodage.

## Prédiction et issues
* Prédit : COUPABLE — KL_moy cublas / bf16 = 1,5-3 sur ≥ 4/5 (le W8A8 explique l'excédent 0,054 − 0,038 ≈ 0,02 nat) ;
  L2 cublas / bf16 = 1,05-1,20 ; préfill bf16 ≤ forcé sur ≥ 4/5.
* FAUX si le ratio KL ≤ 1,25 sur ≥ 2/5 : l'excédent vient d'ailleurs (attention flash bf16 contre paginée, glue C15, MoE de
  préfill en 3 lancements) — à nommer, pas à mesurer ici.
* Ce qui me gênerait : **préfill bf16 PIRE que cublas** (ratio < 0,8 sur ≥ 2/5) — le W8A8 serait innocent et ma lecture
  de la 125 fausse ; je le dirais tel quel.
* Décision sur le défaut servi : **à chef seul** (débit ET KL ; le W8A8 est le levier C15, ×1,29-1,34 de préfill).

## Durée et régime
Une prise ≤ 5 min (2 chargements de 21-60 s + 5 invites chacun) ; carte tenue par poste2 (102, b=1) au moment du scellé :
la prise passe juste après, poste2 prévenue. cpu-safe=off ; relevés au début et à la fin.
