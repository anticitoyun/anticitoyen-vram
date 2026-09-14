# Prédiction scellée — collect.py doit résoudre l'attention par structure

poste1, 15/09/2026, avant mesure. Signalé par poste2 via chef : conversion
GLM-4.7-Flash-srcbf16-nvfp4, « calibration indisponible
('model.layers.0.self_attn.q_proj.weight') ; repli sur l'arrondi au plus
proche » — AWQ désactivé pour tout le modèle.

## Cause (lue)

`_build_bf16_layer` (acvram/quant/collect.py:191-225) construit toujours
une `Attention` plate (`self_attn.q_proj/k_proj/v_proj/o_proj`) — absents
d'un MLA à q_lora (GLM-4.7-Flash a `q_a_proj`/`q_b_proj`, pas `q_proj`).
Le `KeyError` remonte jusqu'à `cli.py:469` (`except Exception`), qui
désactive AWQ pour TOUT le modèle plutôt que pour la seule couche 0. Même
famille que les 3 bogues d'hier (nom/liste en dur au lieu de la structure
— `spec.est_mla` existe déjà, comme pour bead 992).

## Prédiction

1. Avec `_build_bf16_layer` branché sur `spec.est_mla` (MLAttention,
   q_a_proj/q_b_proj si présents, `kv_a_proj_with_mqa`/`o_proj` sinon
   `q_proj`/`k_proj`/`v_proj`/`o_proj`), `collect_activation_stats` sur un
   mini-modèle MLA synthétique (q_lora_rank>0, kv_lora_rank>0) rend des
   statistiques pour CHAQUE tenseur linéaire attendu de l'attention
   (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj`), sans lever
   d'exception.
2. Avec en plus la résilience par couche (try/except autour de chaque
   couche dans la boucle), une couche délibérément cassée (nom de tenseur
   invalide injecté) ne fait perdre que SES statistiques — les autres
   couches restent calibrées, `len(stats) > 0`.

**Seuil de réfutation** : `collect_activation_stats` lève encore une
exception sur le mini-modèle MLA, OU une couche cassée fait tomber
`len(stats)` à 0 (perte totale au lieu de partielle).

## MESURÉ : CONFIRMÉ

`tests/test_collect_mla.py`, deux tests, tous deux verts avec le
correctif, tous deux cassants sans lui (vérifié par `git stash`
temporaire, remise en place à l'identique) :
- `test_calibration_mla_ne_leve_pas_et_couvre_les_bons_tenseurs` :
  `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` tous présents
  dans `stats`, sans exception.
- `test_une_couche_cassee_ne_perd_que_ses_statistiques` : une couche
  1 délibérément privée de `q_a_proj` perd SES statistiques (rien pour
  elle), la couche 0 reste calibrée normalement — le message
  d'avertissement nomme la couche et le tenseur.

Piège trouvé en cours de route (pas anticipé dans la prédiction) :
aliaser un attribut Python (`attn.kv_a_proj_with_mqa = attn.kv_a_proj`,
`couche.self_attn = couche.linear_attn`) pour donner un second nom à un
même sous-module ne suffit PAS — `nn.Module.named_modules()` déduplique
par identité et ne revisite jamais un module déjà rencontré sous un
autre chemin. Résolu par un attachement direct par clé explicite
(`_StatCollector.attach_named`), pas par le nom d'attribut.

Bonus (même famille, même fichier) : la construction de `MoEBlock` dans
`_build_bf16_layer` ignorait `scoring`/`score_bias`/`routed_scale`/
`shared` — calibrait GLM-4.7-Flash en routage softmax sans biais (le
défaut), donc sur des experts différents de la production. Corrigé pour
matcher loader.py:441-456 (structure, pas de valeur par défaut
implicite). Non couvert par un test dédié : le test MLA ci-dessus vérifie
que les statistiques existent, pas qu'elles portent sur les bons
experts — à faire si un doute se présente.
