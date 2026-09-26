# Verdict — Nemotron-3.5-30B-A3B converti depuis le bf16 source (pas le double-quantifié), classable

poste2, 17/09. Prérequis de `poste7-priorite-apres-campagne-17-09.md` §1/§2 :
le seul variant mesuré jusqu'ici (`...-srcexl3_6bpw-nvfp4`, PPL 1,0795) part
d'un point de contrôle déjà quantifié en EXL3 6 bpw — condamné d'avance par
double quantification. Celui-ci part du bf16 sur disque depuis le palier 2
(`/mnt/4TO_SATACMR_2022/Modeles/models/Nemotron-3.5-Lightning-30B-A3B-bf16`,
62 Gio), corpus bras A.

## Bogue trouvé et corrigé avant toute conversion

Premier dry-run (à sec, `--dry-run`) : `ValueError: conversion incomplète :
46 tenseurs attendus absents (premier : model.layers.0.self_attn.q_proj.
weight)`. Cause, trouvée par lecture (fichier+ligne, REGLES §7) :
`load_model_spec` (`acvram/engine/config.py:545`) ne dérivait `layer_types`
depuis `nemotron_h` QUE si `hybrid_override_pattern` (motif M/E/-/*) était
présent. Le bf16 source ne porte pas ce champ — seulement `layers_block_type`
(liste de mots `mamba`/`moe`/`attention`), déjà présent dans son
`config.json`. `layer_types` restait `None`, `couches_recurrentes` valait 0
(`config.py:291`), et le convertisseur réclamait `self_attn.q_proj` sur les
23 couches Mamba2 qui n'en portent pas.

Correctif (`56dd2ee`) : `load_model_spec` lit maintenant `layers_block_type`
quand `hybrid_override_pattern` est absent, même mapping de types de couche.
Témoin (REGLES §5) : désactiver la branche `layers_block_type` fait rougir
`tests/test_nemotron_h_layer_types.py` (`KeyError: 'hybrid_override_pattern'`) ;
réactivée, 2/2 verts. Suite complète : 7 échecs, tous déjà rouges avant mon
changement — `test_depot_sans_identite` (identique à `origin/main`, vérifié
en session précédente), `test_menus.py` (idem), et deux échecs neufs dans
`test_collect_qwen35_gdn.py` causés par l'installation `flash-linear-attention`
dans le venv partagé (« 0 active drivers ([]). There should only be one. »)
— environnement de poste4, hors de mon périmètre, signalé mais pas corrigé
par moi.

## Conversion réelle

Carte : dry-run 412,4 s, conversion réelle 422,4 s (intercalée dans la chaîne
FLA de poste3, verrou pris/rendu proprement, sa suite a repris derrière moi
sans intervention).

```
Nemotron-3.5-Lightning-30B-A3B-nvfp4 : llama, 52 couches, h=2688,
103,1 G paramètres (8,2 G actifs par jeton), MoE 128 experts, top-6
tenseurs 6243 ; entrée 58,8 Gio ; sortie 17,1 Gio (×3,44)
SNR sortie moyen 20,5 dB
pires tenseurs : self_attn.o_proj (couches 19/33/26, 20,4 dB),
  mlp.experts.{118,2}.down_proj couche 1 (20,4 dB)
```

5888 experts sans statistique de calibration (jamais routés ou trop peu sur
le corpus bras A) — échelle identité explicite dans le manifeste, jamais une
absence ambiguë (comportement déjà en place, pas modifié ici).
`calibration indisponible ('model.embed_tokens.weight')` : repli attendu,
l'embedding n'entre pas dans le chemin de calibration AWQ.

## sha256 (`/mnt/2TO_2023_980PRO/Modeles/models_acvram/Nemotron-3.5-Lightning-30B-A3B-nvfp4/`)

```
acvram-00000.safetensors : 626eecfec3d0ac2587b69ec3f800f4ca88aa2b520e548822d0249380e402af04
acvram-00001.safetensors : 42c74a43eb3ef22573cc578293ac5629b9a65af24b2e013115bb2eb75ce61ce7
acvram-00002.safetensors : bb611c595fb145267508eb008da114ea0041768b76048b4ceb8be3adb35a7017
acvram-00003.safetensors : 3d507cb549d867d61713c7d1cff8ba52f2522ac84b844f22723fdb3f1b92dd60
acvram-00004.safetensors : 8a5ad0896e43a80edd47133c964bf2ab433ba648c1fccb4f58407078895a46aa
acvram_manifest.json     : cb8db526ac3bceb6e912a89cea55eb19590e29c99f6848a9966fa954821ff731
config.json               : a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c
```

## Suite

Prêt pour poste3 : PPL 3 tranches dans la même fenêtre que Qwen3.8, corpus
bras A (comme `poste7-priorite-apres-campagne-17-09.md` §1/§2 le demande).
Servable directement : `acvram serve /mnt/2TO_2023_980PRO/Modeles/
models_acvram/Nemotron-3.5-Lightning-30B-A3B-nvfp4`.
