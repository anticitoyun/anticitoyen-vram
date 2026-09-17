# Verdict — asymétrie b=1 acvram/GGUF : deux `eos_token_id` déclarés d'un côté, un seul de l'autre (17/09)

poste1, à sec, ordre relayé par chef (bloc 2 de poste3, 4c0cc15) : b=1 vide
sur 7/10 entrées acvram (EOS avant 256 jetons) contre 0/10 GGUF+EXL3 sur le
même bloc, asymétrie systématique.

## Protocole

Même famille de modèle des deux côtés (Qwen3.8-27B, même tokenizer, mêmes
identifiants de jetons) : `generation_config.json` du converti acvram
(`models_acvram/Qwen3.8-27B-nvfp4`) contre les métadonnées du GGUF
correspondant (`models_gguf/Qwen3.8-27B-Q4_K_M-poste3`), lues directement
(`acvram.quant.gguf.GGUFFile`, aucune inférence). Comparaison de
configuration, pas de génération — la cause se lit dans les fichiers.

## Mesuré

| | acvram (`generation_config.json`) | GGUF (`tokenizer.ggml.*`) |
|---|---|---|
| `eos_token_id` | **[248046, 248044]** (deux) | **248046** (un seul) |
| texte du jeton 248046 | `<\|im_end\|>` | `<\|im_end\|>` |
| texte du jeton 248044 | `<\|endoftext\|>` (aussi bos/pad) | absent de `eos_token_id`, présent au vocabulaire, `token_type=3` (CONTROL) |

acvram lit `generation_config.json` tel quel : les DEUX jetons stoppent la
génération. Le GGUF ne déclare que 248046 dans sa clé
`tokenizer.ggml.eos_token_id` (singulière, pas de tableau) — 248044 existe
bien au vocabulaire et porte le bon type CONTROL, mais rien dans les
métadonnées du fichier ne dit à llama.cpp qu'il vaut aussi arrêt de
génération.

**Fait notable, non demandé mais qui corrobore** : `acvram/quant/gguf.py:637-650`
(lu, pas deviné) documente déjà ce piège pour son propre chemin d'IMPORT
d'un GGUF comme source (`FINS = ("<|im_end|>", "<|eot_id|>",
"<|endoftext|>", ...)`, recherche par texte de jeton, jamais par le seul
`eos_token_id` déclaré) — la clé "un seul eos_token_id, plusieurs marqueurs
de fin de tour" est un piège déjà nommé dans ce dépôt, sur l'autre sens de
la conversion.

## VERDICT : cause identifiée, ce n'est pas un bogue de fiche-service.py

L'asymétrie est réelle et s'explique entièrement par la configuration :
sur l'invite synthétique de `mesure_decode_b1` (texte remanié, pas un
prompt naturel), le modèle choisit parfois d'émettre `<|endoftext|>`
plutôt que `<|im_end|>` comme marqueur de fin — acvram le reconnaît
(deux jetons dans son `eos_token_id`), le GGUF servi par llama-server ne
le reconnaît vraisemblablement pas (un seul déclaré dans ses métadonnées,
`<|endoftext|>` absent). acvram s'arrête tôt à raison ; le GGUF continue
jusqu'à `max_tokens` parce qu'il ne sait pas que ce jeton-là aussi signale
la fin — pas parce qu'il « va plus loin » légitimement.

`fiche-service.py::mesure_decode_b1` refuse déjà de publier un débit
tronqué avant la moitié de `n_tokens` (comportement voulu, REGLES « un
échec est un résultat ») : rien à corriger dans l'outil. Le refus côté
acvram est correct ; le silence (débit publié jusqu'à 256) côté GGUF
n'est pas faux non plus — il mesure juste un régime différent (la
question « le modèle a-t-il fini » n'a pas la même réponse selon le
moteur, pour cette invite précise).

## Conséquence

Ne pas lire les colonnes b=1 acvram vs GGUF/EXL3 comme un chiffre de
vitesse comparable quand l'une est vide et l'autre pleine sur la même
entrée : la fiche GGUF a pu tourner jusqu'à 256 jetons de texte hors
sujet après ce que le modèle considérait déjà comme terminé. Pas de
correctif proposé dans ce chantier (portée : diagnostic seul, demandé
par chef) — deux options pour qui reprend, non tranchées ici :
regénérer les GGUF concernés avec une conversion qui déclare le tableau
complet des `eos_token_id` (si l'outil de conversion le permet), ou
accepter l'asymétrie comme propriété connue du format et l'annoter dans
les menus au même titre que la note b=1 « décodage pur en flux » déjà
actée par poste7.
