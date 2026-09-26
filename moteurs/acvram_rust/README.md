# acvram_rust

Moteur de comparaison d'acvram : même modèle, mêmes poids, **mêmes noyaux (le SASS que le moteur
Python sert)**, hôte écrit en Rust. Il mesure la question (a) du contrat
(`acvram-memoire/revue/moteurs-rust-mojo-23-09.md`) : ce que coûte le langage hôte, rien d'autre.

## État (étape 1, squelette)

| partie | état |
|---|---|
| manifeste, poids safetensors projetés, envoi sur la carte | écrit |
| tokeniseur + gabarit Jinja | **ids d'invite identiques au Python sur 5/5** (`tests/a_sec.rs`) |
| noyaux : fatbin de l'extension servie, énumération, clés stables | écrit, testé à sec (fatbin) |
| séquence d'un pas (préfill, décodage) | **à transcrire depuis le relevé de la prise 1** — `generer` refuse d'ici là |
| `/v1/models`, `/v1/chat/completions` (non streamé, glouton) | écrit |

## Porte (scellée avant la mesure)

sha256 des ids générés identique au moteur Python sur les 5 invites de `tests/invites.json`,
glouton, 128 jetons, graphes Python actifs : **5/5**, sinon faux.

## Commandes

```bash
cargo check                                    # d'abord, toujours
CUDA_VISIBLE_DEVICES= cargo test --test a_sec   # à sec
cargo clippy --all-targets
acvram-rust serve <dossier-modèle> --noyaux ~/.cache/acvram/kernels-<hash>/acvram_kernels.so --port 8110
```

Pendant une fenêtre de mesure : `nice -n 19 taskset -c <2 cœurs> cargo … -j 2`. Build release seulement
pour une prise, sous `outils/carte.sh`.

Fixture des ids : `moteurs/acvram_rust/outils/ids_python.py` (à sec, `CUDA_VISIBLE_DEVICES=`).
