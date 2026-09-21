# acvram-parc — menus de modèles portables

`acvram-parc` empaquette `claude-modele(s)`, `kimi-modele(s)`, `modeles-a-jour`, `integrite-modeles`,
`telecharger-modele` et `parc-installer`. Le paquet ne contient **aucun chemin de machine** :
tout ce qui dépend du poste vit dans `~/.config/acvram-parc/parc.toml`, écrit par `parc-installer`.

## Installation
```
sudo apt install ./acvram-parc_<version>_amd64.deb
parc-installer            # questions ; --auto pour une machine sans écran
```
`parc-installer` : lit les GPU (`compute_cap < 12.0` → la classe `acvram-*` est retirée, NVFP4 = sm_120 ;
aucun GPU → llama.cpp CPU), détecte les moteurs présents (acvram, llama-server, vLLM, TabbyAPI, YALS, Jan,
kimi, claude ; un moteur absent n'a ni alias ni port), balaye les disques montés (`findmnt`, profondeur 4,
60 s par disque, le reste non balayé est compté et bloque `--auto`), demande confirmation, puis écrit :
`~/TSV/*.tsv`, les blocs `[models.*]` et `[providers.*]` de `~/.kimi-code/config.toml` (le reste du fichier
n'est jamais touché, sauvegarde `.avant-<date>`), `~/.config/ia-secrets.env` (clés aléatoires si absent),
`parc.toml`, le minuteur `modeles-a-jour` (systemd --user). Relancer = fusion ; deux passages = fichiers identiques.
Code de retour ≠ 0 si un alias ne résout pas (2), si le balayage est tronqué en `--auto` (3).

## D'une machine à l'autre
`parc-installer --exporter parc.tar` puis, là-bas, `parc-installer --importer parc.tar` : les notes gardent
la colonne « mesuré sur <hôte> » — un débit d'ici n'est pas un débit de là-bas.

## Tests à sec
`pytest tests/test_paquet_parc.py` : arbre du paquet sans motif de machine, installation à blanc
(3 faux modèles, sans GPU ni vLLM → un seul alias gguf, `config.toml` préexistant intact), idempotence, rc ≠ 0.
Construction : `tools/construire-deb-parc.sh` (refuse tout chemin de machine).
