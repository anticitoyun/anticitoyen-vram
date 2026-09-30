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

## Menus (claude-modeles, kimi-modeles) — depuis 0.1.9

**Seuils par client, jamais une constante partagée.** Chaque menu lit son propre seuil dans
son lanceur CLI, jamais de copie dans la GUI (`parc/bin/claude-modele:9-21`,
`parc/bin/kimi-modele:8-14` ; `seuils_lanceur()`, `parc/bin/claude-modeles:26`) :
`claude-modele --seuils` imprime `seuil_complet=45000` (`SEUIL_COMPLET`) et
`seuil_minimum=29120` (`FENETRE_MIN`, calculé) ; `kimi-modele --seuils` imprime
`seuil_complet=65536` (`KIMI_MCP_CTX_MIN`) et `seuil_minimum=` (vide — pas de plancher dur).
Une copie chiffrée dans la GUI avait survécu au passage du lanceur à 29 120, causant un
lancement suivi d'un refus (`tests/test_seuils_lanceur_ph1.py`).

**Colonne Contexte : mode par alias.** `Fenetre._mode_client()` (`parc/lib/menu_modeles/fenetre.py:801-813`)
rend `complet`, `reduit` ou `refuse` selon la fenêtre de l'alias contre les deux seuils du
profil du client ; `_raison_mode_client()` (:815-819) donne le texte de l'infobulle. Rendu
(:336-346) : ⛔ rouge si `refuse`, ◐ orange si `reduit`, rien sinon. `ouvrir_kimi()` (:944-948)
refuse le lancement (toast, pas de spawn) si `refuse`, avertit et lance quand même si `reduit`.

**Attente de la carte rendue au changement de modèle (kwh, 30/09).** Les trois lanceurs
(`acvram-serveur`, `vllm-serveur`, `llamacpp-serveur`) attendent que le serveur remplacé ait
rendu sa VRAM avant de démarrer le suivant : `parc/lib/carte_rendue.py`, appelé par
`attendre_carte_rendue()` (`parc/bin/acvram-serveur:48-50`, invoqué :91 et :157). Remplace
l'ancien seuil fixe « libre > 24 000 Mio », qui laissait partir un gros modèle trop tôt quand
un processus vivant (ComfyUI, appoint) occupait déjà la carte, ou attendait à tort derrière un
PID mort encore listé par `nvidia-smi --query-compute-apps`. Délai `ACVRAM_ATTENTE_VRAM`
(défaut 60 s), refus nommé (rc 1) au-delà plutôt qu'un OOM qui accuse la taille du modèle.

## Tests à sec
`pytest tests/test_paquet_parc.py` : arbre du paquet sans motif de machine, installation à blanc
(3 faux modèles, sans GPU ni vLLM → un seul alias gguf, `config.toml` préexistant intact), idempotence, rc ≠ 0.
Construction : `tools/construire-deb-parc.sh` (refuse tout chemin de machine).
