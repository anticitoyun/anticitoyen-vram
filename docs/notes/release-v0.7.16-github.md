# acvram v0.7.16 — English / français

## English

Model menus hardened after a full real test of every alias.

- **Real test of the menus**: 343 aliases started and queried under Claude Code and kimi at their serving context;
  Claude Code works on 95.2 % and kimi on 82.8 % of the aliases in range. The failures found are fixed here or named.
- **Slow starts**: a fixed 241 s launcher delay killed servers that were still loading (the slowest served start took
  238 s); the launcher now waits up to 480 s and names the last step reached.
- **GPU handover**: acvram, vLLM and llama.cpp launchers wait until the previous server has released the GPU memory
  before starting (bounded wait, named refusal).
- **YALS and TabbyAPI** (companion parc package 0.1.10): their launchers join the package without personal paths; a
  one-token completion after loading makes an unreadable chat template, a model the loader refuses or a missing GGUF
  fail at once with its name, instead of a silent 300 s wait. kimi on TabbyAPI now uses the right API key.
- **Menus** read each client's context thresholds from its launcher; Claude Code's prompt is measured before launch.
- **Tool calls**: arguments sent as JSON strings are decoded before the chat template (Qwen3-Coder used to fall back
  to a template without tools).
- **Conversion**: `acvram convert --echelle` offers a searched NVFP4 block scale (ScaleSweep); the default is unchanged.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.16/acvram-0.7.16.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.16/CHANGELOG.md)

---

## Français

Menus de modèles durcis après un test réel de tous les alias.

- **Test réel des menus** : 343 alias démarrés et interrogés sous Claude Code et kimi à leur contexte de service ;
  Claude Code fonctionne sur 95,2 % et kimi sur 82,8 % des alias dans le champ. Les pannes trouvées sont corrigées ici
  ou nommées.
- **Démarrages lents** : un délai fixe de 241 s coupait des serveurs encore en chargement (le plus long démarrage servi
  a pris 238 s) ; le lanceur attend désormais 480 s et nomme la dernière étape atteinte.
- **Passage de la carte** : les lanceurs acvram, vLLM et llama.cpp attendent que le serveur précédent ait rendu la
  mémoire du GPU avant de démarrer (attente bornée, refus nommé).
- **YALS et TabbyAPI** (paquet parc 0.1.10) : leurs lanceurs rejoignent le paquet sans chemin personnel ; une
  complétion d'un jeton après le chargement fait échouer tout de suite, en le nommant, un gabarit illisible, un modèle
  refusé par le chargeur ou un GGUF absent, au lieu d'une attente muette de 300 s. kimi sur TabbyAPI utilise la bonne clé.
- **Menus** : chaque client lit ses seuils de contexte dans son lanceur ; l'invite de Claude Code est mesurée avant le
  lancement.
- **Appels d'outils** : les arguments reçus en chaîne JSON sont décodés avant le gabarit (Qwen3-Coder retombait sur un
  gabarit sans outils).
- **Conversion** : `acvram convert --echelle` propose une échelle de bloc NVFP4 cherchée (ScaleSweep) ; le défaut ne
  change pas.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.16/acvram-0.7.16.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.16/CHANGELOG.md)
