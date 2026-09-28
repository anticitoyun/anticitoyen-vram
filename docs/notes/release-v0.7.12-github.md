# acvram v0.7.12 — English / français

## English

A clear warning when offloading would collapse throughput; video models in Open WebUI.

- **"Offload cliff" warning** (piece 295): when the load plan pushes MLP layers off the GPU and the expected throughput
  drops below 25 % of the resident one (`ACVRAM_SEUIL_FALAISE`), `acvram serve` says so at startup and gives the context
  length that would fit without offloading (`--max-model-len N`). Nothing is refused. Real case: Devstral 24B at 32k
  context, 7.9 tokens/s instead of 94.5.
- **Video in Open WebUI** (companion parc package 0.1.6): Wan 2.2 14B image-to-video and text-to-video, Wan VACE / Fun
  Control and LTX-2.3 appear as models in the selector; an attached image or video is used as input and the video is
  shown in the chat. Model menus: unknown values stay at the end of the list in both sort directions.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.12/acvram-0.7.12.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.12/CHANGELOG.md)

---

## Français

Un avertissement clair quand l'exil ferait s'effondrer le débit ; modèles vidéo dans Open WebUI.

- **Avertissement « falaise d'exil »** (pièce 295) : quand le plan de chargement sort des couches MLP du GPU et que le
  débit prévu tombe sous 25 % du débit résident (`ACVRAM_SEUIL_FALAISE`), `acvram serve` le dit au démarrage et donne la
  longueur de contexte qui tiendrait sans exil (`--max-model-len N`). Rien n'est refusé. Cas réel : Devstral 24B à 32 k
  de contexte, 7,9 jetons/s au lieu de 94,5.
- **Vidéo dans Open WebUI** (paquet parc 0.1.6) : Wan 2.2 14B image-vers-vidéo et texte-vers-vidéo, Wan VACE / Fun
  Control et LTX-2.3 apparaissent comme des modèles du sélecteur ; l'image ou la vidéo jointe sert d'entrée et la vidéo
  s'affiche dans la conversation. Menus de modèles : les valeurs inconnues restent en fin de liste dans les deux sens.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.12/acvram-0.7.12.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.12/CHANGELOG.md)
