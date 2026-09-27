# acvram v0.7.7 — English / français

## English

Safer A/B measurements and bilingual release notes.

- **A/B measurement guard** (piece 276f): `ACVRAM_ARBRE=<tree>` makes acvram refuse to import code from any other
  source tree and logs an `ARBRE` line, so that both arms of a comparison really run the code they claim to run.
  Without the variable nothing changes.
- **Bilingual release notes** (piece 287): every release is now described in English and French; the release script
  refuses notes that lack either language.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.7/acvram-0.7.7.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.7/CHANGELOG.md)

---

## Français

Mesures A/B plus sûres et notes de version bilingues.

- **Garde des mesures A/B** (pièce 276f) : `ACVRAM_ARBRE=<arbre>` fait refuser à acvram l'import du code de tout
  autre arbre source et imprime une ligne `ARBRE`, pour que les deux bras d'une comparaison exécutent vraiment le code
  annoncé. Sans la variable, rien ne change.
- **Notes de version bilingues** (pièce 287) : chaque version est décrite en anglais et en français ; le script de
  sortie refuse des notes auxquelles il manque l'une des deux langues.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.7/acvram-0.7.7.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.7/CHANGELOG.md)
