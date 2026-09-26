# Publier sur chaque canal — pas à pas pour l'utilisateur

> Ce document ne fait rien lui-même : aucune session Claude n'a de compte AUR, COPR,
> Flathub ou Weblate, et aucune ne doit en créer un. Chaque section liste les gestes
> EXACTS que le propriétaire du dépôt effectue lui-même, une fois, en dehors de ce dépôt.

Chaque release (`.github/workflows/release.yml`) construit et joint les fichiers que ces
canaux consomment — AUR (`aur-<version>.tar.gz`), RPM/COPR (`.rpm` + `.src.rpm`), Flatpak
(dépôt gh-pages + `.flatpakref`), traductions (`translations-<version>.zip`). Publier un
canal, c'est faire connaître ces fichiers au canal — jamais les reconstruire à la main.

---

## AUR (Arch User Repository)

Le job `aur` de `release.yml` produit `aur-<version>.tar.gz` (PKGBUILD, `.SRCINFO`,
`acvram.install`), joint à chaque release GitHub. Publier sur AUR, c'est pousser ces deux
premiers fichiers vers le dépôt git AUR du paquet — un geste indépendant de GitHub.

1. **Compte AUR** : créer un compte sur [aur.archlinux.org](https://aur.archlinux.org)
   (aucun lien avec le compte GitHub).
2. **Clé SSH** : générer une clé dédiée (`ssh-keygen -t ed25519 -f ~/.ssh/aur`), coller la
   clé PUBLIQUE dans *My Account → SSH Public Key* du compte AUR. Ajouter dans
   `~/.ssh/config` :
   ```
   Host aur.archlinux.org
     IdentityFile ~/.ssh/aur
     User aur
   ```
3. **Premier envoi** (le paquet `acvram` n'existe pas encore côté AUR) :
   ```bash
   V=0.7.2   # la version de la release à publier
   curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/download/v$V/aur-$V.tar.gz
   tar xzf aur-$V.tar.gz && cd acvram
   git init && git remote add origin ssh://aur@aur.archlinux.org/acvram.git
   git add PKGBUILD .SRCINFO acvram.install
   git commit -m "acvram $V"
   git push -u origin master
   ```
4. **Chaque release suivante** : recloner `ssh://aur@aur.archlinux.org/acvram.git` (le
   dépôt AUR existe déjà), remplacer PKGBUILD/.SRCINFO/acvram.install par ceux de la
   nouvelle `aur-<version>.tar.gz`, committer, pousser. Aucun autre fichier n'entre dans
   ce dépôt (AUR limite la taille du dépôt git lui-même).
5. **Vérifier** : `makepkg --printsrcinfo | diff - .SRCINFO` doit rendre vide avant de
   pousser — `.SRCINFO` est généré, jamais édité à la main (source de vérité = PKGBUILD).

Aucun webhook : AUR ne construit rien côté serveur, `makepkg -si` tourne toujours chez
la personne qui installe.

---

## COPR (Fedora)

Le job `rpm` de `release.yml` construit `.rpm` et `.src.rpm` dans un conteneur
`fedora:latest` et les joint à la release (`packaging/rpm/acvram.spec`, venv privé au
premier lancement — pas de `python3-torch-cuda` empaqueté sous Fedora, voir le
commentaire de `acvram.spec:12-20`). COPR reconstruit depuis le `.src.rpm`, il ne le sert
jamais tel quel : publier, c'est déclencher cette reconstruction.

1. **Compte** : se connecter sur [copr.fedorainfracloud.org](https://copr.fedorainfracloud.org)
   avec un compte Fedora (FAS — *Fedora Account System*, à créer si absent).
2. **Nouveau projet** : *New Project* → nom `acvram`, description courte, **Chroots** :
   cocher `fedora-rawhide-x86_64` et les versions supportées du moment (`fedora-41-x86_64`,
   `fedora-42-x86_64` — noarch, donc pas besoin d'autres architectures). Le paquet est
   `BuildArch: noarch` (`acvram.spec:8`) : un seul jeu de RPM sert tous les chroots x86_64.
3. **Construction depuis le `.src.rpm` de la release** (pas depuis un dépôt git suivi par
   COPR — la source de vérité reste la release GitHub) :
   ```bash
   dnf install -y copr-cli   # ou pip install copr-cli
   copr-cli login             # jeton depuis copr.fedorainfracloud.org/api
   V=0.7.2
   curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/download/v$V/acvram-$V-1.fc42.src.rpm
   copr-cli build acvram acvram-$V-1.fc42.src.rpm
   ```
4. **Webhook (optionnel)** : COPR peut reconstruire automatiquement sur un push vers un
   dépôt git qu'il surveille — inutile ici, puisque la source de vérité est le `.src.rpm`
   déjà construit et versionné par `release.yml`, pas un dépôt que COPR cloit lui-même.
   Ne PAS brancher le webhook GitHub → COPR sur ce dépôt : COPR reconstruirait depuis les
   *sources*, pas depuis le spec généré (`@VERSION@` substitué par le job, jamais commité).
5. **Utilisateurs** : une fois le premier build vert,
   `sudo dnf copr enable <compte>/acvram && sudo dnf install acvram`.

---

## Flathub

Deux voies distinctes, à ne pas confondre — le dépôt gh-pages sert déjà, la soumission
Flathub officielle reste à faire.

### Ce qui sert déjà : le dépôt gh-pages (auto-hébergé, automatique)

Le job `flatpak` de `release.yml` construit le paquet hors ligne (roues PyPI + torch
cu130 en `extra-data`, `packaging/flathub/README.md`), publie un dépôt OSTree qui
S'ACCUMULE sur la branche `gh-pages` (jamais réécrit — pièce 266 l,
`tests/test_flathub_gh_pages_266l.py`), et joint un `.flatpakref` (~1 Kio) à la release.
**Rien à publier ici** : ce canal est déjà automatique à chaque release, tag pertaine.
Le fichier `docs/README.*.md` § Installer documente la commande (`flatpak install --user
<url>.flatpakref`) — la phrase ajoutée par cette pièce (278) rappelle qu'un `.flatpakref`
suit toujours le dépôt : réinstaller le même fichier après une nouvelle release met à
jour, il n'installe jamais une version figée.

### Ce qui resterait à faire : la soumission Flathub officielle

Un paquet sur Flathub (visible dans le magasin GNOME Software, `flatpak search acvram`
sans ajouter de dépôt tiers) est un dépôt **séparé**, géré par les mainteneurs Flathub :

1. **Dépôt de soumission** : fork de
   [github.com/flathub/flathub](https://github.com/flathub/flathub), créer la branche
   `new-pr` avec un sous-module ou un manifeste pointant vers
   `packaging/flathub/io.github.anticitoyen.acvram.yml` de CE dépôt (à une référence
   figée, jamais `main` mouvant) — voir la
   [documentation de soumission Flathub](https://docs.flathub.org/docs/for-app-authors/submission).
2. **Manifeste** : celui déjà présent, `packaging/flathub/io.github.anticitoyen.acvram.yml`
   — runtime GNOME 51, `--device=all`, noyaux précompilés. Les réviseurs Flathub relisent
   TOUT manifeste soumis, y compris les permissions (`--device=all` est large ; à justifier
   dans la description de la PR — accès GPU nécessaire au calcul, pas au réseau).
3. **`extra-data` — la vraie question ouverte** : ce dépôt embarque torch et la fermeture
   CUDA complète en `extra-data` (`torch-cu130.json`, téléchargés à l'installation plutôt
   qu'au *build*, `packaging/flathub/README.md`, pièce 266 h/i) parce que le tout dépasse
   2 Gio et que Flathub REFUSE un bundle publié au-delà de cette taille. Les réviseurs
   Flathub acceptent `extra-data` au cas par cas (des paquets comme les moteurs de jeu
   volumineux l'utilisent), mais ce n'est **pas garanti d'avance** pour un nouveau paquet —
   la pièce 236 (`revue/poste6-piece236-flathub-26-09.md`) recommandait déjà d'écrire aux
   réviseurs AVANT de soumettre, en donnant la taille exacte et la raison. Sans cet accord
   préalable, la PR peut être refusée sur ce seul point après coup.
4. **Revue** : un mainteneur Flathub relit le manifeste, peut demander des changements
   (sandboxing, permissions, métadonnées AppStream — `io.github.anticitoyen.acvram.metainfo.xml`
   existe déjà), puis fusionne. Flathub construit alors LUI-MÊME le paquet à chaque
   changement du manifeste sur cette branche — ce n'est plus `release.yml` qui publie.
5. **Après acceptation** : `packaging/flathub/io.github.anticitoyen.acvram.yml` de ce dépôt
   reste la source que Flathub reconstruit (via le sous-module/pointeur posé à l'étape 1) —
   toute pièce qui le modifie doit garder en tête qu'un changement se répercute aussi côté
   Flathub, pas seulement sur le dépôt gh-pages.

**Comparaison** : le dépôt gh-pages sert AUJOURD'HUI, sans revue ni compte externe, mais
exige d'ajouter un dépôt tiers (`flatpak remote-add`, implicite dans le `.flatpakref`).
Flathub officiel donne une visibilité et une installation sans dépôt tiers, mais exige une
revue humaine et n'est pas garanti d'accepter `extra-data` — décision à prendre par la
personne qui publie, pas par ce dépôt.

---

## Weblate

Les chaînes traduisibles de l'interface (pas le README, qui reste hors Weblate — pièce
236 c, `revue/poste2-piece236c-25-09.md`, un `.md` entier se découpe mal en unités
traduisibles) vivent dans `packaging/langues/_source.json` (les clés,
`packaging/langues/_cles.json`) et sont jointes à chaque release dans
`translations-<version>.zip` (job `translations`, `release.yml:273-282`).

1. **Compte** : créer un compte sur [hosted.weblate.org](https://hosted.weblate.org) (offre
   gratuite pour les projets libres) ou un Weblate auto-hébergé.
2. **Nouveau projet** : *Add new translation project* → nom `acvram`, licence
   `GPL-3.0-or-later`, dépôt source `https://github.com/anticitoyun/anticitoyen-vram`.
3. **Composant `packaging/langues`** : *Add new translation component* →
   - Format de fichier : **JSON nested** (ou *JSON i18next* selon la structure exacte de
     `_source.json` — vérifier au moment de créer le composant, une clé plate suffit ici).
   - Fichier modèle (monolingue de base) : `packaging/langues/_source.json`.
   - Masque des fichiers traduits : `packaging/langues/*.json` (Weblate exclut lui-même
     `_source.json` et `_cles.json` comme fichiers de base, à vérifier à la création).
   - Dépôt de push : le même dépôt GitHub, une branche dédiée (`weblate` par exemple) —
     Weblate ouvre alors des PR vers `main` au lieu de pousser dessus directement.
4. **Langues** : les 31 déjà traduites (`docs/README.*.md`, indépendantes de Weblate) ne
   sont PAS celles de `packaging/langues/` par construction (32 fichiers `README`, mais
   la liste des langues de l'INTERFACE est celle de `packaging/langues/*.json` —
   `tests/test_readme_traductions.py::test_les_31_langues_de_la_gui_sont_celles_du_readme`
   les tient égales). Ajouter dans Weblate exactement les langues déjà présentes comme
   fichiers dans `packaging/langues/`, pas une liste devinée.
5. **Cycle** : les traducteurs modifient sur Weblate → Weblate pousse une PR vers la
   branche dédiée → relecture et fusion normales (REGLES du dépôt s'appliquent à cette PR
   comme à toute autre) → la prochaine release embarque les nouvelles chaînes dans
   `translations-<version>.zip`.
