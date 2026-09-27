# Pièce 259 (+ 259 b, c, d) — `outils/verifier-release.sh` : vérifier une release fichier par fichier : verdict

poste6, 26/09/2026. Branche poste6-259, code **1a4625acd**, fusion **1c96f473c** ; 259 b **91c62b6ae** ; 259 c **fb88ac5b4**,
**1e93d1742** (fusionnés par poste6-266h) ; 259 d **d1435212e** (fusion poste6-verif-070).

**Quoi.** `outils/verifier-release.sh <tag> [--dossier D | --simule D] [--sans-flatpak | --flatpak-installer | --flatpak-doctor]` :
fichiers annoncés par `release.yml` (un seul de chaque), SHA256SUMS, `.deb` lu sans installation (dpkg-deb : Package,
Version, entrées, `usr/bin/acvram`), AUR (PKGBUILD, pkgver, sha256sums), translations, `.flatpakref` (GPGKey), installation
Flatpak dans un `FLATPAK_USER_DIR` dédié puis `acvram doctor` dans le bac à sable ; repli sur l'API publique de GitHub quand
`gh` n'est pas authentifié (aucun jeton ne transite) ; refus de `/tmp` (un rejeu doit retrouver les fichiers).
259 b : job `sommes` de `release.yml` (SHA256SUMS après tous les jobs qui joignent, `!cancelled()`). 259 c : doctor lancé
depuis le dossier de la release (`_garde_arbre`), `--reinstall`. 259 d : ce qui est joint = `liste-release.txt` (un fichier
d'un téléchargement précédent va dans `hors-release/`, jamais effacé) ; app retirée avant `install --from` (`--reinstall`
ne traverse pas les remotes).

**Tests.** `tests/test_verifier_release_259.py` (release simulée : complète TENUE, somme fausse vue, sans sommes dit,
incomplète FAUX, refus de /tmp, ancien bundle FAUX, 259 d ×2) ; `tests/test_release_sommes_259b.py`. Sous carte.sh : 8 verts
à la 259 (1c96f473c), 8 à la 259 d (témoins rouges sur l'ancien script).

**Résultat.** Trois releases vérifiées avec lui le 26/09 — v0.7.0 (P1/P3 TENUS, réserve 241 fermée), v0.7.1 (fichiers et
sommes), v0.7.2 (P1 + P2, doctor 0, « ok vision ») : `poste6-serie266-flatpak-verdict-26-09.md` § 4-5. Ce qu'il a trouvé
en chemin : GitHub Pages non activé, `acvram doctor` à 1 partout (070 b), extras vision absents (273), ses deux propres
bogues (259 d). **Décision** : outil de référence avant tout tag ; rpm SAUTÉ tant que `rpm` manque au poste.
