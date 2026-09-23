# Livraison acvram-parc 0.1.3 (Laure, 23/09, pièce 84)

## Cause de la liste vide après installation

`acvram-parc` n'a pas de `postinst` : `dpkg -i` installe les fichiers, jamais
`parc-installer`. Un poste neuf n'a donc ni `~/.config/acvram-parc/parc.toml`
ni `~/.kimi-code/config.toml`. `charger_parc()`
(`parc/lib/menu_modeles/parc.py:139`) ouvrait `CONFIG` sans vérifier son
existence : `FileNotFoundError` → `RuntimeError` → `_recharger()`
(fenetre.py) l'attrapait et se contentait d'un toast — le store restait vide
sans explication actionnable. **C'est la cause de la pièce 84** ; ce n'est
**pas** la cause du « texte qui disparaît au tri » (pièce 83) : la pièce 83 a
été mesurée sur un parc avec 62 alias réels (sans fiche), pas un parc vide —
un parc réellement vide n'a aucune ligne, donc rien à faire disparaître ;
confirmé à nouveau ici en rejouant `tests/test_gui_tri_colonnes.py` (14/14
verts, inchangé).

## Ce qui a été ajouté

- **`charger_parc()`** (`parc.py:136`) : `CONFIG` absent → parc vide, pas une
  faute. Un fichier PRÉSENT mais invalide (TOML cassé) reste une vraie
  `RuntimeError` (contrôle négatif dans le test).
- **Bouton « Dossiers des modèles… »** (en-tête, les deux menus) :
  `Gtk.FileDialog.select_multiple_folders` (GTK ≥ 4.10, disponible ici en
  4.22), puis balayage.
- **Bouton « Rebalayer »** (en-tête) : relance le balayage sur les dossiers
  déjà connus, sans en choisir de nouveaux.
- **Balayage = `parc-installer` existant, rien de neuf écrit** (consigne
  « cherche d'abord ») : les deux boutons appellent
  `parc-installer --auto --sans-balayage --sans-minuteur [--racine DIR ...]`
  — le même outil, la même classification (`classer()`), la même écriture
  des TSV et de `~/.kimi-code/config.toml` que l'installation manuelle.
  `--sans-balayage` borne le balayage aux dossiers déjà connus (`parc.toml`)
  + ceux donnés : **aucun balayage automatique de tous les disques montés
  depuis un clic de menu** (ce balayage-là, `disques_montes()`, reste le
  comportement de `parc-installer` lancé à la main, pas celui d'un clic
  GUI). `parc-installer` persiste lui-même les racines choisies dans
  `parc.toml` (`ecrire_parc_toml`, déjà existant) — c'est le fichier que
  Jerome demandait de relire d'abord.
- **Premier lancement, liste vide → proposition directe** : `_recharger()`
  ouvre le sélecteur de dossiers automatiquement, une seule fois par
  session, jamais sous `ACVRAM_GUI_TEST` (un `Gtk.FileDialog` réel ne rend
  jamais sous test headless).

## Tests joués avant livraison (carte libre)

- `tests/test_classer_dossiers.py` (neuf) : `classer()` sur les trois
  classes (acvram/manifeste, GGUF fichier et dossier, HF vLLM/AWQ) + dossier
  vide ; chaque cas positif cassé une fois (retrait du signe qui fait la
  classe) → redevient `None`.
- `tests/test_gui_dossiers_modeles.py` (neuf) : `charger_parc()` sans config
  → `[]` (pas de RuntimeError) / avec config cassée → RuntimeError ;
  « Rebalayer » spawn exactement `parc-installer --auto --sans-balayage
  --sans-minuteur` sans `--racine`, dans les deux menus.
- Rejoué sans régression : `test_gui_tri_colonnes.py` (14),
  `test_gui_crochet_clic.py` (8), `test_paquet_parc.py`,
  `test_lanceur_source.py`, `test_lanceur_parc_vide.py`,
  `test_commande_unique_parc.py`, `test_parc_lib_resolution.py`.
- **Total : 56 passed.**

## Paquet

- `parc/VERSION` : 0.1.2 → 0.1.3.
- `acvram-parc_0.1.3_amd64.deb`, sha256 `<jeton-masqué>`.
- `dpkg-deb -f … Version` = 0.1.3 ; extrait, compilé (`py_compile`
  fenetre.py + parc.py), aucun `anticitoyenpartage` dans le paquet.

Rien installé, rien publié.
