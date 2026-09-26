# Série 266 b → k et 259 : la livraison Flatpak de la 0.7.0, causes, correctifs, preuves — et verdict verif-070

poste6, 26/09/2026. Pièces données par chef. Une seule question de bout en bout : **un utilisateur qui lit la ligne
« Flatpak » du README obtient-il un acvram qui charge ses noyaux précompilés sur sa carte ?** La 0.7.0 a répondu « non »
onze fois avant de répondre « oui ». Chaque ligne ci-dessous est une cause réelle, lue dans le journal d'un run, jamais
devinée ; le sha est celui de `git log origin/main`.

## 1. La chaîne, cause par cause

| Pièce | Cause lue dans le run | Correctif (sha main) | Garde (rouge sur l'ancien) |
|---|---|---|---|
| 266 | deb : `pip --break-system-packages` ne désinstalle pas `typing_extensions` (paquet Debian sans RECORD) ; aur : `mkdir /build/pkg` sans `-p` ; noyaux : `tee build/noyaux/…` sans dossier, `Python.h` et numpy absents | `--ignore-installed`, `mkdir -p`, `python3-dev` + numpy (e8fa4b18c, fusion e3c345bf7) | `tests/test_release_dossiers_266.py` |
| 259 | `gh` non authentifié dans ma session : rien ne se vérifie | `outils/verifier-release.sh` : repli sur l'API publique, SHA256SUMS, dpkg-deb, AUR, Flatpak (1a4625acd, fusion 1c96f473c) | `tests/test_verifier_release_259.py` |
| 259 b | aucun SHA256SUMS joint : un fichier corrompu passe | job `sommes` après tous les jobs qui joignent, `!cancelled()` (91c62b6ae) | `tests/test_release_sommes_259b.py` |
| 266 b | `pip download --dry-run` ne rend pas l'URL d'une roue torch ; le CDN `download-r2.pytorch.org` refuse `Python-urllib` ; le job au TAG lit `packaging/flathub` tel qu'il était au tag (absent) | `roue_url.py` (index PEP 503, User-Agent), `git checkout ${{ github.sha }} -- packaging/flathub` avec `fetch-depth: 0` (3e7b3b63b, fusion a296e1348) | `tests/test_sources_torch_266b.py`, `tests/test_release_flatpak_outillage_266b.py` |
| 266 c | `flatpak-builder --force-clean` efface `build/`, où vivaient les noyaux précompilés | builddir `construction-flatpak`, manifeste `skip` (8cd94dbe9, fusion dec54df89) | `tests/test_release_flatpak_noyaux_266c.py` |
| 266 d | des sdist dans les sources : sans réseau dans le bac à sable de construction, rien ne se compile | `sources-pypi.py`, roues binaires seulement, `nvidia-ml-py` conservé (49049744b + 5bafad64e, fusion d71d2eeeb) | `tests/test_flathub_roues_266d.py` |
| 266 e | roues et `.so` précompilé au Python de l'hôte (3.13) alors que le runtime fd 26.08 porte 3.14 | `PYTHON_RUNTIME` = 3.14, `roue_compatible`, SOABI dans l'empreinte des noyaux, job noyaux sous deadsnakes 3.14 (0647f07eb, fusion cc2904200) | `tests/test_flathub_python_266e.py` |
| 266 f | `cat: PYTHON_RUNTIME: No such file` : le job noyaux lit aussi l'outillage au tag ; le conteneur `nvidia/cuda` n'a pas git (checkout = archive sans `.git`) | second `actions/checkout` de `github.sha` dans `outillage-ref/`, `cp -a` (e60ab12a3, 8357c0088, a50c94df6, fusion 167af3a25) | garde 266 b généralisée |
| 266 g | `dest-filename` gardait `%2B` : `torch-2.14.0%2Bcu130…whl` n'est pas une roue pour pip | nom décodé sur toute source `file` (01f7b874e, fusion 075461731) | `tests/test_flathub_dest_filename_266g.py` |
| 266 h | fermeture CUDA de torch cu130 incomplète (cudnn, nccl, cusparselt…) : `import torch` casse dans le bac à sable | toutes les `Requires-Dist` lues, `ELAGAGE = {}`, toutes les `PLATEFORMES` manylinux 2_5…2_28 (pip n'élargit pas vers le bas : cupti 2_25) (b29c70e4d) | `tests/test_flathub_torch_deps_266h.py` |
| 266 i | un bundle seul ne porte pas d'extra-data (« Extra data missing in detached metadata ») et GitHub plafonne un fichier de release à 2 Gio ; `build-update-repo --gpg-sign` ne signe pas le commit | dépôt OSTree sur `gh-pages/flatpak` (+ `.nojekyll`), `flatpak build-sign` puis `build-update-repo --generate-static-deltas`, `.flatpakref` avec `GPGKey`, ancien bundle retiré, 32 README (44b81c8f3, 7687fce66, fa93c4b3c, fusion 9d05d95d3) | `tests/test_flathub_depot_266i.py`, `test_readme_canaux_release` |
| 259 c | rejeu sur l'installation dédiée : « already installed » rendait FAUX ; `flatpak run` garde le cwd et `_garde_arbre` refuse l'import depuis `/app` dans un arbre acvram | `--reinstall`, doctor lancé depuis le dossier de la release (1e93d1742, fb88ac5b4) | `tests/test_verifier_release_259.py` |
| 266 j | run 36234968029 : `empty ident name` sur `commit-tree` (le coureur n'a pas d'identité git) → sha vide → `git branch gh-pages ''` → 128 | `-c user.name='github-actions[bot]' -c user.email='41898282+github-actions[bot]@users.noreply.github.com'` sur commit-tree et commit, `[ -n "$RACINE" ]`, `\|\| true` retiré (a9df632f5, fusion edbd73764) | 2 tests 266 j dans `tests/test_flathub_depot_266i.py` |
| Pages | run 36236711208 vert, mais `install --from` : « No such ref … in remote acvram » — le site `anticitoyun.github.io` rendait 404 partout : **GitHub Pages n'était pas activé** (action du propriétaire, hors dépôt) | chef : Pages activé (source gh-pages, racine), statut `built`, `flatpak/summary` 200 | verif-070 (rejeu) |
| 259 d | le vérificateur jugeait le dossier local : l'ancien bundle d'un téléchargement précédent y traînait → « encore joint » à tort ; « already installed » : `--reinstall` ne traverse pas les remotes (app posée depuis `acvram-origin`, `.flatpakref` propose `acvram`) | ce qui est joint = `liste-release.txt`, l'intrus va dans `hors-release/` (jamais effacé) ; app retirée avant `install --from` (d1435212e) | 2 tests 259 d dans `tests/test_verifier_release_259.py` |
| 070 b | `acvram doctor` sortait **1 partout, hôte compris** : `_doctor_eco` (poste7-eco 19/09) appelle `subprocess.run` sans que `cli.py` importe `subprocess` — `NameError` en dernière ligne, après le rapport, invisible à qui ne lit que les « ok / ECHEC » | `import subprocess` (77343eed0) | `tests/test_doctor_eco_070b.py` (sudo factice ; rouge sur l'ancien) |
| 266 k | CHANGELOG 0.7.1 sans entrée Paquets | entrée « Paquets » (origin/poste6-266k 9e9570182, à fusionner avec la 0.7.1) | `tests/test_version_changelog_223.py` |

Trois leçons qui dépassent le Flatpak :
* **un job de release lancé au tag lit l'outillage du tag**, pas celui du workflow ; tout outil de construction se prend de
  `github.sha` (266 b, 266 f) ;
* **le coureur GitHub n'a aucune identité git** : tout `git commit` d'un workflow la porte par `-c` (266 j) ;
* **un bac à sable de construction est hors ligne** : roues binaires épinglées, fermeture complète, rien à compiler (266 d, h).

## 2. Preuve locale de la 266 i (avant le run)

Sous carte.sh (ACVRAM_NOM=poste6-266i, 12:03:36 → 12:03:38), dépôt local signé par la même clé, installation `--from` dans
un `FLATPAK_USER_DIR` dédié (`~/.cache/acvram/releases/v0.7.0/flatpak-user`), extra-data 5,2 Go dépaquetées par `apply_extra` :
`import torch OK 2.14.0+cu130 python 3.14.7`, `cuda dispo True` (RTX 5090), convolution cuDNN et matmul cuBLAS dans le bac
à sable. Gardes de traduction sur fa93c4b3c : 1 855 verts. Carnet `acvram-memoire/poste6.md`, 26/09 12 h.

## 3. verif-070 : scellé

Écrit avant la mesure (`scratchpad/poste6-p070-26-09/prise-verif-070.sh`) :
* **P1** : `outils/verifier-release.sh v0.7.0 --flatpak-installer` à sec rend **0 FAUX** — tout fichier attendu présent
  (deb, rpm, src.rpm, aur, translations, `.flatpakref`, SHA256SUMS), sommes exactes, `GPGKey` dans le `.flatpakref`,
  installation par le dépôt gh-pages réussie, extra-data dépaquetées ;
* **P2** : `--flatpak-doctor` sous carte.sh (ACVRAM_NOM=poste6-verif-070) sort 0 et nomme les **noyaux précompilés cp314
  chargés** — ni « compilation », ni « référence » ; c'est ce qui ferme la réserve 241 ;
* **P3** : `origin/gh-pages` ne contient que `flatpak/` et `.nojekyll`.
Issues défavorables nommées : (a) run rouge, rien à vérifier ; (b) SHA256SUMS incomplet ; (c) `install --from` refusé
(gh-pages non servi, signature) ; (d) doctor retombe sur la référence, 241 reste ouverte.

## 4. verif-070 : verdict (run 36236711208 vert, release v0.7.0 relue le 26/09 à 13 h 09, vérificateur d1435212e)

**P1 TENU.** À sec : 7 fichiers par l'API publique (deb 6 229 412 o, rpm 934 739, src.rpm 12 103 927, aur 1 562,
translations 208 096, `.flatpakref` 1 788, SHA256SUMS 542) ; `SHA256SUMS vérifié` ; deb : Package/Version 0.7.0, 274
entrées, `usr/bin/acvram` ; aur : PKGBUILD + .SRCINFO, pkgver 0.7.0, sha256sums renseigné ; translations 34 fichiers ;
rpm : nom et taille seulement (`rpm` absent du poste, SAUTÉ) ; ancien bundle absent de la release ; `.flatpakref` avec
`GPGKey` ; `flatpak install --from` par Pages dans l'installation dédiée : 0.7.0, **38 paquets extra-data dépaquetés**.
Verdict du script : TENU, code 0. Avant l'activation de Pages, la même installation a été faite depuis le clone de
`gh-pages` en `file://` avec la même `GPGKey` (172,3 Mo, 38 paquets, code 0) : le dépôt construit par le CI est bon
indépendamment du service.

**P3 TENU.** `gh-pages` = `.nojekyll` + `flatpak/` (4 290 entrées, 105 Mio : objets 84, deltas 21, `summary` 9 222 o +
`summary.sig`), commit 03dd26e95 par `github-actions[bot]`, rien d'autre.

**P2 : réserve 241 FERMÉE, doctor FAUX pour deux causes étrangères aux noyaux.** Sous carte.sh (ACVRAM_NOM
poste6-verif-070, carte obtenue après 355 s d'attente derrière chef-suite-071), dans le bac à sable :
`build_info()` → `available True`, `precompile /app/lib/acvram/noyaux/b97645914e4bea8b/acvram_kernels.so`, `device_caps
['sm_120']`, `nvcc` absent (`CUDA_HOME` None, Python 3.14.7) : **le `.so` cp314 construit par le job noyaux-precompiles
se charge sur la RTX 5090 dans le bac à sable, sans compilateur** — c'est exactement ce que la 241 réservait. Le doctor
lui-même le confirme en clair : `torch 2.14.0+cu130, cuda 13.0`, `noyaux CUDA fusionnes compiles pour sm_120`, noyaux
processeur AVX2+FMA, `cuda:0 nvfp4 -> cuda-fusionne`. Mais il sort **1**, sur l'hôte comme dans le bac à sable :
1. **070 b** (code, tous canaux) : `NameError: name 'subprocess' is not defined` dans `_doctor_eco`, dernière ligne du
   rapport ; corrigé 77343eed0, hors de la v0.7.0 publiée — le Flatpak de la 0.7.0 gardera ce code 1 jusqu'à la 0.7.1 ;
2. **extras** (décision) : `ECHEC transformers est absent`, `ECHEC PIL est absent` — le doctor les exige, `pyproject`
   les range dans l'extra `vision` (« indispensable » pour un modèle multimodal), et la liste de `sources-pypi.py` dans
   `release.yml` (numpy … nvidia-ml-py) ne les prend pas. Deux issues : le Flatpak embarque `vision` (transformers,
   pillow — un Flatpak se veut complet, mon choix), ou le doctor les rétrograde en `alerte` hors multimodal. À chef →
   poste4.
Le `.deb` ne déclare pas transformers non plus (`Depends` : python3, venv, pip, gi, webkit, psutil) : même question pour
lui — hors de ce verdict, non mesuré ici.

Issues nommées au scellé : (a) non ; (b) non ; (c) **oui, une fois** (Pages non activé — corrigé par le propriétaire,
rejeu TENU) ; (d) non : les noyaux ne retombent pas sur la référence. Ce que ce verdict ne dit pas : rien sur le rpm
(outil absent), rien sur un poste sans pilote nvidia 595, rien sur la taille réelle tirée par l'utilisateur (extra-data
≈ 3 Go annoncés par le README, non chronométrés ici).

Fichiers : `~/.cache/acvram/releases/v0.7.0/{verif-070.log,flatpak-install.log,flatpak-doctor.txt,liste-release.txt}`,
`scratchpad/poste6-p070-26-09/{prise-verif-070.sh,prise-p2-noyaux.sh}`.

## 5. verif-071 et verif-072 (26/09, 14 h) — deux releases publiées à vingt minutes d'écart

Runs 36238780414 (v0.7.1, flatpak fini 11:43:37Z) et 36239670393 (v0.7.2, 12:03:59Z), tous deux verts. **Version
servie** : `gh-pages` 6fe7d2de2 « flatpak : dépôt v0.7.2 », ref `app/…/master` = 2a2036cca2413f16 dans git ET par Pages,
summary Pages = summary git (sha a94d37f07c03, 9 222 o) — aucune relance nécessaire, l'ordre était le bon. Piège :
`flatpak remote-ls/remote-info` dans l'installation dédiée montrait encore le commit 25027dac (11:02, la 0.7.0) — cache
local du summary, pas Pages ; remote retirée avant les prises. Ce hasard d'ordre est ce que la 266 l (fusionnée) rend
impossible : job `flatpak` sérialisé, dépôt amorcé depuis l'existant, ref qui ne recule pas, gh-pages en un commit orphelin.

**verif-072 TENU, P1 et P2** (vérificateur d1435212e, 14:09 → 14:11) : 7 fichiers, SHA256SUMS vérifié, deb 274 entrées,
aur pkgver 0.7.2, translations 34, `.flatpakref` GPGKey, install par Pages : **0.7.2, 38 paquets extra-data** ; sous
carte.sh, `acvram doctor` dans le bac à sable **sort 0** (070 b), `noyaux CUDA fusionnes compiles pour sm_120`, noyaux
processeur AVX2+FMA, **`ok    vision (transformers, PIL)`** (273 : l'extra est embarqué, 44 roues cp314), seule alerte :
pas de droit sudo sur nvidia-smi dans le bac à sable (attendu, éco à l'horloge libre). rpm SAUTÉ (outil absent).

**verif-071 TENU, fichiers et sommes seulement** (`--sans-flatpak`, décision chef) : 7 fichiers, SHA256SUMS vérifié, deb
274 entrées, aur pkgver 0.7.1, translations 34. Le bras Flatpak n'a pas été joué : **un `.flatpakref` ancien mène à la
dernière version publiée** — le dépôt n'a qu'un ref, `master`, et `acvram-0.7.1.flatpakref` installe la 0.7.2. C'est le
comportement voulu d'un canal Flatpak (une ligne à ajouter à la section Installer à la prochaine vague) ; le Flatpak
0.7.1 tel que construit (doctor 1, 070 b) n'est plus observable, seuls le .deb et pip de la 0.7.1 gardent ce code 1.
Une première prise 071 avec le bras Flatpak avait démarré (2,8 Go tirés) : arrêtée par les PID de ma propre chaîne
(prise → vérificateur → `flatpak install`, lignes de commande et parents relus), installation partielle effacée.

Fichiers : `~/.cache/acvram/releases/v0.7.2/{verif-072.log,flatpak-doctor.txt}`, `~/.cache/acvram/releases/v0.7.1/verif-071-sans-flatpak.log`,
`scratchpad/poste6-p07{1,2}-26-09/`.
