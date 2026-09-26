# Flathub — `io.github.anticitoyen.acvram` (pièce 236, 26/09/2026)

Ce répertoire est la charge utile Flathub : manifeste, metainfo, sources Python hors ligne. **Rien n'est soumis ici** : la
soumission à Flathub (dépôt `flathub/io.github.anticitoyen.acvram`) est le geste de l'utilisateur ; le job de CI qui construit et
lint le paquet est branché par poste3 dans `release.yml` (modèle : `rog-flare2-anime-matrix/.github/workflows/flathub.yml`).

## Fichiers
| fichier | rôle | qui le produit |
|---|---|---|
| `io.github.anticitoyen.acvram.yml` | manifeste (runtime GNOME 51, `--device=all`, noyaux précompilés, roues hors ligne) | ce répertoire |
| `io.github.anticitoyen.acvram.metainfo.xml` | AppStream (en/fr, captures, releases) | ce répertoire ; `<release>` à tenir par version |
| `python3-modules.json` | dépendances PyPI en sources `file`, ROUES binaires seulement (266 d : plus de sdist, plus de meson ni cargo dans le bac à sable) | `./sources-pypi.py numpy safetensors fastapi "uvicorn[standard]" pydantic pyyaml tokenizers huggingface-hub jinja2 psutil nvidia-ml-py` (réseau) |
| `torch-cu130.json` | torch 2.14 cu130 + fermeture CUDA complète (cuda-toolkit, cudnn, nccl, cusparselt, nvshmem… : libtorch les lie) en sources **extra-data** — téléchargées à l'installation, dépaquetées par `apply_extra` dans /app/extra/site-packages (266 h/i) | `./sources-torch.sh 2.14.0 [3.14]` (réseau) |
| `PYTHON_RUNTIME` | version de python3 du SDK/runtime GNOME 51 (3.14) : celle des roues (sources-pypi.py, sources-torch.sh), des chemins `cleanup` du manifeste et du .so précompilé (job noyaux-precompiles) — release.yml la vérifie contre le SDK réel (266 e) | à mettre à jour avec `runtime-version` |
| `build/noyaux/` (hors dépôt) | `acvram_kernels.so` et port Marlin précompilés par empreinte, sm_120 (+ sm_89, sm_86) | CI avec nvcc (script à écrire : `tools/noyaux-precompiles.sh`) |

## Construction locale (réseau pour les deux générateurs, puis hors ligne)
```bash
flatpak install --user flathub org.gnome.Platform//51 org.gnome.Sdk//51 org.flatpak.Builder
cd packaging/flathub && ./sources-torch.sh 2.14.0 && cd -
flatpak run org.flatpak.Builder --user --force-clean --sandbox --repo=repo build packaging/flathub/io.github.anticitoyen.acvram.yml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest packaging/flathub/io.github.anticitoyen.acvram.yml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder repo repo
```

## Ce qui reste avant une soumission (dans l'ordre)
1. ~~Chargement des noyaux précompilés sans nvcc~~ — **fait (240)** : `ACVRAM_KERNELS_PRECOMPILES`, `kernels._precompile_utilisable`,
   `build_info()["precompile"]` ; tests `tests/test_noyaux_precompiles_240.py`.
2. ~~Produire les .so~~ — **fait (241)** : `python -m acvram.kernels.precompiles --dossier build/noyaux --archs 12.0,8.6` dans un conteneur
   CUDA sans carte (`job-noyaux-precompiles.yml`, à coller dans `release.yml`) ; garde : `tests/test_noyaux_precompiles_ci_241.py`.
   Reste le **port Marlin** (`marlin_port`, .so à part, 161) : même geste, pièce à part.
3. Générer `python3-modules.json` et `torch-cu130.json`, construire, lint, publier le DÉPÔT (pas un bundle : les extra-data n'y tiennent pas et il dépasserait 2 Gio), essayer sur une machine avec le pilote NVIDIA :
   `flatpak run io.github.anticitoyen.acvram` puis `flatpak run --command=acvram io.github.anticitoyen.acvram doctor`.
4. Écrire aux réviseurs Flathub AVANT la soumission (taille, voir la note `revue/poste6-piece236-flathub-26-09.md`).
