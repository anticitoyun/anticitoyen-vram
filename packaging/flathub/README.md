# Flathub — `io.github.anticitoyen.acvram` (pièce 236, 26/09/2026)

Ce répertoire est la charge utile Flathub : manifeste, metainfo, sources Python hors ligne. **Rien n'est soumis ici** : la
soumission à Flathub (dépôt `flathub/io.github.anticitoyen.acvram`) est le geste de l'utilisateur ; le job de CI qui construit et
lint le paquet est branché par poste3 dans `release.yml` (modèle : `rog-flare2-anime-matrix/.github/workflows/flathub.yml`).

## Fichiers
| fichier | rôle | qui le produit |
|---|---|---|
| `io.github.anticitoyen.acvram.yml` | manifeste (runtime GNOME 51, `--device=all`, noyaux précompilés, roues hors ligne) | ce répertoire |
| `io.github.anticitoyen.acvram.metainfo.xml` | AppStream (en/fr, captures, releases) | ce répertoire ; `<release>` à tenir par version |
| `python3-modules.json` | dépendances PyPI en sources `file` | CI : `flatpak-pip-generator --runtime org.gnome.Sdk//51 numpy safetensors fastapi "uvicorn[standard]" pydantic pyyaml tokenizers huggingface-hub jinja2 psutil nvidia-ml-py` |
| `torch-cu130.json` | torch 2.14 cu130, triton 3.8, roues nvidia-* (sans cudnn/nccl/nvshmem/cusparselt) | `./sources-torch.sh 2.14.0` (réseau) |
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
1. **Chargement des noyaux précompilés sans nvcc** : `acvram/kernels/__init__.py` passe par `torch.utils.cpp_extension.load` (ninja, nvcc
   pour les drapeaux d'architecture) ; sans nvcc il retombe sur le chemin de référence, lent. Il faut un chemin « `.so` présent dans
   `ACVRAM_KERNEL_CACHE` à la bonne empreinte → `torch.ops.load_library` / import direct, sans ninja » (≈ 30 lignes, test à sec qui
   casse si l'empreinte diffère). Même geste pour le port Marlin (déjà `load_library` depuis la 161).
2. `tools/noyaux-precompiles.sh` : compile les deux `.so` pour sm_120 (et sm_89/sm_86 si on les publie) et les range sous
   `build/noyaux/kernels-<empreinte>/`, avec l'empreinte calculée comme `kernels/__init__.py` la calcule.
3. Générer `python3-modules.json` et `torch-cu130.json`, construire, lint, essayer sur une machine avec le pilote NVIDIA :
   `flatpak run io.github.anticitoyen.acvram` puis `flatpak run --command=acvram io.github.anticitoyen.acvram doctor`.
4. Écrire aux réviseurs Flathub AVANT la soumission (taille, voir la note `revue/poste6-piece236-flathub-26-09.md`).
