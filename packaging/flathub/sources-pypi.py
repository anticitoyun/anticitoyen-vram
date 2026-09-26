#!/usr/bin/env python3
"""Pièce 266 d : produit `python3-modules.json` — les dépendances PyPI d'acvram en ROUES BINAIRES (manylinux cp312 ou
py3-none-any), sources `file` (URL + sha256) pour une construction Flathub sans réseau. Remplace flatpak-pip-generator,
qui livre des sdists et fait CONSTRUIRE numpy (meson), tokenizers et pydantic-core (cargo) dans le bac à sable : la
v0.7.0 y est tombée (« BackendUnavailable: Cannot import 'mesonpy' », run 36223036314). Même modèle que sources-torch.sh :
`pip download --only-binary=:all:` (aucun sdist ne passe, une dépendance sans roue fait échouer ici, pas dans le bac à
sable), URL lue dans l'index PEP 503 (roue_url.py), somme calculée sur la roue téléchargée.
    ./sources-pypi.py numpy safetensors fastapi "uvicorn[standard]" pydantic pyyaml tokenizers huggingface-hub jinja2 psutil nvidia-ml-py
À jouer HORS bac à sable (CI ou poste), depuis packaging/flathub ; les roues déjà servies par torch-cu130.json (torch, triton,
nvidia-*) sont exclues pour ne pas être installées deux fois."""
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roue_url import url_de_la_roue  # noqa: E402

PYPI = "https://pypi.org/simple"
PLATEFORMES = ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux_2_24_x86_64", "manylinux_2_17_x86_64",
               "manylinux2014_x86_64", "manylinux_2_12_x86_64", "manylinux2010_x86_64", "manylinux_2_5_x86_64", "manylinux1_x86_64"]
DEJA_SERVIES = ("torch", "triton", "nvidia_")            # torch-cu130.json (sources-torch.sh)


def telecharger(exigences: list[str], dossier: str) -> None:
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--python-version", "3.12", "--implementation", "cp",
           "--index-url", PYPI, "-d", dossier]
    for p in PLATEFORMES:
        cmd += ["--platform", p]
    subprocess.check_call(cmd + exigences)


def modules_depuis_roues(dossier: str, resoudre=url_de_la_roue, index: str = PYPI) -> list[dict]:
    """Un module flatpak par roue du dossier (hors torch/triton/nvidia-*), trié par nom — `resoudre(fichier, index, sha)` rend l'URL."""
    mods = []
    for w in sorted(glob.glob(os.path.join(dossier, "*"))):
        fichier = os.path.basename(w)
        if not fichier.endswith(".whl"):
            raise ValueError(f"{fichier} : pas une roue (sdist ?) — --only-binary=:all: devait l'empêcher")
        nom = fichier.split("-")[0]
        if nom.lower().startswith(DEJA_SERVIES):
            continue
        sha = hashlib.sha256(open(w, "rb").read()).hexdigest()
        mods.append({"name": "python3-" + nom.replace("_", "-").lower(), "buildsystem": "simple",
                     "build-commands": [f"pip3 install --verbose --exists-action=i --no-index --find-links=\"file://${{PWD}}\" "
                                        f"--prefix=${{FLATPAK_DEST}} --no-deps --no-build-isolation \"{nom}\""],
                     "sources": [{"type": "file", "url": resoudre(fichier, index, sha), "sha256": sha}]})
    return mods


def main(exigences: list[str], sortie: str = "python3-modules.json") -> int:
    with tempfile.TemporaryDirectory() as d:
        telecharger(exigences, d)
        mods = modules_depuis_roues(d)
        taille = sum(os.path.getsize(w) for w in glob.glob(os.path.join(d, "*.whl"))) // 2 ** 20
    json.dump({"name": "python3-modules", "buildsystem": "simple", "build-commands": [], "modules": mods},
              open(sortie, "w", encoding="utf-8"), indent=2)
    print(len(mods), "modules →", sortie, "; taille roues", taille, "Mio")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
