#!/usr/bin/env python3
"""Pièce 266 d : produit `python3-modules.json` — les dépendances PyPI d'acvram en ROUES BINAIRES (manylinux cp312 ou
py3-none-any), sources `file` (URL + sha256) pour une construction Flathub sans réseau. Remplace flatpak-pip-generator,
qui livre des sdists et fait CONSTRUIRE numpy (meson), tokenizers et pydantic-core (cargo) dans le bac à sable : la
v0.7.0 y est tombée (« BackendUnavailable: Cannot import 'mesonpy' », run 36223036314). Même modèle que sources-torch.sh :
`pip download --only-binary=:all:` (aucun sdist ne passe, une dépendance sans roue fait échouer ici, pas dans le bac à
sable), URL lue dans l'index PEP 503 (roue_url.py), somme calculée sur la roue téléchargée.
    ./sources-pypi.py [--python 3.14] numpy safetensors fastapi "uvicorn[standard]" pydantic pyyaml tokenizers huggingface-hub jinja2 psutil nvidia-ml-py
266 e : la version de Python est celle du RUNTIME (PYTHON_RUNTIME, 3.14 pour GNOME 51 — la v0.7.0 prenait des roues cp312 que le
pip 3.14 du bac à sable ne trouvait pas : « No matching distribution found for httptools ») ; chaque roue est vérifiée : cpXY, abi3
ou py3-none, jamais cpXYt (Python sans GIL). À jouer HORS bac à sable (CI ou poste), depuis packaging/flathub ; les roues déjà servies par torch-cu130.json (torch, triton,
nvidia-*) sont exclues pour ne pas être installées deux fois."""
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roue_url import url_de_la_roue  # noqa: E402

PYPI = "https://pypi.org/simple"
ICI = os.path.dirname(os.path.abspath(__file__))


def python_du_runtime() -> str:
    """« 3.14 » : version de python3 du SDK/runtime du manifeste, tenue dans PYTHON_RUNTIME (vérifiée contre le SDK par release.yml)."""
    return open(os.path.join(ICI, "PYTHON_RUNTIME"), encoding="utf-8").read().strip()


def roue_compatible(fichier: str, version: str) -> bool:
    """La roue s'installe-t-elle sous CPython `version` (« 3.14 ») ? Étiquettes PEP 427 : python cpXY ou py3/py2.py3 ; ABI cpXY,
    abi3 ou none — et jamais cpXYt (sans GIL : une autre ABI, torch cu130 en publie une pour 3.14)."""
    if not fichier.endswith(".whl"):
        return False
    champs = fichier[:-4].split("-")
    if len(champs) < 5:
        return False
    py, abi = champs[-3], champs[-2]
    mineur = int(version.split(".")[1])
    xy = f"cp3{mineur}"

    def _au_plus(x: str, prefixe: str) -> bool:      # « cp310 » / « py3 » / « py38 » : exigence minimale ≤ la version courante
        reste = x[len(prefixe):]
        return x.startswith(prefixe) and (reste == "" or (reste.isdigit() and int(reste) <= mineur))

    abis = abi.split(".")
    if "abi3" in abis:                              # ABI stable : la roue cp310-abi3 s'installe sous tout CPython ≥ 3.10
        return any(_au_plus(x, "cp3") for x in py.split("."))
    py_ok = any(x == xy or x in ("py3", "py2.py3") or _au_plus(x, "py3") for x in py.split("."))
    abi_ok = any(a in (xy, "none") for a in abis)   # jamais cpXYt (sans GIL)
    return py_ok and abi_ok
# pip n'accepte que les étiquettes de plateforme DONNÉES (aucune descente automatique manylinux_2_28 → 2_25) : la roue
# nvidia_cuda_cupti-13.0.85-py3-none-manylinux_2_25_x86_64.whl restait introuvable avec 2_28/2_27/2_17 seules (266 h).
# Toutes les glibc de 2.5 à 2.28 (le runtime GNOME 51 est en 2.4x) + les alias historiques.
PLATEFORMES = [f"manylinux_2_{k}_x86_64" for k in range(28, 4, -1)] + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
DEJA_SERVIES = ("torch", "triton", "nvidia_")            # torch-cu130.json (sources-torch.sh) : torch, triton, roues nvidia-* de CUDA
GARDEES = ("nvidia_ml_py",)                               # … sauf nvidia-ml-py (NVML pour /metrics), qui n'y est pas


def telecharger(exigences: list[str], dossier: str, version: str) -> None:
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--python-version", version, "--implementation", "cp",
           "--index-url", PYPI, "-d", dossier]
    for p in PLATEFORMES:
        cmd += ["--platform", p]
    subprocess.check_call(cmd + exigences)


def source_de_roue(fichier: str, url: str, sha256: str) -> dict:
    """La source `file` d'une roue : URL, somme, et `dest-filename` = le NOM DE ROUE décodé (266 g). Sans lui, flatpak-builder
    garde le dernier segment de l'URL tel quel — `torch-2.14.0%2Bcu130-…whl` — et pip, qui lit la version dans le nom,
    n'y voit pas une roue (« No matching distribution found for torch », run 36227976586)."""
    fichier = urllib.parse.unquote(fichier)
    if "%" in fichier or not fichier.endswith(".whl") or len(fichier[:-4].split("-")) < 5:
        raise ValueError(f"{fichier} : pas un nom de roue valide pour dest-filename")
    return {"type": "file", "url": url, "sha256": sha256, "dest-filename": fichier}


def modules_depuis_roues(dossier: str, resoudre=url_de_la_roue, index: str = PYPI, version: str | None = None) -> list[dict]:
    """Un module flatpak par roue du dossier (hors torch/triton/nvidia-*), trié par nom — `resoudre(fichier, index, sha)` rend l'URL.
    `version` : chaque roue doit s'installer sous ce CPython (266 e), sinon ValueError."""
    mods = []
    for w in sorted(glob.glob(os.path.join(dossier, "*"))):
        fichier = os.path.basename(w)
        if not fichier.endswith(".whl"):
            raise ValueError(f"{fichier} : pas une roue (sdist ?) — --only-binary=:all: devait l'empêcher")
        if version and not roue_compatible(fichier, version):
            raise ValueError(f"{fichier} : roue incompatible avec le Python {version} du runtime (266 e)")
        nom = fichier.split("-")[0]
        if nom.lower().startswith(DEJA_SERVIES) and not nom.lower().startswith(GARDEES):
            continue
        sha = hashlib.sha256(open(w, "rb").read()).hexdigest()
        mods.append({"name": "python3-" + nom.replace("_", "-").lower(), "buildsystem": "simple",
                     "build-commands": [f"pip3 install --verbose --exists-action=i --no-index --find-links=\"file://${{PWD}}\" "
                                        f"--prefix=${{FLATPAK_DEST}} --no-deps --no-build-isolation \"{nom}\""],
                     "sources": [source_de_roue(fichier, resoudre(fichier, index, sha), sha)]})
    return mods


def main(argv: list[str], sortie: str = "python3-modules.json") -> int:
    version = python_du_runtime()
    if argv[:1] == ["--python"]:
        version, argv = argv[1], argv[2:]
    with tempfile.TemporaryDirectory() as d:
        telecharger(argv, d, version)
        mods = modules_depuis_roues(d, version=version)
        taille = sum(os.path.getsize(w) for w in glob.glob(os.path.join(d, "*.whl"))) // 2 ** 20
    json.dump({"name": "python3-modules", "buildsystem": "simple", "build-commands": [], "modules": mods},
              open(sortie, "w", encoding="utf-8"), indent=2)
    print(len(mods), "modules →", sortie, "; Python", version, "; taille roues", taille, "Mio")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
