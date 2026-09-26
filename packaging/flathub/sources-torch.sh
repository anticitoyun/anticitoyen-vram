#!/usr/bin/env bash
# Pièce 236 : produit packaging/flathub/torch-cu130.json — les roues PyTorch cu130 (torch, triton) et les roues nvidia-* qu'elles
# tirent, en sources `file` (URL + sha256) pour une construction Flathub SANS réseau. À jouer HORS bac à sable, avec réseau
# (CI ou poste) ; le résultat se versionne. Roues cp<PYTHON_RUNTIME> manylinux x86_64 (le SDK GNOME 51 porte Python 3.14, vérifié par release.yml).
#   ./sources-torch.sh 2.14.0 [3.14]     # → torch-cu130.json ; 2e argument = Python du runtime (défaut : PYTHON_RUNTIME, 266 e)
set -euo pipefail
V="${1:?version de torch, ex. 2.14.0}"
PYV="${2:-$(cat "$(dirname "$0")/PYTHON_RUNTIME")}"
INDEX="https://download.pytorch.org/whl/cu130"
D=$(mktemp -d)
python3 -m pip download --no-deps --only-binary=:all: --python-version "$PYV" --implementation cp --platform manylinux_2_28_x86_64 \
    --index-url "$INDEX" -d "$D" "torch==$V" triton
# les roues nvidia-* exigées par cette roue torch (Requires-Dist), résolues sur le même index puis PyPI
python3 - "$D" "$INDEX" "$(cd "$(dirname "$0")" && pwd)" "$PYV" <<'PY'
import glob, json, os, re, subprocess, sys, zipfile, hashlib
d, index = sys.argv[1], sys.argv[2]
sys.path.insert(0, sys.argv[3])              # packaging/flathub : roue_url.py, sources-pypi.py (roue_compatible)
from roue_url import url_de_la_roue
import importlib.util as _iu
_sp = _iu.spec_from_file_location("sources_pypi", os.path.join(sys.argv[3], "sources-pypi.py")); _m = _iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
pyv = sys.argv[4]
torch = glob.glob(os.path.join(d, "torch-*.whl"))[0]
with zipfile.ZipFile(torch) as z:
    meta = next(n for n in z.namelist() if n.endswith("METADATA"))
    reqs = [l.split(":", 1)[1].strip() for l in z.read(meta).decode().splitlines() if l.startswith("Requires-Dist: nvidia")]
noms = sorted({re.split(r"[ ;=<>!]", r)[0] for r in reqs} - {"nvidia-cudnn-cu13", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13", "nvidia-cusparselt-cu13"})
for n in noms:  # cudnn, nccl, nvshmem, cusparselt : retirés (cleanup du manifeste) — inutilisés par acvram
    subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", pyv, "--implementation", "cp",
                           "--platform", "manylinux_2_28_x86_64", "--index-url", index, "--extra-index-url", "https://pypi.org/simple", "-d", d, n])
mods = []
for w in sorted(glob.glob(os.path.join(d, "*.whl"))):
    nom = os.path.basename(w).split("-")[0]
    if not _m.roue_compatible(os.path.basename(w), pyv):     # 266 e : cpXY / abi3 / none, jamais cpXYt
        raise SystemExit(f"{os.path.basename(w)} : roue incompatible avec le Python {pyv} du runtime")
    sha = hashlib.sha256(open(w, "rb").read()).hexdigest()
    # 266 b : l'URL vient de l'index (PEP 503, roue_url.py) — l'ancien appel passait à `pip download` une option
    # qu'il n'a pas (elle n'existe que pour `pip install`) : job flatpak de la v0.7.0 rouge, étape « sources Python »
    url = url_de_la_roue(os.path.basename(w), index, sha)
    mods.append({"name": "python3-" + nom, "buildsystem": "simple",
                 "build-commands": [f"pip3 install --verbose --exists-action=i --no-index --find-links=\"file://${{PWD}}\" --prefix=${{FLATPAK_DEST}} --no-deps \"{nom}\" --no-build-isolation"],
                 "sources": [{"type": "file", "url": url, "sha256": sha}]})
json.dump({"name": "torch-cu130", "buildsystem": "simple", "build-commands": [], "modules": mods},
          open("torch-cu130.json", "w"), indent=2)
print(len(mods), "modules →", "torch-cu130.json", "; taille roues", sum(os.path.getsize(w) for w in glob.glob(os.path.join(d, "*.whl"))) // 2**20, "Mio")
PY
