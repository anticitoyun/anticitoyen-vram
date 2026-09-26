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
# 266 h : toutes les étiquettes manylinux (pip ne descend pas de 2_28 vers 2_25 tout seul) — la même liste que sources-pypi.py
PLATEFORMES=$(python3 -c "import importlib.util,os,sys; s=importlib.util.spec_from_file_location('sp', os.path.join(sys.argv[1], 'sources-pypi.py')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(' '.join('--platform ' + p for p in m.PLATEFORMES))" "$(dirname "$0")")
# shellcheck disable=SC2086
python3 -m pip download --no-deps --only-binary=:all: --python-version "$PYV" --implementation cp $PLATEFORMES \
    --index-url "$INDEX" -d "$D" "torch==$V" triton
# les dépendances de cette roue torch (Requires-Dist, 266 h), résolues sur le même index puis PyPI
python3 - "$D" "$INDEX" "$(cd "$(dirname "$0")" && pwd)" "$PYV" <<'PY'
import glob, json, os, re, subprocess, sys, zipfile, hashlib
d, index = sys.argv[1], sys.argv[2]
sys.path.insert(0, sys.argv[3])              # packaging/flathub : roue_url.py, sources-pypi.py (roue_compatible)
from roue_url import url_de_la_roue
import importlib.util as _iu
_sp = _iu.spec_from_file_location("sources_pypi", os.path.join(sys.argv[3], "sources-pypi.py")); _m = _iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
pyv = sys.argv[4]
_st = _iu.spec_from_file_location("sources_torch", os.path.join(sys.argv[3], "sources_torch.py")); _t = _iu.module_from_spec(_st); _st.loader.exec_module(_t)
torch = glob.glob(os.path.join(d, "torch-*.whl"))[0]
with zipfile.ZipFile(torch) as z:
    meta = next(n for n in z.namelist() if n.endswith("METADATA"))
    metadata = z.read(meta).decode()
# 266 h : TOUTES les dépendances de torch (cuda-toolkit[extras], cuda-bindings, sympy, networkx, …), marqueurs évalués pour le
# runtime, élagage justifié (cudnn, cusparselt, nccl, nvshmem : sources_torch.ELAGAGE) — téléchargées AVEC leurs dépendances
exigences, exclues = _t.exigences_de_torch(metadata, pyv)
print("exigences de torch :", exigences, "; élaguées :", sorted(exclues))
plateformes = [a for p in _m.PLATEFORMES for a in ("--platform", p)]
subprocess.check_call([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--python-version", pyv, "--implementation", "cp"]
                      + plateformes + ["--index-url", index, "--extra-index-url", "https://pypi.org/simple", "-d", d] + exigences)
# roues déjà servies par python3-modules.json (sources-pypi.py) : pas deux fois
deja = set()
if os.path.exists("python3-modules.json"):
    deja = {m["name"] for m in json.load(open("python3-modules.json", encoding="utf-8"))["modules"]}
mods = []
for w in sorted(glob.glob(os.path.join(d, "*.whl"))):
    nom = os.path.basename(w).split("-")[0]
    if "python3-" + nom.replace("_", "-").lower() in deja or nom.replace("_", "-").lower() in exclues:
        continue
    if not _m.roue_compatible(os.path.basename(w), pyv):     # 266 e : cpXY / abi3 / none, jamais cpXYt
        raise SystemExit(f"{os.path.basename(w)} : roue incompatible avec le Python {pyv} du runtime")
    sha = hashlib.sha256(open(w, "rb").read()).hexdigest()
    # 266 b : l'URL vient de l'index (PEP 503, roue_url.py) — l'ancien appel passait à `pip download` une option
    # qu'il n'a pas (elle n'existe que pour `pip install`) : job flatpak de la v0.7.0 rouge, étape « sources Python »
    url = url_de_la_roue(os.path.basename(w), index, sha)
    mods.append({"name": "python3-" + nom.replace("_", "-").lower(), "buildsystem": "simple",
                 "build-commands": [f"pip3 install --verbose --exists-action=i --no-index --find-links=\"file://${{PWD}}\" --prefix=${{FLATPAK_DEST}} --no-deps \"{nom}\" --no-build-isolation"],
                 "sources": [_m.source_de_roue(os.path.basename(w), url, sha)]})   # 266 g : dest-filename décodé (« + », pas « %2B »)
manquantes = _t.elagage_couvert(metadata, pyv, {m["name"][len("python3-"):] for m in mods} | {n[len("python3-"):] for n in deja})
if manquantes:
    raise SystemExit(f"266 h : dépendances CUDA de torch ni émises ni élaguées : {manquantes}")
json.dump({"name": "torch-cu130", "buildsystem": "simple", "build-commands": [], "modules": mods},
          open("torch-cu130.json", "w"), indent=2)
print(len(mods), "modules →", "torch-cu130.json", "; taille roues", sum(os.path.getsize(w) for w in glob.glob(os.path.join(d, "*.whl"))) // 2**20, "Mio")
PY
