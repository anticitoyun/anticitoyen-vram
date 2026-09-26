#!/usr/bin/env bash
# Pièce 236 : produit packaging/flathub/torch-cu130.json — les roues PyTorch cu130 (torch, triton) et les roues nvidia-* qu'elles
# tirent, en sources `file` (URL + sha256) pour une construction Flathub SANS réseau. À jouer HORS bac à sable, avec réseau
# (CI ou poste) ; le résultat se versionne. Roues cp312 manylinux x86_64 (le SDK GNOME 51 porte Python 3.12).
#   ./sources-torch.sh 2.14.0            # → torch-cu130.json
set -euo pipefail
V="${1:?version de torch, ex. 2.14.0}"
INDEX="https://download.pytorch.org/whl/cu130"
D=$(mktemp -d)
python3 -m pip download --no-deps --only-binary=:all: --python-version 3.12 --platform manylinux_2_28_x86_64 \
    --index-url "$INDEX" -d "$D" "torch==$V" triton
# les roues nvidia-* exigées par cette roue torch (Requires-Dist), résolues sur le même index puis PyPI
python3 - "$D" "$INDEX" <<'PY'
import glob, json, os, re, subprocess, sys, zipfile, hashlib
d, index = sys.argv[1], sys.argv[2]
torch = glob.glob(os.path.join(d, "torch-*.whl"))[0]
with zipfile.ZipFile(torch) as z:
    meta = next(n for n in z.namelist() if n.endswith("METADATA"))
    reqs = [l.split(":", 1)[1].strip() for l in z.read(meta).decode().splitlines() if l.startswith("Requires-Dist: nvidia")]
noms = sorted({re.split(r"[ ;=<>!]", r)[0] for r in reqs} - {"nvidia-cudnn-cu13", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13", "nvidia-cusparselt-cu13"})
for n in noms:  # cudnn, nccl, nvshmem, cusparselt : retirés (cleanup du manifeste) — inutilisés par acvram
    subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", "3.12",
                           "--platform", "manylinux_2_28_x86_64", "--index-url", index, "--extra-index-url", "https://pypi.org/simple", "-d", d, n])
mods = []
for w in sorted(glob.glob(os.path.join(d, "*.whl"))):
    nom = os.path.basename(w).split("-")[0]
    sha = hashlib.sha256(open(w, "rb").read()).hexdigest()
    url = subprocess.check_output([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", "3.12",
                                   "--platform", "manylinux_2_28_x86_64", "--index-url", index, "--extra-index-url", "https://pypi.org/simple",
                                   "--dry-run", "--report", "-", nom + ("==" + os.path.basename(w).split("-")[1])], text=True)
    url = json.loads(url)["install"][0]["download_info"]["url"]
    mods.append({"name": "python3-" + nom, "buildsystem": "simple",
                 "build-commands": [f"pip3 install --verbose --exists-action=i --no-index --find-links=\"file://${{PWD}}\" --prefix=${{FLATPAK_DEST}} --no-deps \"{nom}\" --no-build-isolation"],
                 "sources": [{"type": "file", "url": url, "sha256": sha}]})
json.dump({"name": "torch-cu130", "buildsystem": "simple", "build-commands": [], "modules": mods},
          open("torch-cu130.json", "w"), indent=2)
print(len(mods), "modules →", "torch-cu130.json", "; taille roues", sum(os.path.getsize(w) for w in glob.glob(os.path.join(d, "*.whl"))) // 2**20, "Mio")
PY
