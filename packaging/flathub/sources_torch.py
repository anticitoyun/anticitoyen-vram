"""Pièce 266 h : les dépendances de la roue torch cu130, lues dans ses métadonnées (Requires-Dist), TOUTES — pas seulement
les lignes « nvidia-* ». Depuis cu130, torch déclare ses bibliothèques CUDA par le métapaquet
`cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.x` et `cuda-bindings` ;
les seules lignes `nvidia-*` restantes sont cudnn, cusparselt, nccl, nvshmem — exactement l'élagage de la 236 b. Le filtre
« Requires-Dist: nvidia » de la 236 ne voyait donc plus rien : le bundle v0.7.0 n'avait aucune bibliothèque CUDA
(« libcublasLt.so not found », verif-070). Pur, testé à sec sur la métadonnée réelle (tests/test_flathub_torch_deps_266h.py)."""
import urllib.parse

from packaging.markers import Marker
from packaging.requirements import Requirement

# 266 i : `apply_extra`, joué par flatpak à l'installation dans /app/extra (cwd) — dépaquette chaque roue (zip) dans
# site-packages avec le python3 du runtime (pas de pip dans org.gnome.Platform), puis retire la roue ; `apply_extra.ok`
# liste ce qui a été posé (lu par verifier-release.sh). Aucun réseau ici : flatpak a déjà téléchargé et vérifié les fichiers.
APPLY_EXTRA = [
    "set -e",
    "mkdir -p site-packages",
    "for w in *.whl; do python3 -m zipfile -e \"$w\" site-packages && rm -f \"$w\"; done",
    "ls site-packages > apply_extra.ok",
]

# Élagage : VIDE depuis la 266 h. La 236 b retirait cudnn, cusparselt, nccl et nvshmem (« inutilisés par acvram ») — mais
# libtorch_cuda.so de torch 2.14.0+cu130 les LIE (ldd : libcudnn.so.9, libnccl.so.2, libcusparseLt.so.0, libnvshmem_host.so.3)
# et `import torch` échoue dès qu'une manque, prouvé une à une dans un venv 3.14 (verif-070 / 266 h) : +527 (cudnn), +205 (nccl),
# +162 (cusparselt), +57 Mio (nvshmem). Un nom ne revient ici qu'avec la preuve qu'un `import torch` passe sans lui.
ELAGAGE: dict[str, str] = {}
ENV_RUNTIME = {"platform_system": "Linux", "sys_platform": "linux", "os_name": "posix", "platform_machine": "x86_64",
               "implementation_name": "cpython", "platform_python_implementation": "CPython"}


def source_extra_data(fichier: str, url: str, sha256: str, taille: int) -> dict:
    """Source `extra-data` d'une roue : nom décodé (266 g), URL, sha256 et taille — flatpak refuse le fichier si l'un diffère."""
    fichier = urllib.parse.unquote(fichier)
    if "%" in fichier or not fichier.endswith(".whl") or len(fichier[:-4].split("-")) < 5 or taille <= 0:
        raise ValueError(f"{fichier} : nom de roue ou taille invalide pour une extra-data")
    return {"type": "extra-data", "filename": fichier, "url": url, "sha256": sha256, "size": int(taille), "only-arches": ["x86_64"]}


def requires_dist(metadata: str) -> list[str]:
    return [l.split(":", 1)[1].strip() for l in metadata.splitlines() if l.startswith("Requires-Dist:")]


def exigences_de_torch(metadata: str, python_version: str) -> tuple[list[str], dict[str, str]]:
    """(exigences à télécharger, exclusions {nom: raison}) — marqueurs évalués pour le runtime Linux x86_64 / CPython
    `python_version`, exigences d'extras (`extra == …`) ignorées, élagage appliqué. Une exigence gardée est rendue telle
    quelle, extras et spécificateur compris (« cuda-toolkit[cublas,…]==13.0.2 »)."""
    env = dict(ENV_RUNTIME, python_version=python_version, python_full_version=python_version + ".0")
    gardees, exclues = [], {}
    for texte in requires_dist(metadata):
        r = Requirement(texte)
        if r.marker is not None:
            if "extra" in str(r.marker):
                continue
            if not Marker(str(r.marker)).evaluate(env):
                continue
        nom = r.name.lower()
        if nom in ELAGAGE:
            exclues[nom] = ELAGAGE[nom]
            continue
        extras = "[" + ",".join(sorted(r.extras)) + "]" if r.extras else ""
        gardees.append(f"{r.name}{extras}{r.specifier}")
    return gardees, exclues


def elagage_couvert(metadata: str, python_version: str, emis: set[str]) -> list[str]:
    """Garde 266 h : chaque dépendance CUDA de torch (nvidia-*, cuda-*) est soit ÉMISE (roue dans le json), soit dans
    l'élagage avec sa raison. Rend la liste des manquantes (vide = tenu)."""
    gardees, exclues = exigences_de_torch(metadata, python_version)
    manquantes = []
    for texte in gardees:
        nom = Requirement(texte).name.lower().replace("_", "-")
        if (nom.startswith("nvidia-") or nom.startswith("cuda-")) and nom not in emis:
            manquantes.append(nom)
    return manquantes
