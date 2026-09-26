"""Pièce 266 h : sources-torch.sh émet TOUTES les dépendances de la roue torch cu130 lues dans Requires-Dist — depuis cu130
les bibliothèques CUDA passent par `cuda-toolkit[cublas,…]` et `cuda-bindings`, plus par des lignes `nvidia-*` (sauf les
quatre élaguées) ; le filtre « Requires-Dist: nvidia » de la 236 n'émettait plus rien et le bundle v0.7.0 n'avait pas de
libcublasLt. Gardes sur la métadonnée RÉELLE de torch-2.14.0+cu130-cp314 : cuda-toolkit avec ses extras et cuda-bindings
gardés, marqueurs évalués (Linux / 3.14), extras de torch ignorés, chaque nvidia-* soit émis soit élagué avec sa raison ;
témoin : l'ancien filtre ne voit que les quatre élaguées."""
import importlib.util
import pathlib

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
FLATHUB = RACINE / "packaging" / "flathub"
_spec = importlib.util.spec_from_file_location("sources_torch", FLATHUB / "sources_torch.py")
ST = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ST)

# Requires-Dist de torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl (METADATA, lue le 26/09/2026)
METADATA = """Metadata-Version: 2.4
Name: torch
Version: 2.14.0+cu130
Requires-Dist: filelock
Requires-Dist: typing-extensions>=4.10.0
Requires-Dist: setuptools>=77.0.3
Requires-Dist: sympy>=1.13.3
Requires-Dist: networkx>=2.5.1
Requires-Dist: jinja2
Requires-Dist: fsspec>=0.8.5
Requires-Dist: cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2
Requires-Dist: cuda-bindings<14,>=13.0.3; platform_system == "Linux" and python_version < "3.15"
Requires-Dist: nvidia-cudnn-cu13==9.24.0.43; platform_system == "Linux"
Requires-Dist: nvidia-cusparselt-cu13==0.8.1; platform_system == "Linux"
Requires-Dist: nvidia-nccl-cu13==2.30.7; platform_system == "Linux"
Requires-Dist: nvidia-nvshmem-cu13==3.4.5; platform_system == "Linux"
Requires-Dist: triton~=3.8.0; platform_system == "Linux" and python_version < "3.15"
Requires-Dist: optree>=0.13.0; extra == "optree"
Requires-Dist: opt-einsum>=3.3; extra == "opt-einsum"
Requires-Dist: pyyaml; extra == "pyyaml"
"""


def test_les_bibliotheques_cuda_passent_par_cuda_toolkit_et_cuda_bindings():
    gardees, exclues = ST.exigences_de_torch(METADATA, "3.14")
    assert "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2" in gardees
    assert any(g.startswith("cuda-bindings") for g in gardees) and any(g.startswith("triton") for g in gardees)
    for nom in ("filelock", "typing-extensions", "setuptools", "sympy", "networkx", "jinja2", "fsspec"):
        assert any(g.startswith(nom) for g in gardees), nom
    assert not any(g.startswith(("optree", "opt-einsum", "pyyaml")) for g in gardees), "les extras de torch ne sont pas des dépendances"


def test_aucun_elagage_les_quatre_de_la_236b_sont_liees_par_libtorch():
    """ldd libtorch_cuda.so (2.14.0+cu130) : libcudnn.so.9, libnccl.so.2, libcusparseLt.so.0, libnvshmem_host.so.3 — chacune
    manquante fait échouer `import torch` (prouvé une à une, 266 h). Elles sont donc ÉMISES, plus élaguées."""
    gardees, exclues = ST.exigences_de_torch(METADATA, "3.14")
    assert exclues == {} and ST.ELAGAGE == {}
    for nom in ("nvidia-cudnn-cu13", "nvidia-cusparselt-cu13", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13"):
        assert any(g.startswith(nom) for g in gardees), nom


def test_les_marqueurs_sont_evalues_pour_le_runtime():
    gardees, _ = ST.exigences_de_torch(METADATA, "3.15")
    assert not any(g.startswith(("cuda-bindings", "triton")) for g in gardees), "python_version < 3.15 : absentes en 3.15"
    gardees, _ = ST.exigences_de_torch(METADATA, "3.14")
    assert any(g.startswith("cuda-bindings") for g in gardees)


def test_la_garde_dit_ce_qui_manque():
    tout = {"cuda-toolkit", "cuda-bindings", "nvidia-cudnn-cu13", "nvidia-cusparselt-cu13", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13"}
    assert ST.elagage_couvert(METADATA, "3.14", emis=tout) == []
    assert ST.elagage_couvert(METADATA, "3.14", emis={"triton"}) == sorted(tout, key=lambda n: (n.startswith("nvidia"), n)) or \
        set(ST.elagage_couvert(METADATA, "3.14", emis={"triton"})) == tout


def test_le_temoin_l_ancien_filtre_ne_voyait_que_les_quatre_qu_il_elaguait():
    """Ce que faisait la 236 : ne garder que `Requires-Dist: nvidia…` puis retirer cudnn/nccl/nvshmem/cusparselt — il ne restait
    RIEN : ni cublas ni cudart (passés par cuda-toolkit), ni les quatre. Le bundle v0.7.0 n'avait aucune bibliothèque CUDA."""
    anciens = [l for l in METADATA.splitlines() if l.startswith("Requires-Dist: nvidia")]
    noms = {l.split(":", 1)[1].strip().split("=")[0] for l in anciens}
    assert noms == {"nvidia-cudnn-cu13", "nvidia-cusparselt-cu13", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13"}
    assert not any(n.startswith("nvidia-cublas") for n in noms)


def test_sources_torch_sh_utilise_le_module():
    t = (FLATHUB / "sources-torch.sh").read_text(encoding="utf-8")
    assert "exigences_de_torch(metadata, pyv)" in t and "elagage_couvert(" in t and 'startswith("Requires-Dist: nvidia")' not in t
    assert "--no-deps" not in t.split("exigences_de_torch")[1].split("mods = []")[0], "la fermeture se télécharge AVEC dépendances"


def test_toutes_les_glibc_manylinux_sont_demandees():
    """pip n'accepte que les étiquettes données : nvidia_cuda_cupti est en manylinux_2_25, cublas en 2_27, torch en 2_28."""
    _sp = importlib.util.spec_from_file_location("sources_pypi", FLATHUB / "sources-pypi.py")
    SP = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(SP)
    assert {f"manylinux_2_{k}_x86_64" for k in (5, 12, 17, 24, 25, 27, 28)} <= set(SP.PLATEFORMES)
    assert "manylinux2014_x86_64" in SP.PLATEFORMES
