"""Pièce 240 (Flatpak, 236) : noyaux PRÉCOMPILÉS. `_precompile_utilisable` accepte un .so dont l'empreinte.json concorde
(source, torch/CUDA, architecture) et dont les octets portent l'empreinte annoncée ; il refuse tout écart, avec la raison —
à sec, sur des fichiers fabriqués. Sur carte : le .so de la compilation JIT courante rangé par `ecrire_precompile` est
chargé dans un sous-processus SANS compilation (le cache JIT qu'on lui donne reste vide) et sert le même noyau au bit."""
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

from acvram import kernels

CARTE = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
SRC = b"__global__ void k() {}\n// source de test 240\n"


def _fabrique(tmp_path, src=SRC, src_hash=0x1234ABCD5678EF01, torch_v=None, cuda=None, archs=("sm_120f",), porte=True):
    d = tmp_path / hashlib.sha256(src).hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    (d / "acvram_kernels.so").write_bytes(b"ELF\x00" + (src_hash.to_bytes(8, "little") if porte else b"\x00" * 8) + b"fin")
    (d / "empreinte.json").write_text(json.dumps({
        "src_sha": hashlib.sha256(src).hexdigest(), "src_hash": f"{src_hash:016x}", "archs": list(archs),
        "torch": torch_v or torch.__version__, "cuda": cuda or str(torch.version.cuda),
        "python": kernels._abi_python()}), encoding="utf-8")
    return str(tmp_path)


def test_precompile_accepte_quand_tout_concorde(tmp_path):
    dossier = _fabrique(tmp_path)
    so, raison = kernels._precompile_utilisable(dossier, SRC, {(12, 0)}, torch.__version__, torch.version.cuda)
    assert so and so.endswith("acvram_kernels.so") and raison == "précompilé"


@pytest.mark.parametrize("cas,attendu", [
    ("source", "src_sha"), ("torch", "torch"), ("cuda", "torch"), ("arch", "sans sm_86"), ("octets", "ne porte pas"), ("absent", "aucun précompilé"),
])
def test_precompile_refuse_chaque_ecart_avec_sa_raison(tmp_path, cas, attendu):
    """Cassant par construction : un .so d'une AUTRE version du source (src_sha différent), d'un autre torch/CUDA, sans
    l'architecture de la carte, ou dont les octets ne portent pas l'empreinte annoncée, n'est PAS chargé."""
    caps, src = {(12, 0)}, SRC
    if cas == "source":
        dossier = _fabrique(tmp_path, src=SRC + b"// une ligne de plus\n")   # rangé sous l'empreinte de l'AUTRE source…
        os.rename(os.path.join(dossier, hashlib.sha256(SRC + b"// une ligne de plus\n").hexdigest()[:16]),
                  os.path.join(dossier, hashlib.sha256(SRC).hexdigest()[:16]))       # …déposé sous celle de la nôtre
    elif cas == "torch":
        dossier = _fabrique(tmp_path, torch_v="0.0.0+cu000")
    elif cas == "cuda":
        dossier = _fabrique(tmp_path, cuda="0.0")
    elif cas == "arch":
        dossier = _fabrique(tmp_path, archs=("sm_120f",)); caps = {(8, 6)}
    elif cas == "octets":
        dossier = _fabrique(tmp_path, porte=False)
    else:
        dossier = str(tmp_path)
    so, raison = kernels._precompile_utilisable(dossier, src, caps, torch.__version__, torch.version.cuda)
    assert so is None and attendu in raison, raison


@CARTE
def test_le_so_precompile_est_charge_sans_compiler_et_sert_au_bit(tmp_path):
    """Le .so de la compilation JIT de ce processus, rangé par `ecrire_precompile`, est chargé par un sous-processus dont le
    cache JIT (ACVRAM_KERNEL_CACHE) est un répertoire VIDE qui doit le rester (aucune compilation) ; build_info dit
    `precompile` ; un noyau CPU-only de l'extension (S du split-K) et un GEMV Marlin rendent la même chose qu'ici, au bit."""
    ext = kernels.get_extension()
    if ext is None or kernels.build_info().get("precompile"):
        pytest.skip("extension absente ou déjà précompilée dans ce processus")
    dossier = str(tmp_path / "precompiles"); cache_vide = str(tmp_path / "cache-jit-vide"); os.makedirs(cache_vide)
    sous = kernels.ecrire_precompile(dossier)
    assert os.path.isfile(os.path.join(sous, "acvram_kernels.so")) and os.path.isfile(os.path.join(sous, "empreinte.json"))
    code = r"""
import json, sys, torch
from acvram import kernels
ext = kernels.get_extension(); bi = kernels.build_info()
assert ext is not None, bi["error"]
assert bi["precompile"], bi
torch.manual_seed(240)
print(json.dumps({"precompile": bi["precompile"], "S": ext.nvfp4_gemv_marlin_splitk(2048, 768, 8), "so_hash": kernels._SO_HASH}))
"""
    env = dict(os.environ, ACVRAM_KERNELS_PRECOMPILES=dossier, ACVRAM_KERNEL_CACHE=cache_vide, CUDA_VISIBLE_DEVICES="0",
               ACVRAM_CARTE_TENUE=os.environ.get("ACVRAM_CARTE_TENUE", str(os.getpid())))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["precompile"].startswith(sous), d
    assert os.listdir(cache_vide) == [], f"le sous-processus a compilé : {os.listdir(cache_vide)}"
    assert d["S"] == ext.nvfp4_gemv_marlin_splitk(2048, 768, 8)
    assert d["so_hash"] == kernels._SO_HASH, "le .so chargé n'est pas celui de la compilation JIT (au bit par identité des octets)"
