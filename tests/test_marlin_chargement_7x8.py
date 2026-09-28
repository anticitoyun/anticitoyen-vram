"""bd 7x8 : OOM au CHARGEMENT de Qwen3-Coder-30B (qkvo-i8c) servi depuis l'arbre, pas depuis le paquet.

Deux fautes, deux tests chacune à sec :
* `facteur_nvfp4` (marlin_port/__init__.py) prenait le max par `ws[nz]` — un nonzero de 8 o × ndim par élément
  (24 × 128·768·128 = 288 Mio pour une pile Coder) alloué pendant la construction Marlin, carte déjà remplie par le
  KV. Même facteur au bit, sans temporaire plus gros que la copie flottante.
* `pyproject.toml:package-data` n'avait pas `*.hpp` : le venv du paquet n'a jamais `libtorch_stable/core/math.hpp`,
  Marlin n'y compile pas (repli nommé sur la ligne de régime seulement) — le paquet servait sans Marlin, d'où
  « le paquet sert, l'arbre meurt »."""
import fnmatch
import pathlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from acvram.kernels.marlin_port import facteur_nvfp4

RACINE = pathlib.Path(__file__).resolve().parent.parent


def _facteur_reference(marlin_scales: torch.Tensor) -> float:
    """L'écriture d'avant 7x8, gardée comme témoin d'équivalence."""
    ws = marlin_scales.float() * (2 ** 7)
    nz = ws > 0
    if nz.any():
        mx = ws[nz].max()
        if mx < 448 * (2 ** 7):
            return (448 * (2 ** 7) / mx).log2().floor().exp2().item()
    return 1.0


def _cas():
    g = torch.Generator().manual_seed(7)
    base = torch.rand(8, 64, 32, generator=g)
    yield "aléatoire < 1", base.to(torch.bfloat16)
    yield "avec zéros", (base * (base > 0.3)).to(torch.bfloat16)
    yield "avec négatifs", (base - 0.5).to(torch.bfloat16)
    yield "max 448", (base * 448).clamp(max=448).to(torch.bfloat16)
    yield "sous-normales", (base * 2 ** -9).to(torch.bfloat16)
    yield "tout nul", torch.zeros(4, 64, 16, dtype=torch.bfloat16)
    nan = base.clone()
    nan[0, 0, 0] = float("nan")
    yield "avec NaN", nan.to(torch.bfloat16)
    yield "2D (dense)", base[0].to(torch.bfloat16)


@pytest.mark.parametrize("nom,scales", list(_cas()), ids=[n for n, _ in _cas()])
def test_facteur_nvfp4_egal_a_l_ecriture_d_avant(nom, scales):
    assert facteur_nvfp4(scales) == _facteur_reference(scales), nom


class _Operations(TorchDispatchMode):
    """Relève les opérations qui, SUR CARTE, matérialisent des indices : nonzero, et un index par masque booléen
    (aten.index.Tensor le résout par nonzero sur CUDA — 8 o × ndim par élément). Sur processeur l'index booléen ne
    passe pas par un nonzero visible : on juge donc l'opération, pas les octets."""
    def __init__(self):
        super().__init__()
        self.fautives = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        nom = str(func)
        if "nonzero" in nom or "masked_select" in nom:
            self.fautives.append(nom)
        elif nom.startswith("aten.index.") or nom.startswith("aten.index_put"):
            idx = args[1] if len(args) > 1 else ()
            if any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in (idx or ())):
                self.fautives.append(nom + "(masque)")
        return func(*args, **(kwargs or {}))


def test_facteur_nvfp4_sans_index_par_masque():
    # Casse sur l'écriture d'avant (`ws[nz]`) : 288 Mio d'indices int64 sur la pile Coder-30B au chargement.
    scales = (torch.rand(16, 96, 48) + 0.01).to(torch.bfloat16)
    with _Operations() as m:
        facteur_nvfp4(scales)
    assert not m.fautives, f"indices matérialisés : {m.fautives}"


def test_package_data_couvre_toutes_les_sources_des_noyaux():
    # Casse si l'on retire "kernels/**/*.hpp" : math.hpp et scalar_type.hpp manquent au venv du paquet.
    try:
        import tomllib
    except ImportError:                                  # Python < 3.11
        pytest.skip("tomllib absent")
    motifs = tomllib.loads((RACINE / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["acvram"]
    hors_paquet = {"LICENSE-vllm", "NOTICE"}           # licences : livrées sous /usr/share/acvram par le .deb
    manquants = []
    for f in (RACINE / "acvram" / "kernels").rglob("*"):
        if not f.is_file() or f.suffix in (".py", ".pyc") or f.name in hors_paquet or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(RACINE / "acvram").as_posix()
        if not any(fnmatch.fnmatch(rel, m) or fnmatch.fnmatch(rel, m.replace("**/", "")) for m in motifs):
            manquants.append(rel)
    assert not manquants, f"absents de package-data (donc du venv du paquet) : {manquants}"
