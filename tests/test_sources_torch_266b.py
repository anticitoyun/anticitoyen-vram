"""Pièce 266 b : `packaging/flathub/sources-torch.sh` ne passe jamais `--dry-run` (ni `--report`) à `pip download` — options
de `pip install` seulement ; le job flatpak de la v0.7.0 est tombé dessus (« no such option: --dry-run »). L'URL d'une roue
se lit dans l'index simple (roue_url.py), testé ici sans réseau sur une page PyTorch et une page PyPI."""
import hashlib
import pathlib
import re
import sys

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "packaging" / "flathub" / "sources-torch.sh"
sys.path.insert(0, str(RACINE / "packaging" / "flathub"))
from roue_url import url_depuis_index  # noqa: E402


def commandes_pip_download(texte: str) -> list[str]:
    """Chaque invocation `pip download` du script — ligne shell (continuations recollées) ou liste Python d'arguments
    (jusqu'à la parenthèse fermante de l'appel) — sans les commentaires."""
    texte = re.sub(r"(?m)^\s*#.*$", "", texte)
    texte = re.sub(r"\\\n\s*", " ", texte)
    appels = re.findall(r"(?m)^[^\n]*python3? -m pip download[^\n]*$", texte)
    appels += re.findall(r'"pip",\s*"download".*?\]', texte, flags=re.S)
    return appels


def test_pip_download_ne_recoit_ni_dry_run_ni_report():
    appels = commandes_pip_download(SCRIPT.read_text(encoding="utf-8"))
    assert len(appels) >= 2, "gabarit : le script télécharge torch/triton puis les roues nvidia-* par pip download"
    fautifs = [a for a in appels if "--dry-run" in a or "--report" in a]
    assert not fautifs, f"`pip download` ne connaît ni --dry-run ni --report : {fautifs}"


def test_le_test_sait_dire_faux():
    ancien = ('    url = subprocess.check_output([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",\n'
              '                                   "--dry-run", "--report", "-", nom], text=True)\n')
    assert any("--dry-run" in a for a in commandes_pip_download(ancien)), "l'appel fautif de la v0.7.0 doit être vu"


ROUE = "torch-2.14.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
SHA = hashlib.sha256(b"roue simulee").hexdigest()
PAGE_PYTORCH = f'<html><body>\n<a href="/whl/cu130/torch-2.14.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256={SHA}">' \
               f'torch-2.14.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl</a><br/>\n<a href="/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl">x</a>\n</body></html>'
PAGE_PYPI = f'<a href="https://files.pythonhosted.org/packages/ab/cd/nvidia_cublas-13.0.0-py3-none-manylinux_2_27_x86_64.whl#sha256={SHA}">nvidia_cublas-13.0.0-py3-none-manylinux_2_27_x86_64.whl</a>'


def test_url_depuis_l_index_pytorch_relatif_et_pypi_absolu():
    assert url_depuis_index(PAGE_PYTORCH, "https://download.pytorch.org/whl/cu130/torch/", ROUE, SHA) == \
        "https://download.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert url_depuis_index(PAGE_PYPI, "https://pypi.org/simple/nvidia-cublas/", "nvidia_cublas-13.0.0-py3-none-manylinux_2_27_x86_64.whl", SHA) \
        .startswith("https://files.pythonhosted.org/packages/ab/cd/nvidia_cublas-13.0.0")


def test_une_roue_absente_ou_une_somme_differente_sont_refusees():
    with pytest.raises(LookupError):
        url_depuis_index(PAGE_PYTORCH, "https://download.pytorch.org/whl/cu130/torch/", "torch-9.9.9-cp312-cp312-manylinux_2_28_x86_64.whl")
    with pytest.raises(ValueError):
        url_depuis_index(PAGE_PYTORCH, "https://download.pytorch.org/whl/cu130/torch/", ROUE, "0" * 64)


def test_meme_nom_autre_somme_on_passe_a_l_index_suivant(monkeypatch):
    """266 i : triton-3.8.0 existe sur l'index PyTorch ET sur PyPI avec deux sha256 ; l'URL retenue est celle dont la somme
    est celle de la roue téléchargée, pas la première venue (témoin : la première seule → ValueError)."""
    import roue_url
    sha_pypi = hashlib.sha256(b"pypi").hexdigest(); sha_torch = hashlib.sha256(b"torch").hexdigest()
    roue = "triton-3.8.0-cp314-cp314-manylinux_2_28_x86_64.whl"
    pages = {"https://download.pytorch.org/whl/cu130/triton/": f'<a href="/whl/cu130/{roue}#sha256={sha_torch}">{roue}</a>',
             "https://pypi.org/simple/triton/": f'<a href="https://files.pythonhosted.org/p/{roue}#sha256={sha_pypi}">{roue}</a>'}

    class _R:
        def __init__(self, h): self.h = h.encode()
        def read(self): return self.h
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(roue_url.urllib.request, "urlopen", lambda req, timeout=60: _R(pages[req.full_url]))
    assert roue_url.url_de_la_roue(roue, "https://download.pytorch.org/whl/cu130", sha_pypi).startswith("https://files.pythonhosted.org/")
    assert roue_url.url_de_la_roue(roue, "https://download.pytorch.org/whl/cu130", sha_torch).startswith("https://download.pytorch.org/")
    with pytest.raises(LookupError):
        roue_url.url_de_la_roue(roue, "https://download.pytorch.org/whl/cu130", "0" * 64)
