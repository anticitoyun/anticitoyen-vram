"""Pièce 266 b : l'URL d'une roue déjà téléchargée, lue dans l'index simple (PEP 503) — jamais par `pip download --dry-run`,
option qui n'existe pas (`pip download` ne connaît ni --dry-run ni --report : le job flatpak de la v0.7.0 est tombé dessus).
L'index PyTorch (`https://download.pytorch.org/whl/cu130/<nom>/`) liste des hrefs relatifs `/whl/cu130/<fichier>#sha256=…`,
PyPI (`https://pypi.org/simple/<nom>/`) des hrefs absolus vers files.pythonhosted.org : `urljoin` couvre les deux ; le
fragment `#sha256=` est retiré de l'URL et, quand il existe, comparé à la somme de la roue téléchargée."""
import html
import re
import urllib.parse
import urllib.request

_HREF = re.compile(r'href="([^"]+)"')


def url_depuis_index(page: str, base: str, fichier: str, sha256: str | None = None) -> str:
    """URL absolue de `fichier` (nom de roue exact) dans la page d'index `page` servie à `base`."""
    for href in _HREF.findall(page):
        href = html.unescape(href)
        sans_fragment, _, fragment = href.partition("#")
        if urllib.parse.unquote(sans_fragment.rsplit("/", 1)[-1]) != fichier:
            continue
        if sha256 and fragment.startswith("sha256=") and fragment[7:] != sha256:
            raise ValueError(f"{fichier} : sha256 de l'index {fragment[7:16]}… ≠ roue téléchargée {sha256[:9]}…")
        return urllib.parse.urljoin(base, sans_fragment)
    raise LookupError(f"{fichier} absent de l'index {base}")


def url_de_la_roue(fichier: str, index: str, sha256: str | None = None, pypi: str = "https://pypi.org/simple") -> str:
    """Cherche `fichier` sur `index` (PyTorch) puis sur PyPI — même ordre que le `--index-url … --extra-index-url` du téléchargement."""
    nom = fichier.split("-")[0].replace("_", "-").lower()
    erreurs = []
    for base in (f"{index.rstrip('/')}/{nom}/", f"{pypi.rstrip('/')}/{nom}/"):
        try:
            # UA explicite : download-r2.pytorch.org (CDN des roues) refuse « Python-urllib » (403), pas les autres
            with urllib.request.urlopen(urllib.request.Request(base, headers={"User-Agent": "acvram-sources-torch/266b"}), timeout=60) as r:
                return url_depuis_index(r.read().decode("utf-8", "replace"), base, fichier, sha256)
        except (LookupError, OSError, ValueError) as exc:
            # ValueError = même nom de roue mais autre sha256 sur cet index (triton 3.8.0 existe sur l'index PyTorch ET sur PyPI,
            # deux fichiers différents ; pip a pris l'un des deux) : on cherche l'index qui porte CE fichier, on ne renonce pas
            erreurs.append(f"{base} : {exc}")
    raise LookupError(" ; ".join(erreurs))
