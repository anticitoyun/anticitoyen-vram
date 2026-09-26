"""Pièce 266 : aucun job de release.yml n'écrit dans un dossier qu'il n'a pas créé. Le run 36216576007 de la v0.7.0 a perdu
deux jobs sur des `mkdir` sans `-p` ou absents (`/build/pkg` dans archlinux:latest, `tee build/noyaux/precompiles.json`) :
une erreur de forme, visible à sec, qui a attendu la release pour se montrer. Casse si une écriture (`tee`, `>`, `>>`,
`cp … dossier/`, `--dossier`, `--output`) vise un dossier qu'aucun `mkdir -p` / `mkdir` / `--dir` antérieur du même job
n'a créé, ou si un `mkdir` sans `-p` crée un chemin dont le parent n'est pas créé avant."""
import pathlib
import re

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
RELEASE = RACINE / ".github" / "workflows" / "release.yml"

_CREE = re.compile(r"(?:^|&&|;|\|\|)\s*mkdir\s+(-p\s+)?((?:\S+\s*)+?)(?=\s*(?:&&|;|\|\||$))", re.M)
_DIR_GH = re.compile(r"--dir\s+(\S+)")
_ECRIT = re.compile(r"(?:\btee\s+|(?<![<>])>>?\s*|--output\s+|-o\s+)(?P<cible>[^\s;&|]+)")   # fichiers : leur dossier doit exister
_DOSSIER_ARG = re.compile(r"--dossier\s+(?P<dossier>[^\s;&|]+)")                                  # dossiers : eux-mêmes
_CP = re.compile(r"\bcp\s+(?:-\S+\s+)*(?:\S+\s+)+(?P<dest>\S+/)\s*(?:$|&&|;)", re.M)


def _jobs() -> dict:
    return yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"]


def _run(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in job.get("steps", []))


def _lignes_shell(run: str):
    """Lignes de commande, commentaires retirés, continuations `\\` recollées."""
    run = re.sub(r"\\\n\s*", " ", run)
    for l in run.splitlines():
        l = l.split(" #", 1)[0].strip()
        if l and not l.startswith("#"):
            yield l


def _dossier_de(cible: str) -> str:
    cible = cible.strip('"\'')
    return cible.rsplit("/", 1)[0] if "/" in cible else ""


def _couvert(dossier: str, crees: list) -> bool:
    if dossier in ("", ".", "..", "/dev") or dossier.startswith("/dev/"):
        return True
    if not dossier.startswith(("/", "~", "$")) and (RACINE / dossier).is_dir():
        return True                                    # dossier du dépôt : le checkout l'a créé
    return any(dossier == c or dossier.startswith(c.rstrip("/") + "/") for c in crees)


def ecritures_hors_dossier(run: str) -> list:
    """[(ligne, dossier)] des écritures dont le dossier n'a pas été créé plus haut dans le MÊME job."""
    crees, fautes = [], []
    for l in _lignes_shell(run):
        for m in _CREE.finditer(l):
            for d in m.group(2).split():
                if d.startswith("-"):
                    continue
                d = d.strip('"\'')
                if not m.group(1) and not _couvert(_dossier_de(d), crees):
                    fautes.append((l, f"mkdir sans -p : parent de {d} jamais créé"))
                crees.append(d)
        for m in _DIR_GH.finditer(l):
            crees.append(m.group(1).strip('"\''))
        for m in _ECRIT.finditer(l):
            d = _dossier_de(m.group("cible"))
            if not _couvert(d, crees):
                fautes.append((l, d))
        for m in _DOSSIER_ARG.finditer(l):
            d = m.group("dossier").strip('"\'')
            if not _couvert(d, crees):
                fautes.append((l, d))
        for m in _CP.finditer(l):
            d = m.group("dest").rstrip("/")
            if not _couvert(d, crees):
                fautes.append((l, d))
    return fautes


def test_aucun_job_n_ecrit_dans_un_dossier_jamais_cree():
    fautes = {nom: ecritures_hors_dossier(_run(j)) for nom, j in _jobs().items()}
    fautes = {k: v for k, v in fautes.items() if v}
    assert not fautes, fautes


def test_le_test_sait_dire_faux():
    """Les deux fautes de la v0.7.0, telles quelles, doivent être vues ; leurs corrections, acceptées."""
    assert ecritures_hors_dossier("mkdir /build/pkg\ncp a /build/pkg/")
    assert not ecritures_hors_dossier("mkdir -p /build/pkg\ncp a /build/pkg/")
    assert ecritures_hors_dossier("python -m x --dossier build/noyaux | tee build/noyaux/p.json")
    assert not ecritures_hors_dossier("mkdir -p build/noyaux\npython -m x --dossier build/noyaux | tee build/noyaux/p.json")
    assert ecritures_hors_dossier("sed s/a/b/ f > sortie/f")
    assert not ecritures_hors_dossier("mkdir joints && gh release download v1 --dir joints\ncd joints && sha256sum -- * > ../SHA256SUMS")
