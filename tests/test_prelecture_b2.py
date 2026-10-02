"""B2 (poste6, 02/10, scellé revue/poste6-prelecture-scelle-02-10.md) : prélecture des fragments avant le chargement — elle
remplit le cache de pages (readahead par pas) et ne touche pas au chemin des tenseurs.
(1) identité : tous les tenseurs du jouet sont égaux À L'OCTET avec et sans prélecture, et les fragments gardent leur sha256 ;
(2) elle couvre chaque fragment en entier, une fois (cassant : une prélecture qui ne lit rien, ou qui saute des plages) ;
(3) coupée par `ACVRAM_PRELECTURE=0`, sans fichier, ou quand les fragments dépassent la RAM disponible (dit) ;
(4) une erreur de lecture arrête la prélecture, jamais le chargement ; (5) `load_model` la lance et le dit."""
import hashlib
import json
import os

import torch

from acvram.engine import loader as LD


def _fragments(dossier):
    with open(os.path.join(dossier, "acvram_manifest.json"), encoding="utf-8") as fh:
        carte = json.load(fh)["weight_map"]
    return carte, sorted({os.path.join(dossier, f) for f in carte.values()})


def _empreintes(dossier, carte):
    lecteur = LD._ShardReader(dossier, carte)
    try:
        return {k: hashlib.sha256(lecteur.get(k).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for k in carte}
    finally:
        lecteur.close()


def _sha(f):
    with open(f, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_identite_a_l_octet_avec_et_sans_prelecture(converted):
    carte, fichiers = _fragments(converted)
    avant = {f: _sha(f) for f in fichiers}
    sans = _empreintes(converted, carte)
    bilan = LD._prelire(fichiers, fils=2)
    assert bilan is not None and bilan["fini"].wait(30) and "erreur" not in bilan, bilan
    avec = _empreintes(converted, carte)
    assert len(sans) > 0 and avec == sans, "un tenseur diffère avec la prélecture"
    assert {f: _sha(f) for f in fichiers} == avant, "la prélecture a modifié un fragment"


def test_chaque_fragment_est_couvert_en_entier(converted, monkeypatch):
    carte, fichiers = _fragments(converted)
    vus = {}

    def espion(fd, debut, octets):
        vus.setdefault(os.fstat(fd).st_ino, []).append((debut, octets))
    monkeypatch.setattr(LD, "_readahead", espion)
    bilan = LD._prelire(fichiers, fils=3)
    assert bilan["fini"].wait(30) and bilan["octets"] == bilan["total"] == sum(os.path.getsize(f) for f in fichiers)
    for f in fichiers:
        plages = sorted(vus[os.stat(f).st_ino])
        assert [d for d, _ in plages] == list(range(0, os.path.getsize(f), LD._PRELECTURE_PAS)), f"{f} : plages sautées ou relues"
        assert all(n == LD._PRELECTURE_PAS for _, n in plages)


def test_coupee(converted, monkeypatch, capsys):
    carte, fichiers = _fragments(converted)
    assert LD._prelire(fichiers, fils=0) is None
    assert LD._prelire([os.path.join(converted, "absent.safetensors")], fils=2) is None
    monkeypatch.setattr(LD, "_PRELECTURE", 0)
    assert LD._prelire(fichiers) is None
    monkeypatch.setattr(os.path, "getsize", lambda f: 1 << 50)               # plus que toute RAM disponible
    assert LD._prelire(fichiers, fils=2) is None
    assert "prélecture coupée" in capsys.readouterr().out


def test_une_erreur_n_arrete_que_la_prelecture(converted, monkeypatch):
    carte, fichiers = _fragments(converted)
    reference = _empreintes(converted, carte)

    def panne(fd, debut, octets):
        raise OSError("disque retiré")
    monkeypatch.setattr(LD, "_readahead", panne)
    bilan = LD._prelire(fichiers, fils=2)
    assert bilan["fini"].wait(30) and "disque retiré" in bilan["erreur"]
    assert _empreintes(converted, carte) == reference


def test_load_model_la_lance_et_le_dit(converted, monkeypatch, capsys):
    from acvram.engine.loader import load_model
    monkeypatch.setattr(LD, "_PRELECTURE", 1)
    load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
    assert "[acvram] prélecture : " in capsys.readouterr().out
    monkeypatch.setattr(LD, "_PRELECTURE", 0)
    load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
    assert "prélecture" not in capsys.readouterr().out
