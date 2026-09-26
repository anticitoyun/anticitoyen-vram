"""Pièce 241 : la CI qui produit les noyaux précompilés, le manifeste Flathub qui les embarque et le chargeur (240) parlent
du MÊME chemin et de la MÊME empreinte — à sec. Casse si le job attend un chemin (`build/noyaux`) ou une disposition
(`<src_sha16>/acvram_kernels.so` + `empreinte.json`) que `compiler_precompile`/`_ecrire_empreinte` ne produit pas, si le
manifeste n'expose pas `ACVRAM_KERNELS_PRECOMPILES` sur le répertoire installé, ou si les architectures forcées ne donnent
pas les `-gencode` attendus."""
import hashlib
import os
import pathlib
import re

import pytest
import torch
import yaml

from acvram import kernels
from acvram.kernels import precompiles as CLI

RACINE = pathlib.Path(__file__).resolve().parents[1]
FLATHUB = RACINE / "packaging" / "flathub"
MANIFESTE = FLATHUB / "io.github.anticitoyen.acvram.yml"
JOB = FLATHUB / "job-noyaux-precompiles.yml"
RELEASE = RACINE / ".github" / "workflows" / "release.yml"


def _texte_job() -> str:
    """Le fragment de job, ou release.yml s'il porte déjà le job (poste3) — le second prime."""
    if RELEASE.is_file() and "acvram.kernels.precompiles" in RELEASE.read_text(encoding="utf-8"):
        return RELEASE.read_text(encoding="utf-8")
    return JOB.read_text(encoding="utf-8")


def test_le_job_produit_le_dossier_que_le_manifeste_embarque():
    m = yaml.safe_load(MANIFESTE.read_text(encoding="utf-8"))
    mod = next(x for x in m["modules"] if isinstance(x, dict) and x["name"] == "acvram-noyaux")
    src = mod["sources"][0]["path"]                                      # ../../build/noyaux (relatif au manifeste)
    dossier = os.path.normpath(os.path.join("packaging/flathub", src))
    assert dossier == CLI.DOSSIER_DEFAUT == "build/noyaux", (dossier, CLI.DOSSIER_DEFAUT)
    job = _texte_job()
    assert re.search(r"python(3)? -m acvram\.kernels\.precompiles --dossier build/noyaux --archs 12\.0", job), "le job n'appelle pas le producteur sur build/noyaux avec sm_120"
    assert "path: build/noyaux" in job, "le job n'archive pas build/noyaux"
    # le répertoire installé est celui que le chargeur (240) lit
    dest = [c for c in mod["build-commands"] if "cp -r noyaux/." in c][0].split()[-1].rstrip("/")
    envs = [a for a in m["finish-args"] if a.startswith("--env=ACVRAM_KERNELS_PRECOMPILES=")]
    assert envs and envs[0].split("=", 2)[2] == dest, (envs, dest)
    assert not any(a.startswith("--env=ACVRAM_KERNEL_CACHE=") for a in m["finish-args"]), "KERNEL_CACHE ferait compiler (ninja) : c'est PRECOMPILES qu'il faut"


def test_ce_que_la_ci_ecrit_est_ce_que_le_chargeur_relit(tmp_path):
    """Aller-retour `_ecrire_empreinte` → `_precompile_utilisable` sur un faux .so qui porte l'empreinte : accepté ; le même
    fichier sous une autre source : refusé (c'est le contrat que la CI et le Flatpak doivent tenir)."""
    src = b"// source 241\n"; src_sha = hashlib.sha256(src).hexdigest(); src_hash = "00ab12cd34ef5678"
    faux = tmp_path / "faux.so"; faux.write_bytes(b"ELF" + int(src_hash, 16).to_bytes(8, "little"))
    cand = kernels._ecrire_empreinte(str(tmp_path / "noyaux" / src_sha[:16]), str(faux), src_sha, src_hash, ["sm_120f", "sm_86"])
    assert sorted(os.listdir(cand)) == ["acvram_kernels.so", "empreinte.json"]
    so, raison = kernels._precompile_utilisable(str(tmp_path / "noyaux"), src, {(12, 0), (8, 6)}, torch.__version__, torch.version.cuda)
    assert so and raison == "précompilé"
    so2, raison2 = kernels._precompile_utilisable(str(tmp_path / "noyaux"), src + b"x", {(12, 0)}, torch.__version__, torch.version.cuda)
    assert so2 is None and "aucun précompilé" in raison2


@pytest.mark.parametrize("texte,attendu", [("12.0", [(12, 0)]), ("12.0,8.9,8.6", [(12, 0), (8, 9), (8, 6)]), ("", [])])
def test_archs_depuis_texte(texte, attendu):
    assert kernels.archs_depuis_texte(texte) == attendu


def test_archs_illisibles_levent():
    with pytest.raises(ValueError):
        kernels.archs_depuis_texte("sm_120")


def test_les_archs_forcees_donnent_les_gencode_famille_sans_carte():
    flags = kernels._arch_flags(nvcc_ver=(13, 0), archs_forcees=[(12, 0), (8, 6)])
    codes = kernels._archs_des_drapeaux(flags)
    assert codes == ["sm_120f", "sm_86"], flags                         # famille pour ≥ 10 (cuda_fp4 natif), générique en dessous
    assert any("compute_120,code=compute_120" in f for f in flags), "repli PTX de la plus haute architecture attendu"
    job = _texte_job()
    assert re.search(r"--archs 12\.0(,8\.9)?(,8\.6)?", job)
