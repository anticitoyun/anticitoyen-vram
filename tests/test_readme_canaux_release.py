"""Pièce 248/248b : les noms de fichiers cités dans la section « Installer » du README (FR et EN)
sont ceux que produit réellement .github/workflows/release.yml (ou tools/construire-deb.sh pour le
.deb, qui n'est pas dans le workflow — canal documenté comme hors release avant la 248b), jamais
inventés (REGLES § 7). Casse si un nom cité dérive d'un motif qui a changé dans sa source.
"""
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
README = (RACINE / "README.md").read_text(encoding="utf-8")
README_EN = (RACINE / "docs" / "README.en.md").read_text(encoding="utf-8")
RELEASE_YML = (RACINE / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CONSTRUIRE_DEB = (RACINE / "tools" / "construire-deb.sh").read_text(encoding="utf-8")

# motifs fixes attendus dans release.yml pour chaque canal (partie non variable du nom)
MOTIFS_RELEASE_YML = [
    "aur-$V.tar.gz",           # AUR : job aur
    "*.src.rpm",               # RPM/COPR : job rpm
    "acvram-${TAG#v}.flatpak", # Flatpak : job flatpak
]


def test_les_motifs_cites_existent_dans_release_yml():
    for motif in MOTIFS_RELEASE_YML:
        assert motif in RELEASE_YML, f"motif absent de release.yml : {motif}"


def test_le_job_deb_construit_et_joint_le_paquet_de_construire_deb_sh():
    assert "tools/construire-deb.sh" in RELEASE_YML, "release.yml n'appelle pas tools/construire-deb.sh"
    assert re.search(r"gh release upload.*acvram_\*_amd64\.deb", RELEASE_YML), (
        "release.yml ne joint pas acvram_*_amd64.deb à la release"
    )


def test_le_nom_du_deb_correspond_a_construire_deb_sh():
    m = re.search(r'PKG="\$STAGE/(acvram_\$\{VERSION\}_amd64)"', CONSTRUIRE_DEB)
    assert m, "tools/construire-deb.sh:14 ne produit plus le motif acvram_${VERSION}_amd64 attendu"
    for texte, nom in ((README, "README.md"), (README_EN, "docs/README.en.md")):
        assert "acvram_<version>_amd64.deb" in texte, f"nom du .deb absent ou modifié dans {nom}"
        assert re.search(r"gh release upload.*acvram_\*_amd64\.deb", RELEASE_YML), (
            "release.yml n'attache plus le .deb sous ce motif"
        )


def test_les_noms_cites_dans_le_readme_correspondent_aux_motifs_release_yml():
    correspondances = {
        "aur-<version>.tar.gz": "aur-$V.tar.gz",
        ".src.rpm": ".src.rpm",
        "acvram-<version>.flatpak": "acvram-${TAG#v}.flatpak",
    }
    for texte, nom in ((README, "README.md"), (README_EN, "docs/README.en.md")):
        for cite, attendu_release_yml in correspondances.items():
            assert cite in texte, f"{nom} ne cite pas {cite}"
            assert attendu_release_yml in RELEASE_YML, f"release.yml n'a plus {attendu_release_yml}"
