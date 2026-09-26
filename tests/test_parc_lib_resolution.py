"""D1 — kimi-menu.lib.sh résolu depuis l'arbre (paquet non installé).

acvram_parc.charger() doit retourner un self.lib existant même quand
/usr/share/acvram-parc/kimi-menu.lib.sh n'est pas installé.
Le fallback cherche parc/share/ à côté de acvram_parc.py.
"""
import sys
from pathlib import Path

PARC_LIB = Path(__file__).parent.parent / "parc" / "lib"
sys.path.insert(0, str(PARC_LIB))
from acvram_parc import charger, Parc


def test_lib_existe_depuis_arbre():
    """PARC_LIB doit pointer vers un fichier existant quand lancé depuis l'arbre."""
    p = charger()
    assert p.lib is not None, "self.lib est None"
    assert p.lib.exists(), f"PARC_LIB introuvable : {p.lib}"


def test_lib_est_kimi_menu():
    """Le fichier résolu doit être kimi-menu.lib.sh."""
    p = charger()
    assert p.lib.name == "kimi-menu.lib.sh", f"nom inattendu : {p.lib.name}"


def test_lib_fallback_contourne_usr_share(monkeypatch, tmp_path):
    """Quand le chemin configuré n'existe pas, fallback vers parc/share/."""
    import acvram_parc as mod

    # chemin inexistant simulant /usr/share/acvram-parc/kimi-menu.lib.sh absent
    brut = {"chemins": {"lib": str(tmp_path / "absent.lib.sh")}}
    p = Parc(brut, source=None)
    assert p.lib is not None and p.lib.exists(), (
        f"fallback échoué : lib={p.lib}"
    )
    assert p.lib.name == "kimi-menu.lib.sh"
    # Le fallback pointe dans parc/share/, pas dans /usr/share
    assert "usr" not in str(p.lib)
