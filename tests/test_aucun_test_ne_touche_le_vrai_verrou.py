"""Cliquet jxm (24/09) : aucun test n'écrit ni n'efface le VRAI fichier de verrou de carte.

test_gui_crochet_clic.py écrivait « mesure en cours » dans /tmp/acvram-carte-0.lock.qui puis l'effaçait dans son
finally : lancé pendant une vraie prise (CI de publication, tests ciblés du chef), il détruisait l'étiquette du
détenteur réel — les « .qui disparus » du ticket jxm, vus trois soirs de suite. Un test passe par ACVRAM_VERROU vers
tmp_path. Lecteurs tolérés : conftest.py (garde de mesure) et toute ligne marquée « lecture seule du vrai verrou »
(attendre qu'une prise se termine) — jamais une écriture ni un effacement."""
import pathlib
import re

ICI = pathlib.Path(__file__).resolve().parent
VRAI = re.compile(r"/tmp/acvram-carte-\d+\.lock")


def test_aucun_test_ne_nomme_le_vrai_verrou():
    fautes = []
    for f in sorted(ICI.glob("*.py")):
        if f.name in ("conftest.py", pathlib.Path(__file__).name):
            continue
        for n, ligne in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if VRAI.search(ligne) and not ligne.lstrip().startswith("#") and "lecture seule du vrai verrou" not in ligne:
                fautes.append(f"{f.name}:{n}: {ligne.strip()[:100]}")
    assert not fautes, "vrai verrou de carte nommé dans un test :\n  " + "\n  ".join(fautes)
