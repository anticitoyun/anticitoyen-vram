"""La garde des contextes doit refuser la campagne — et ne refuser qu'elle.

Un test qui n'aurait vérifié que le message aurait laissé passer le défaut du
9/09/2026 : le `return` était au niveau du `if`, donc le banc sortait
**toujours**, y compris quand les contextes concordaient. Le message, lui,
était juste dans les deux cas — il parlait sur divergence, il se taisait
sinon. La garde vérifiait la propriété qu'elle annonçait et détruisait la
chose qu'elle protégeait.

Ces tests portent donc sur la CONSÉQUENCE, pas sur le texte : la campagne
part-elle, ou non. Et ils n'utilisent pas `--simuler`, qui sort au même
endroit que le défaut et le masquait entièrement.
"""
import importlib.util
import os
import sys

import pytest

# BANC_SOUS_TEST permet de pointer une copie volontairement defectueuse pour
# eprouver CES tests : sur la version au `sys.exit` mal indente, les deux
# echouent ; sur la version corrigee, les deux passent. Un test de garde qu'on
# n'a pas vu echouer sur le defaut qu'il vise n'est pas un test.
_BANC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outils", os.environ.get("BANC_SOUS_TEST", "banc-4moteurs.py"))


# Le banc attrape les exceptions de `demarrer` et poursuit avec le moteur
# suivant : une sentinelle ordinaire y serait avalée et le test croirait la
# garde bloquante. `SystemExit` hérite de `BaseException`, donc il traverse.
_PASSEE = 99


def _banc():
    spec = importlib.util.spec_from_file_location("banc_sous_test", _BANC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def demarrer_factice(*_a, **_k):
        raise SystemExit(_PASSEE)
    m.demarrer = demarrer_factice
    m.arreter = lambda *_a, **_k: None
    m.arreter_tout_sauf = lambda *_a, **_k: None
    return m


def _lancer(m, ctx_acvram, ctx_llamacpp, sortie):
    m.parc = lambda: {"M": {"acvram": ("a", "/d", ctx_acvram),
                            "llamacpp": ("b", "/d", ctx_llamacpp)}}
    sys.argv = ["banc", "--moteurs", "acvram,llamacpp", "--ctx", "8192",
                "--sortie", sortie]
    m.main()


def test_des_contextes_divergents_refusent_la_campagne(tmp_path):
    """Et le refus doit être distinguable d'une absence de travail."""
    m = _banc()
    with pytest.raises(SystemExit) as sortie:
        _lancer(m, 4096, 8192, str(tmp_path / "x.tsv"))
    assert sortie.value.code == 2, \
        "un refus doit sortir en erreur, sinon un enchaînement le prend pour un succès"


def test_des_contextes_egaux_laissent_la_campagne_partir(tmp_path):
    """LE test qui manquait : la garde ne doit pas arrêter ce qu'elle protège."""
    m = _banc()
    with pytest.raises(SystemExit) as sortie:
        _lancer(m, 8192, 8192, str(tmp_path / "y.tsv"))
    assert sortie.value.code == _PASSEE, (
        f"la garde a arrêté une campagne valide (code {sortie.value.code}) : "
        f"elle détruit ce qu'elle protège")
