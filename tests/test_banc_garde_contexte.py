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
                     "outils", "gpu", "mesure",
                     os.environ.get("BANC_SOUS_TEST", "banc-4moteurs.py"))


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


def _banc_qui_mesure(resultat_ou_erreur):
    """Un banc dont les serveurs demarrent et dont `mesurer` fait ce qu'on veut."""
    spec = importlib.util.spec_from_file_location("banc_mesure", _BANC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.demarrer = lambda *_a, **_k: 1.0
    m.arreter = lambda *_a, **_k: None
    m.arreter_tout_sauf = lambda *_a, **_k: None
    m.modele_servi = lambda *_a, **_k: "x"
    if isinstance(resultat_ou_erreur, Exception):
        def mesurer(*_a, **_k):
            raise resultat_ou_erreur
    else:
        def mesurer(*_a, **_k):
            return resultat_ou_erreur
    m.mesurer = mesurer
    return m


def test_une_campagne_sans_aucune_mesure_echoue(tmp_path):
    """Trois campagnes de suite ont fini en code 0 sans mesurer quoi que ce soit."""
    m = _banc_qui_mesure(RuntimeError("too many values to unpack"))
    with pytest.raises(SystemExit) as sortie:
        _lancer(m, 8192, 8192, str(tmp_path / "vide.tsv"))
    assert sortie.value.code == 1, \
        "une campagne qui ne mesure rien doit echouer, pas rendre 0"


def test_une_campagne_qui_mesure_ne_signale_pas_d_echec(tmp_path):
    """L'autre sens : la garde ne doit pas condamner une campagne valide."""
    # Le faux resultat doit etre COMPLET : un dict d'energie ampute declenche
    # une KeyError attrapee par le banc, donc le chemin d'erreur — et le test
    # aurait mesure l'inverse de ce qu'il annonce.
    m = _banc_qui_mesure((12.5, 0.1, 200.0, 60.0, 200, "texte",
                          {"invalidations": "aucune", "t_s_min": 12.0,
                           "t_s_max": 13.0, "jkj_net": 55.0}))
    _lancer(m, 8192, 8192, str(tmp_path / "plein.tsv"))   # ne doit pas lever


# --- le compteur de validité, et ce qu'il comptait vraiment ------------------

_MESURE_OK = (12.5, 0.1, 200.0, 60.0, 200, "texte",
              {"invalidations": "aucune", "t_s_min": 12.0, "t_s_max": 13.0,
               "jkj_net": 55.0})
_MESURE_INVALIDE = (12.5, 0.1, 200.0, 60.0, 200, "texte",
                    {"invalidations": "bridage pendant la fenêtre : puissance",
                     "t_s_min": 12.0, "t_s_max": 13.0, "jkj_net": 55.0})


def test_une_campagne_entierement_invalidee_echoue(tmp_path):
    """Le défaut du 9/09/2026 : « 2 mesure(s) valide(s) » annoncées alors que
    la seule mesure portait « MESURE INVALIDE : bridage ».

    Le compteur incrémentait `reussies` dès qu'aucune exception n'était levée —
    **l'absence d'exception, propriété voisine de la validité.** Et la garde
    `reussies == 0` ne pouvait pas voir le défaut : elle protégeait du cas
    ABSENT, pas du cas FAUX. Le commentaire écrit juste à côté disait pourtant
    « des lignes qui existent sans rien valoir sont pires qu'un fichier vide ».
    """
    m = _banc_qui_mesure(_MESURE_INVALIDE)
    with pytest.raises(SystemExit) as sortie:
        _lancer(m, 8192, 8192, str(tmp_path / "invalide.tsv"))
    assert sortie.value.code == 1, \
        "une campagne dont toutes les mesures sont invalidees doit echouer"


def test_une_mesure_valide_laisse_partir(tmp_path):
    """L'autre sens, sans quoi le test précédent passerait sur un banc qui
    échoue toujours."""
    m = _banc_qui_mesure(_MESURE_OK)
    _lancer(m, 8192, 8192, str(tmp_path / "ok.tsv"))       # ne doit pas lever


def test_la_reprise_ne_saute_pas_une_ligne_invalidee(tmp_path):
    """Une mesure ratée était réputée faite : elle ne repartait jamais.

    L'épreuve porte sur la conséquence — la ligne est-elle REMESURÉE — et non
    sur le contenu du filtre.
    """
    tsv = tmp_path / "reprise.tsv"
    m = _banc_qui_mesure(_MESURE_INVALIDE)
    with pytest.raises(SystemExit):
        _lancer(m, 8192, 8192, str(tsv))
    avant = sum(1 for l in open(tsv) if l.startswith("M\t"))
    assert avant >= 1, "la premiere campagne doit avoir ecrit sa ligne"

    # Seconde campagne sur le même fichier : la ligne invalidée doit repartir.
    m2 = _banc_qui_mesure(_MESURE_OK)
    _lancer(m2, 8192, 8192, str(tsv))
    apres = sum(1 for l in open(tsv) if l.startswith("M\t"))
    assert apres > avant, \
        "une ligne invalidee doit etre remesuree, pas sautee comme 'faite'"
